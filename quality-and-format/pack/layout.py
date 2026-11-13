"""affine_min v0: explicit little-endian fields, LSB-first codes, aligned K then V."""
import math
import hashlib
from dataclasses import dataclass
import numpy as np
import torch
from quant.golden import FORMATS, DTYPE, encode, decode

BITS = (2, 4, 8)
GROUPS = (32, 64, 128)
PAIRS = ((8,8),(4,4),(2,2),(4,2),(8,4),(8,2),(2,4))


@dataclass(frozen=True)
class Config:
    b_K: int
    g_K: int
    b_V: int
    g_V: int
    scale_fmt: str = "fp16"


def grid(length):
    for bk, bv in PAIRS:
        for gk in GROUPS:
            if gk <= length:
                for gv in GROUPS:
                    yield Config(bk, gk, bv, gv)


def align(n, burst):
    return (n+burst-1)//burst*burst


def layout(config, n, dim, burst):
    if not (0 <= n <= 255 and dim > 0 and burst in (32,64)):
        raise ValueError("geometry")
    code_k, code_v = ((n*dim*b+7)//8 for b in (config.b_K, config.b_V))
    meta_k = 4*dim*math.ceil(n/config.g_K)
    meta_v = 4*n*math.ceil(dim/config.g_V)
    k, v = align(code_k+meta_k,burst), align(code_v+meta_v,burst)
    return dict(K_codes=code_k,V_codes=code_v,K_meta=meta_k,V_meta=meta_v,
                K_pad=k-code_k-meta_k,V_pad=v-code_v-meta_v,K_bytes=k,V_bytes=v,
                B_dram=k+v,b_eff=8*(k+v)/(2*n*dim) if n else None,
                compression=2*align(2*n*dim,burst)/(k+v) if n else None)


def descriptor(config, n, state, base, dim, burst):
    spec = layout(config,n,dim,burst)
    if state not in (0,1,2) or (state == 0) != (n == 0):
        raise ValueError("invalid page state")
    if base < 0 or base >= 2**32 or base % burst or base+spec["B_dram"] > 2**32:
        raise ValueError("invalid address")
    fields = [BITS.index(config.b_K), BITS.index(config.b_V), GROUPS.index(config.g_K),
              GROUPS.index(config.g_V), FORMATS.index(config.scale_fmt)]
    word = sum(v << (2*i) for i,v in enumerate(fields)) | state<<14 | n<<16 | base<<32
    return word.to_bytes(8,"little")


def parse_descriptor(data, dim, burst):
    if len(data)!=8:
        raise ValueError("descriptor length")
    word = int.from_bytes(data,"little")
    fields = [(word >> (2*i)) & 3 for i in range(5)]
    if 3 in fields or word & 0xff003c00:
        raise ValueError("reserved field or unsupported compute mode")
    bk,bv,gk,gv,fmt=fields
    config=Config(BITS[bk],GROUPS[gk],BITS[bv],GROUPS[gv],FORMATS[fmt])
    n,state,base = (word>>16)&255, (word>>14)&3, word>>32
    descriptor(config,n,state,base,dim,burst)
    return config,n,state,base


def code_bytes(codes,bits):
    vals=codes.reshape(-1).numpy()
    per=8//bits
    vals=np.pad(vals,(0,(-len(vals))%per)).reshape(-1,per)
    return np.bitwise_or.reduce(vals << (np.arange(per,dtype=np.uint8)*bits),axis=1).tobytes()


def segment(enc,bits,burst):
    # A uint16 E8M0 field naturally writes the required zero spacer.
    scale=enc.scales.byte().numpy() if enc.scales.dtype==torch.uint8 else enc.scales.view(torch.int16).numpy()
    fields=np.stack([scale,enc.offsets.view(torch.int16).numpy()],axis=-1).astype("<u2")
    data=code_bytes(enc.codes,bits)+fields.tobytes()
    return data.ljust(align(len(data),burst),b"\0")


@dataclass
class Packed:
    descriptor: bytes
    data: bytes
    stats: dict

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def pack(k,v,c,burst=64,base=0,state=2,page="anonymous"):
    if k.ndim!=2 or k.shape!=v.shape:
        raise ValueError("K/V geometry")
    n,dim=k.shape
    desc=descriptor(c,n,state,base,dim,burst)
    if not n:
        return Packed(desc,b"",{})
    encoded=[encode(k,"K",c.b_K,c.g_K,c.scale_fmt,page+"/K"),
             encode(v,"V",c.b_V,c.g_V,c.scale_fmt,page+"/V")]
    return Packed(desc,segment(encoded[0],c.b_K,burst)+segment(encoded[1],c.b_V,burst),
                  {d:e.stats for d,e in zip(("K","V"),encoded)})


def unpack(desc,data,dim,burst=64):
    c,n,_,_=parse_descriptor(desc,dim,burst)
    spec=layout(c,n,dim,burst)
    if len(data)!=spec["B_dram"]:
        raise ValueError("payload length")
    if not n:
        return tuple(torch.empty((0,dim),dtype=torch.bfloat16) for _ in range(2))
    result=[]
    for direction,offset,b,g in [("K",0,c.b_K,c.g_K),("V",spec["K_bytes"],c.b_V,c.g_V)]:
        length=spec[direction+"_codes"]
        buf=np.frombuffer(data,dtype=np.uint8,count=length,offset=offset)
        lanes=((buf[:,None] >> (b*np.arange(8//b))) & ((1<<b)-1)).ravel()
        if lanes[n*dim:].any():
            raise ValueError("nonzero unused code bits")
        q=torch.tensor(lanes[:n*dim].reshape(n,dim),dtype=torch.uint8)
        rows=dim if direction=="K" else n
        meta=spec[direction+"_meta"]
        fields=np.frombuffer(data,"<u2",count=meta//2,offset=offset+length).reshape(rows,-1,2)
        if any(data[offset+length+meta:offset+spec[direction+"_bytes"]]):
            raise ValueError("nonzero alignment padding")
        scales=torch.from_numpy(fields[:,:,0].astype(np.int16))
        offsets=torch.from_numpy(fields[:,:,1].astype(np.int16)).view(torch.float16)
        if c.scale_fmt=="e8m0":
            if (fields[:,:,0]>255).any():
                raise ValueError("nonzero E8M0 spacer")
            scales=scales.byte()
        else:
            scales=scales.view(DTYPE[c.scale_fmt])
        result.append(decode(q,scales,offsets,direction,g,c.scale_fmt))
    return tuple(result)
