"""E0 eight-D_cal numerical controls. No follow-on layers use the changed attention output."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM,AutoTokenizer
from eval.fork import native_module,local_rows
from eval.metrics import reconstruction
from eval.task_inputs import gsm_prompt
from features.extract import coarse_extrema
from quant.golden import FORMATS,encode,decode,encode_scale,decode_scale,expand,orient
from pack.layout import Config,pack,unpack,segment,layout

RUN=Path("runs/p0-affine_min-all-20261113-01")
OUT=RUN/"p0b"


def recode(x,d,fmt):
    fine=encode(x,d,2,32,fmt)
    lo,hi=coarse_extrema(fine.minimum,fine.maximum,2)
    original=encode(x,d,2,64,fmt)
    assert torch.equal(original.minimum,lo) and torch.equal(original.maximum,hi)
    field=encode_scale((hi-lo)/3,fmt)
    offset=lo.half()
    assert torch.equal(original.scales,field) and torch.equal(original.offsets,offset)
    old=orient(fine.codes,d).float()
    width=old.shape[1]
    linear=expand(decode_scale(fine.scales,fmt),width,32)*old+expand(fine.offsets.float(),width,32)
    steps=expand(decode_scale(field,fmt),width,64)
    q=((linear-expand(offset.float(),width,64))/torch.where(steps==0,1,steps)).round().clamp(0,3)
    codes=orient(q.masked_fill(steps==0,0).byte(),d).contiguous()
    original.codes=codes
    return original

@torch.inference_mode()
def main():
    model_path=Path("data/raw/sources/llama")
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,attn_implementation="eager",dtype=torch.bfloat16,
                                                device_map={"":"cuda:0"},local_files_only=True).eval()
    keys=Path("data/splits/gsm8k/D_cal.txt").read_text().splitlines()[:8]
    raw=pq.read_table(next(Path("data/raw/sources/gsm8k/main").glob("train*.parquet")))
    spec=next(Path("data/raw/p0b-upstream").glob("lm-*/lm_eval/tasks/gsm8k/gsm8k-cot.yaml"))
    examples=yaml.safe_load(spec.read_text())["fewshot_config"]["samples"]
    mod=native_module(model)
    real=mod.eager_attention_forward
    base_model=yaml.safe_load(Path("configs/models.yaml").read_text())["models"][0]
    with (OUT/"e0_controls.jsonl").open("w") as out:
        for key in keys:
            doc=raw.slice(int(key.split("/")[-1]),1).to_pylist()[0]
            # Use 256 prompt tokens ending at the actual question so each calibration input is distinct.
            ids=tok(gsm_prompt(doc["question"],examples),return_tensors="pt").input_ids[:,-256:].cuda()
            obs={}
            def capture(attn,q,k,v,mask,scaling,**kw):
                y,w=real(attn,q,k,v,mask,scaling,**kw)
                if attn.layer_idx==8:
                    for t in [128,191,255]:
                        obs[t]=(q[:,4:8,t:t+1].clone(),k[:,1:2,:t+1].clone(),v[:,1:2,:t+1].clone(),
                                w[0,4:8,t,:t+1].clone(),y[0,t,4:8].clone())
                return y,w
            with patch.object(mod,"eager_attention_forward",capture):
                model.model(input_ids=ids,use_cache=False)
            raw_k,raw_v=[v[0,0,64:128].cpu() for v in obs[128][1:3]]
            for fmt in FORMATS:
                for group,mode in [(32,"direct"),(64,"direct"),(64,"recode_32_to_64")]:
                    c=Config(2,group,2,group,fmt)
                    p=pack(raw_k,raw_v,c)
                    data=p.data
                    if mode.startswith("recode"):
                        data=b"".join(segment(recode(x,d,fmt),2,64) for x,d in zip([raw_k,raw_v],"KV"))
                    new=unpack(p.descriptor,data,128)
                    errors=local_rows(obs,new,64,4)
                    record=dict(protocol_version="affine_min",git_revision="3ca9091fde655784434a8570db7d9812831f16fd",
                        implementation_sha256=hashlib.sha256((OUT/"implementation.sha256").read_bytes()).hexdigest(),
                        quant_tests_sha256=hashlib.sha256((RUN/"quant_tests.json").read_bytes()).hexdigest(),
                        model=base_model["id"],model_revision=base_model["revision"],task="gsm8k",
                        sample_id=key,split="D_cal",layer=8,kv_head=1,page_index=1,distance_d=1,bucket="recent",
                        config=asdict(c),mode=mode,B_dram=len(data),b_eff=layout(c,64,128,64)["b_eff"],
                        input_sha256=hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest(),
                        byte_sha256=hashlib.sha256(data).hexdigest(),group_counts=p.stats)
                    rk,rv=[reconstruction(x,y) for x,y in zip([raw_k,raw_v],new)]
                    out.write(json.dumps(dict(record,intervention="recon",rel_k=rk["relative"],rel_v=rv["relative"],K=rk,V=rv))+"\n")
                    for t,error in errors.items():
                        out.write(json.dumps(dict(record,intervention="local_attn",query=t,distance_d=t-127,
                                                  bucket="recent",**error))+"\n")
    print("PASS E0",len(keys),"D_cal inputs",flush=True)

if __name__=="__main__":
    main()
