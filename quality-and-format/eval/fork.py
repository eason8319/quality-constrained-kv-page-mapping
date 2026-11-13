"""Replay BF16 forks from clean closed prefixes, using the model's native eager GQA."""
import copy
import hashlib
import importlib
import json
from dataclasses import asdict
from unittest.mock import patch
import torch
from pack.layout import pack,unpack,layout
from eval.metrics import targets,bucket,reconstruction,local_errors,log_probs


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token_digest(ids):
    return digest(ids.detach().cpu().numpy().astype("<i8").tobytes())


def cache_page(cache,layer,head,start,end):
    entry=cache.layers[layer]
    return tuple(x[0,head,start:end] for x in (entry.keys,entry.values))


def native_module(model):
    return importlib.import_module(type(model.model.layers[0].self_attn).__module__)


def prefill(model,ids,end):
    return model.model(input_ids=ids[:,:end],use_cache=True).past_key_values


def replacements(model,ids,pages,length,layer,head,config):
    # Joint pages are encoded independently from clean FP snapshots at each close.
    clean={}
    for page in pages:
        end=(page+1)*length
        cache=prefill(model,ids,end)
        k,v=cache_page(cache,layer,head,end-length,end)
        original=(k.cpu(),v.cpu())
        payload=pack(*original,config,page=f"layer{layer}/head{head}/page{page}")
        restored=unpack(payload.descriptor,payload.data,k.shape[-1])
        clean[page]=(payload,original,restored)
        if page==min(pages):
            first=cache
    return first,clean


def replay(model,ids,snapshot,clean,length,layer,head,selected,branch,direction="KV",capture=False):
    cache=copy.deepcopy(snapshot)
    module=native_module(model)
    real=module.eager_attention_forward
    positions=sorted(clean)
    start=(positions[0]+1)*length
    parts=[]
    observations={}
    first_read={}
    for ix,page in enumerate(positions):
        payload,original,restored=clean[page]
        left=page*length
        begin=left+length
        stop=(positions[ix+1]+1)*length if ix+1<len(positions) else ids.shape[1]-1
        slots=cache_page(cache,layer,head,left,begin)
        writeback = tuple(x.clone() for x in slots) if branch=="null" else restored
        if branch!="fp":
            for d,dst,src in zip("KV",slots,writeback):
                if d in direction:
                    dst.copy_(src)
        def observed(attn,q,k,v,mask,scaling,**kwargs):
            out,weights=real(attn,q,k,v,mask,scaling,**kwargs)
            if attn.layer_idx==layer:
                # Check cache bytes seen by the attention implementation on the first suffix call.
                for d,actual,fp,quant in zip("KV",(k,v),original,restored):
                    view=actual[0,head,left:begin].cpu()
                    if len(positions)==1 or branch=="quant" and d in direction or ix==0:
                        expected=quant if branch=="quant" and d in direction else fp
                        assert torch.equal(view.view(torch.int16),expected.view(torch.int16))
                first_read[page]=begin
                if capture:
                    a=head*attn.num_key_value_groups
                    b=a+attn.num_key_value_groups
                    for t in sorted({begin,min(stop-1,begin+127),min(stop-1,begin+128),stop-1}):
                        row=t-begin
                        observations[t]=(q[:,a:b,row:row+1].clone(),k[:,head:head+1,:t+1].clone(),
                            v[:,head:head+1,:t+1].clone(),weights[0,a:b,row,:t+1].clone(),out[0,row,a:b].clone())
            return out,weights
        assert begin<stop
        with patch.object(module,"eager_attention_forward",observed):
            hidden=model.model(input_ids=ids[:,begin:stop],past_key_values=cache,use_cache=True).last_hidden_state
        js=[j for j in selected if begin<=j-1<stop]
        if js:
            parts.append(log_probs(model,hidden,[j-1-begin for j in js],ids[0,js]))
    return torch.cat(parts),observations,first_read


def local_rows(observations,replacement,left,group_heads):
    k_new,v_new=replacement
    rows={}
    for t,(q,k,v,orig_w,orig_out) in observations.items():
        k=k.clone()
        v=v.clone()
        def compute():
            logits=(q @ k.transpose(-2,-1))*(q.shape[-1]**-.5)
            weights=logits.softmax(-1,dtype=torch.float32).to(q.dtype)
            return weights[0,:,0],(weights@v)[0,:,0]
        w0,o0=compute()
        # Native causal computation and explicit one-row reconstruction must agree within BF16 rounding.
        torch.testing.assert_close(w0,orig_w,rtol=.012,atol=.001)
        torch.testing.assert_close(o0,orig_out,rtol=.025,atol=.006)
        k[0,0,left:left+len(k_new)]=k_new
        v[0,0,left:left+len(v_new)]=v_new
        w,o=compute()
        rows[t]=local_errors(w0,w,o0,o)
    return rows
