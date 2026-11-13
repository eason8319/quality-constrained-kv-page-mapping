import torch


def targets(mask, close):
    return [j for j,score in enumerate(mask) if score and j-1>close]


def bucket(page,close,t,window=128):
    if t<=close:
        raise ValueError("query has not read the closed page")
    return "sink" if page==0 else "recent" if t-close<=window else "mid"


def reconstruction(x,y):
    error=(y.float()-x.float()).abs()
    maxima=error.amax(0)
    return dict(relative=(error.square().sum()/(x.float().square().sum()+1e-12)).item(),
                channel_linf_max=maxima.max().item(),channel_linf_p95=maxima.quantile(.95).item())


def local_errors(fp_weights,weights,fp_out,out):
    l1=(weights.float()-fp_weights.float()).abs().sum(-1)
    rel=(out.float()-fp_out.float()).norm(dim=-1)/(fp_out.float().norm(dim=-1)+1e-12)
    return dict(attn_l1_heads=l1.tolist(),attn_l1_mean=l1.mean().item(),attn_l1_max=l1.max().item(),
                out_rel_heads=rel.tolist(),out_rel_mean=rel.mean().item(),out_rel_max=rel.max().item())


def log_probs(model,hidden,positions,labels):
    rows=hidden[0,positions]
    # Each projection contains only the selected targets.
    parts=[]
    for chunk,ids in zip(rows.split(16),labels.split(16)):
        logits=model.lm_head(chunk).float()
        parts.append(logits.log_softmax(-1).gather(1,ids[:,None]).flatten().cpu())
    return torch.cat(parts)
