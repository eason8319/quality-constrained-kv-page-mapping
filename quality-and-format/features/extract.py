"""Page-local reductions, supplied only the closed page's statistics."""
import torch
from quant.golden import extrema


def coarse_extrema(fine_min, fine_max, ratio):
    if fine_min.shape != fine_max.shape or ratio < 1:
        raise ValueError("unaligned groups")
    return extrema(fine_min, ratio)[0], extrema(fine_max, ratio)[1]


def query_amplitude(page_q, kv_head, group_heads):
    """page_q contains post-RoPE Q with [query_head, valid_token, channel]."""
    heads = page_q[kv_head*group_heads:(kv_head+1)*group_heads]
    if len(heads) != group_heads or heads.shape[1] == 0:
        raise ValueError("incomplete GQA group or empty page")
    return heads.float().abs().mean(dim=(0,1))
