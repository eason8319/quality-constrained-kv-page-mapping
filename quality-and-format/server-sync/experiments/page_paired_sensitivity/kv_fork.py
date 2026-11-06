"""Mandatory fork: close page → pack once → quant remaining shared prefix → Q fork."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .quantize import PackedPage, materialize_page, pack_page_kv
from .prefix_builder import SmokePrefix


Intervene = Literal["k", "v", "kv", "none"]
Mode = Literal["fp", "quant", "zero_pert"]


@dataclass
class TrajectoryResult:
    name: str
    mode: Mode
    intervene: Intervene
    page: str  # "A" | "B" | "-"
    question: str  # "jia" | "yi"
    role: Optional[str]
    S: float
    S_key: float
    nll_fp: float
    nll_run: float
    page_hash: Optional[str]
    t_key: int
    t_page_end: Optional[int]
    d: Optional[int]
    answer_logprobs: List[float]
    device: str



def _layer_kv(layer_obj):
    """Extract (k,v) from a Cache layer object without bool(Tensor)."""
    for k_name, v_name in (
        ("keys", "values"),
        ("key_cache", "value_cache"),
        ("key", "value"),
    ):
        if hasattr(layer_obj, k_name) and hasattr(layer_obj, v_name):
            k = getattr(layer_obj, k_name)
            v = getattr(layer_obj, v_name)
            if k is not None and v is not None:
                return k, v
    raise TypeError(f"Cannot read KV from layer type {type(layer_obj)} attrs={[a for a in dir(layer_obj) if not a.startswith('_')]}")


def _set_layer_kv(layer_obj, k, v):
    for k_name, v_name in (
        ("keys", "values"),
        ("key_cache", "value_cache"),
        ("key", "value"),
    ):
        if hasattr(layer_obj, k_name) and hasattr(layer_obj, v_name):
            setattr(layer_obj, k_name, k)
            setattr(layer_obj, v_name, v)
            return
    raise TypeError(f"Cannot write KV into layer type {type(layer_obj)}")


def _to_legacy(past):
    if past is None:
        return None
    if isinstance(past, (tuple, list)) and len(past) > 0 and isinstance(past[0], (tuple, list)):
        return tuple(past)
    # DynamicCache with layers
    if hasattr(past, "layers") and past.layers is not None and len(past.layers) > 0:
        out = []
        for layer_obj in past.layers:
            out.append(_layer_kv(layer_obj))
        return tuple(out)
    if hasattr(past, "key_cache") and past.key_cache is not None and len(past.key_cache) > 0:
        return tuple((past.key_cache[i], past.value_cache[i]) for i in range(len(past.key_cache)))
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    raise TypeError(f"Unsupported cache type {type(past)} dir={[a for a in dir(past) if not a.startswith('_')]}")


def _get_kv(past, layer: int):
    if isinstance(past, (tuple, list)):
        return past[layer]
    if hasattr(past, "layers") and past.layers is not None and len(past.layers) > layer:
        return _layer_kv(past.layers[layer])
    if hasattr(past, "key_cache") and past.key_cache is not None and len(past.key_cache) > layer:
        return past.key_cache[layer], past.value_cache[layer]
    raise TypeError(f"Cannot get layer {layer} from {type(past)}")


def _set_kv_slice(past, layer: int, kv_head: int, t0: int, t1: int, k_page: torch.Tensor, v_page: torch.Tensor):
    """Write [t0:t1) for one kv_head into past cache. k_page/v_page: [L, D]."""
    # Prefer mutating DynamicCache layers in place so subsequent forwards keep working
    if hasattr(past, "layers") and past.layers is not None and len(past.layers) > layer:
        k, v = _layer_kv(past.layers[layer])
        k = k.clone()
        v = v.clone()
        k[:, kv_head, t0:t1, :] = k_page.to(dtype=k.dtype, device=k.device)
        v[:, kv_head, t0:t1, :] = v_page.to(dtype=v.dtype, device=v.device)
        _set_layer_kv(past.layers[layer], k, v)
        return past

    # Legacy tuple path
    if isinstance(past, (tuple, list)):
        layers = list(past)
        k, v = layers[layer]
        k = k.clone()
        v = v.clone()
        k[:, kv_head, t0:t1, :] = k_page.to(dtype=k.dtype, device=k.device)
        v[:, kv_head, t0:t1, :] = v_page.to(dtype=v.dtype, device=v.device)
        layers[layer] = (k, v)
        return tuple(layers)

    # key_cache path
    if hasattr(past, "key_cache") and past.key_cache is not None and len(past.key_cache) > layer:
        k = past.key_cache[layer].clone()
        v = past.value_cache[layer].clone()
        k[:, kv_head, t0:t1, :] = k_page.to(dtype=k.dtype, device=k.device)
        v[:, kv_head, t0:t1, :] = v_page.to(dtype=v.dtype, device=v.device)
        past.key_cache[layer] = k
        past.value_cache[layer] = v
        return past

    raise TypeError(f"Cannot set slice on cache type {type(past)}")


def _forward_chunk(model, input_ids: torch.Tensor, past=None, use_cache: bool = True):
    # Keep DynamicCache as-is when provided; only convert for reads via _get_kv
    out = model(input_ids=input_ids, past_key_values=past, use_cache=use_cache, return_dict=True)
    return out


def _teacher_force_nll(
    model,
    past,
    prompt_ids: List[int],
    answer_ids: List[int],
    device: torch.device,
) -> Tuple[List[float], float, object]:
    """Continue from past; feed prompt then teacher-force answer; return per-token NLL (nats)."""
    nlls: List[float] = []
    if not prompt_ids:
        raise RuntimeError("empty prompt")
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    out = _forward_chunk(model, ids, past=past, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    for tid in answer_ids:
        logprob = F.log_softmax(logits.float(), dim=-1)[0, tid]
        nlls.append(float(-logprob.item()))
        tok = torch.tensor([[tid]], device=device, dtype=torch.long)
        out = _forward_chunk(model, tok, past=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    mean_nll = sum(nlls) / max(len(nlls), 1)
    return nlls, mean_nll, past


def run_fp_trajectory(
    model,
    sp: SmokePrefix,
    *,
    question: Literal["jia", "yi"],
    device: torch.device,
    chunk_at: Optional[int] = None,
) -> TrajectoryResult:
    """
    Full-precision trajectory.
    If chunk_at is set, forward shared[:chunk_at] then shared[chunk_at:] to match the
    mandatory fork schedule (avoids SDPA split noise when used as S baseline for forked runs).
    """
    q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
    a_ids = sp.ans_jia_ids if question == "jia" else sp.ans_yi_ids
    key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi

    if chunk_at is None:
        shared = torch.tensor([sp.shared_ids], device=device, dtype=torch.long)
        out = _forward_chunk(model, shared, past=None, use_cache=True)
        past = out.past_key_values
    else:
        first = torch.tensor([sp.shared_ids[:chunk_at]], device=device, dtype=torch.long)
        out = _forward_chunk(model, first, past=None, use_cache=True)
        past = out.past_key_values
        rem = sp.shared_ids[chunk_at:]
        if rem:
            rem_t = torch.tensor([rem], device=device, dtype=torch.long)
            out = _forward_chunk(model, rem_t, past=past, use_cache=True)
            past = out.past_key_values

    nlls, mean_nll, _ = _teacher_force_nll(model, past, q_ids, a_ids, device)
    t_key = len(sp.shared_ids) + len(q_ids) + key_off
    return TrajectoryResult(
        name=f"fp_{question}" + (f"_chunk{chunk_at}" if chunk_at is not None else ""),
        mode="fp",
        intervene="none",
        page="-",
        question=question,
        role=None,
        S=0.0,
        S_key=0.0,
        nll_fp=mean_nll,
        nll_run=mean_nll,
        page_hash=None,
        t_key=t_key,
        t_page_end=None,
        d=None,
        answer_logprobs=[-x for x in nlls],
        device=str(device),
    )


def run_fork_trajectory(
    model,
    sp: SmokePrefix,
    *,
    target_page: Literal["A", "B"],
    question: Literal["jia", "yi"],
    intervene: Intervene,
    mode: Mode,
    layer: int,
    kv_head: int,
    L_p: int,
    bits: int,
    group: int,
    device: torch.device,
    fp_nlls: List[float],
) -> Tuple[TrajectoryResult, PackedPage]:
    """
    Mandatory fork order:
    1) FP to page close + pack once
    2) branch: rewrite page bytes in cache
    3) continue remaining shared prefix under intervention
    4) then question
    """
    page_span = sp.page_a if target_page == "A" else sp.page_b
    t0, t1 = page_span
    t_page_end = t1 - 1
    q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
    a_ids = sp.ans_jia_ids if question == "jia" else sp.ans_yi_ids
    key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi
    role_map = {
        ("A", "jia"): "需要该页",
        ("A", "yi"): "无需该页",
        ("B", "jia"): "无需该页",
        ("B", "yi"): "需要该页",
    }

    # Step 1: FP through page close
    ids_to_close = torch.tensor([sp.shared_ids[:t1]], device=device, dtype=torch.long)
    out = _forward_chunk(model, ids_to_close, past=None, use_cache=True)
    past = out.past_key_values

    # Debug cache once if needed
    k, v = _get_kv(past, layer)
    k_page = k[0, kv_head, t0:t1, :].detach()
    v_page = v[0, kv_head, t0:t1, :].detach()

    if mode == "zero_pert":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=(0 if target_page == "A" else 1),
            t_start=t0, t_end=t1, bits=bits, group=group,
            intervene=intervene if intervene != "none" else "k",
        )
        mat_mode = "zero_pert"
    elif mode == "quant":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=(0 if target_page == "A" else 1),
            t_start=t0, t_end=t1, bits=bits, group=group, intervene=intervene,
        )
        mat_mode = "quant"
    else:
        raise ValueError(mode)

    page_hash = packed.content_hash()

    # Step 2: fork — write fixed bytes into cache at close point
    k_mat, v_mat = materialize_page(packed, mode=mat_mode, device=k.device, dtype=k.dtype)
    if mode == "zero_pert":
        # Full intervention path: write back the exact original GPU page tensors (no CPU round-trip).
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_page, v_page)
    else:
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)

    # Step 3: remaining shared prefix under intervention
    rem = sp.shared_ids[t1:]
    if rem:
        rem_t = torch.tensor([rem], device=device, dtype=torch.long)
        out = _forward_chunk(model, rem_t, past=past, use_cache=True)
        past = out.past_key_values
        # Closed pages are not rewritten by append-only cache; re-assert only for quant
        # (avoid extra copy noise on zero-pert identity path)
        if mode == "quant":
            past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)

    # Step 4: question
    nlls, mean_nll, _ = _teacher_force_nll(model, past, q_ids, a_ids, device)
    deltas = [qn - fn for qn, fn in zip(nlls, fp_nlls)]
    S = sum(deltas) / max(len(deltas), 1)
    S_key = deltas[key_off]
    t_key = len(sp.shared_ids) + len(q_ids) + key_off
    d = t_key - t_page_end

    return TrajectoryResult(
        name=f"{mode}_{target_page}_{intervene}_{question}",
        mode=mode,
        intervene=intervene,
        page=target_page,
        question=question,
        role=role_map[(target_page, question)],
        S=S,
        S_key=S_key,
        nll_fp=sum(fp_nlls) / max(len(fp_nlls), 1),
        nll_run=mean_nll,
        page_hash=page_hash,
        t_key=t_key,
        t_page_end=t_page_end,
        d=d,
        answer_logprobs=[-x for x in nlls],
        device=str(device),
    ), packed


def clone_past(past):
    """Deep-clone DynamicCache or legacy tuple so Q甲/Q乙 can fork after shared rem."""
    if past is None:
        return None
    if hasattr(past, "layers") and past.layers is not None:
        try:
            from transformers import DynamicCache
            new = DynamicCache()
            # Prefer official copy if available
            if hasattr(past, "key_cache"):
                pass
        except Exception:
            DynamicCache = None  # type: ignore
        # Manual: clone each layer's K/V into a fresh DynamicCache via update
        try:
            from transformers import DynamicCache as DC
            new = DC()
            for i, layer_obj in enumerate(past.layers):
                k, v = _layer_kv(layer_obj)
                new.update(k.clone(), v.clone(), i)
            return new
        except Exception:
            pass
        # Fallback legacy
        return tuple((k.clone(), v.clone()) for k, v in _to_legacy(past))
    if isinstance(past, (tuple, list)):
        return tuple((k.clone(), v.clone()) for k, v in past)
    return _to_legacy(past)


def run_fork_pair(
    model,
    sp: SmokePrefix,
    *,
    target_page: Literal["A", "B"],
    intervene: Intervene,
    mode: Mode,
    layer: int,
    kv_head: int,
    L_p: int,
    bits: int,
    group: int,
    device: torch.device,
    fp_nlls_jia: List[float],
    fp_nlls_yi: List[float],
) -> Tuple[TrajectoryResult, TrajectoryResult, PackedPage, bool]:
    """
    Pack once at page close, run rem once, then fork Q甲 and Q乙 from cloned cache.
    Returns (res_jia, res_yi, packed, hash_equal) — hash_equal is True by construction
    (same packed bytes); still reported for gate logging.
    """
    page_span = sp.page_a if target_page == "A" else sp.page_b
    t0, t1 = page_span
    t_page_end = t1 - 1
    role_map = {
        ("A", "jia"): "需要该页",
        ("A", "yi"): "无需该页",
        ("B", "jia"): "无需该页",
        ("B", "yi"): "需要该页",
    }

    ids_to_close = torch.tensor([sp.shared_ids[:t1]], device=device, dtype=torch.long)
    out = _forward_chunk(model, ids_to_close, past=None, use_cache=True)
    past = out.past_key_values
    k, v = _get_kv(past, layer)
    k_page = k[0, kv_head, t0:t1, :].detach()
    v_page = v[0, kv_head, t0:t1, :].detach()

    if mode == "zero_pert":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=(0 if target_page == "A" else 1),
            t_start=t0, t_end=t1, bits=bits, group=group,
            intervene=intervene if intervene != "none" else "k",
        )
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_page, v_page)
        k_mat = v_mat = None
    elif mode == "quant":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=(0 if target_page == "A" else 1),
            t_start=t0, t_end=t1, bits=bits, group=group, intervene=intervene,
        )
        k_mat, v_mat = materialize_page(packed, mode="quant", device=k.device, dtype=k.dtype)
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)
    else:
        raise ValueError(mode)

    page_hash = packed.content_hash()

    rem = sp.shared_ids[t1:]
    if rem:
        rem_t = torch.tensor([rem], device=device, dtype=torch.long)
        out = _forward_chunk(model, rem_t, past=past, use_cache=True)
        past = out.past_key_values
        if mode == "quant" and k_mat is not None:
            past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)

    results = []
    for question, fp_nlls in (("jia", fp_nlls_jia), ("yi", fp_nlls_yi)):
        q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
        a_ids = sp.ans_jia_ids if question == "jia" else sp.ans_yi_ids
        key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi
        past_q = clone_past(past)
        nlls, mean_nll, _ = _teacher_force_nll(model, past_q, q_ids, a_ids, device)
        deltas = [qn - fn for qn, fn in zip(nlls, fp_nlls)]
        S = sum(deltas) / max(len(deltas), 1)
        S_key = deltas[key_off]
        t_key = len(sp.shared_ids) + len(q_ids) + key_off
        d = t_key - t_page_end
        results.append(
            TrajectoryResult(
                name=f"{mode}_{target_page}_{intervene}_{question}",
                mode=mode,
                intervene=intervene,
                page=target_page,
                question=question,
                role=role_map[(target_page, question)],
                S=S,
                S_key=S_key,
                nll_fp=sum(fp_nlls) / max(len(fp_nlls), 1),
                nll_run=mean_nll,
                page_hash=page_hash,
                t_key=t_key,
                t_page_end=t_page_end,
                d=d,
                answer_logprobs=[-x for x in nlls],
                device=str(device),
            )
        )
    return results[0], results[1], packed, True


def run_fork_pair_span(
    model,
    sp: SmokePrefix,
    *,
    page_index: int,
    t0: int,
    t1: int,
    intervene: Intervene,
    mode: Mode,
    layer: int,
    kv_head: int,
    L_p: int,
    bits: int,
    group: int,
    device: torch.device,
    fp_nlls_jia: List[float],
    fp_nlls_yi: List[float],
) -> Tuple[TrajectoryResult, TrajectoryResult, PackedPage, bool]:
    """Like run_fork_pair but for an arbitrary closed page span [t0,t1)."""
    assert t1 - t0 == L_p or (t1 > t0), f"bad span {(t0,t1)}"
    t_page_end = t1 - 1
    page_label = f"p{page_index}"

    ids_to_close = torch.tensor([sp.shared_ids[:t1]], device=device, dtype=torch.long)
    out = _forward_chunk(model, ids_to_close, past=None, use_cache=True)
    past = out.past_key_values
    k, v = _get_kv(past, layer)
    k_page = k[0, kv_head, t0:t1, :].detach()
    v_page = v[0, kv_head, t0:t1, :].detach()

    if mode == "zero_pert":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=page_index,
            t_start=t0, t_end=t1, bits=bits, group=group,
            intervene=intervene if intervene != "none" else "k",
        )
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_page, v_page)
        k_mat = v_mat = None
    elif mode == "quant":
        packed = pack_page_kv(
            k_page, v_page,
            layer=layer, kv_head=kv_head, page_id=page_index,
            t_start=t0, t_end=t1, bits=bits, group=group, intervene=intervene,
        )
        k_mat, v_mat = materialize_page(packed, mode="quant", device=k.device, dtype=k.dtype)
        past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)
    else:
        raise ValueError(mode)

    page_hash = packed.content_hash()
    rem = sp.shared_ids[t1:]
    if rem:
        rem_t = torch.tensor([rem], device=device, dtype=torch.long)
        out = _forward_chunk(model, rem_t, past=past, use_cache=True)
        past = out.past_key_values
        if mode == "quant" and k_mat is not None:
            past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)

    results = []
    for question, fp_nlls in (("jia", fp_nlls_jia), ("yi", fp_nlls_yi)):
        q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
        a_ids = sp.ans_jia_ids if question == "jia" else sp.ans_yi_ids
        key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi
        past_q = clone_past(past)
        nlls, mean_nll, _ = _teacher_force_nll(model, past_q, q_ids, a_ids, device)
        deltas = [qn - fn for qn, fn in zip(nlls, fp_nlls)]
        S = sum(deltas) / max(len(deltas), 1)
        S_key = deltas[key_off]
        t_key = len(sp.shared_ids) + len(q_ids) + key_off
        d = t_key - t_page_end
        results.append(
            TrajectoryResult(
                name=f"{mode}_{page_label}_{intervene}_{question}",
                mode=mode,
                intervene=intervene,
                page=page_label,
                question=question,
                role="sampled_page",
                S=S,
                S_key=S_key,
                nll_fp=sum(fp_nlls) / max(len(fp_nlls), 1),
                nll_run=mean_nll,
                page_hash=page_hash,
                t_key=t_key,
                t_page_end=t_page_end,
                d=d,
                answer_logprobs=[-x for x in nlls],
                device=str(device),
            )
        )
    return results[0], results[1], packed, True


def run_fork_pair_pages(
    model,
    sp: SmokePrefix,
    *,
    page_spans: List[Tuple[int, int, int]],
    intervene: Intervene,
    mode: Mode,
    layer: int,
    kv_head: int,
    L_p: int,
    bits: int,
    group: int,
    device: torch.device,
    fp_nlls_jia: List[float],
    fp_nlls_yi: List[float],
    page_label: str = "multi",
) -> Tuple[TrajectoryResult, TrajectoryResult, str, bool]:
    """
    Jointly quantize multiple closed page spans in one fork.

    page_spans: list of (page_index, t0, t1) — all must satisfy t1 <= len(shared).
    Forward FP through max(t1), pack+rewrite each span, continue rem from max(t1),
    then fork Q甲/Q乙. Returns (res_jia, res_yi, joint_hash, True).

    Chunk-matched FP baseline should use chunk_at=max(t1).
    """
    assert page_spans, "empty page_spans"
    for pi, t0, t1 in page_spans:
        assert 0 <= t0 < t1 <= len(sp.shared_ids), (pi, t0, t1, len(sp.shared_ids))
        assert (t1 - t0) == L_p, f"page len {(t1-t0)} != L_p={L_p}"

    # Close at the latest page end so all selected pages are in cache
    t_close = max(t1 for _, _, t1 in page_spans)
    t_page_end = t_close - 1

    ids_to_close = torch.tensor([sp.shared_ids[:t_close]], device=device, dtype=torch.long)
    out = _forward_chunk(model, ids_to_close, past=None, use_cache=True)
    past = out.past_key_values
    k, v = _get_kv(past, layer)

    packed_list: List[PackedPage] = []
    mats: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []  # t0,t1,k_mat,v_mat
    hash_parts: List[str] = []

    for pi, t0, t1 in page_spans:
        k_page = k[0, kv_head, t0:t1, :].detach()
        v_page = v[0, kv_head, t0:t1, :].detach()
        if mode == "zero_pert":
            packed = pack_page_kv(
                k_page, v_page,
                layer=layer, kv_head=kv_head, page_id=pi,
                t_start=t0, t_end=t1, bits=bits, group=group,
                intervene=intervene if intervene != "none" else "k",
            )
            past = _set_kv_slice(past, layer, kv_head, t0, t1, k_page, v_page)
            mats.append((t0, t1, k_page, v_page))
        elif mode == "quant":
            packed = pack_page_kv(
                k_page, v_page,
                layer=layer, kv_head=kv_head, page_id=pi,
                t_start=t0, t_end=t1, bits=bits, group=group, intervene=intervene,
            )
            k_mat, v_mat = materialize_page(packed, mode="quant", device=k.device, dtype=k.dtype)
            past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)
            mats.append((t0, t1, k_mat, v_mat))
        else:
            raise ValueError(mode)
        packed_list.append(packed)
        hash_parts.append(packed.content_hash())

    import hashlib as _hl
    joint_hash = _hl.sha256("|".join(hash_parts).encode()).hexdigest()

    rem = sp.shared_ids[t_close:]
    if rem:
        rem_t = torch.tensor([rem], device=device, dtype=torch.long)
        out = _forward_chunk(model, rem_t, past=past, use_cache=True)
        past = out.past_key_values
        if mode == "quant":
            for t0, t1, k_mat, v_mat in mats:
                past = _set_kv_slice(past, layer, kv_head, t0, t1, k_mat, v_mat)

    results = []
    for question, fp_nlls in (("jia", fp_nlls_jia), ("yi", fp_nlls_yi)):
        q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
        a_ids = sp.ans_jia_ids if question == "jia" else sp.ans_yi_ids
        key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi
        past_q = clone_past(past)
        nlls, mean_nll, _ = _teacher_force_nll(model, past_q, q_ids, a_ids, device)
        deltas = [qn - fn for qn, fn in zip(nlls, fp_nlls)]
        S = sum(deltas) / max(len(deltas), 1)
        S_key = deltas[key_off]
        t_key = len(sp.shared_ids) + len(q_ids) + key_off
        d = t_key - t_page_end
        results.append(
            TrajectoryResult(
                name=f"{mode}_{page_label}_{intervene}_{question}",
                mode=mode,
                intervene=intervene,
                page=page_label,
                question=question,
                role="multi_page",
                S=S,
                S_key=S_key,
                nll_fp=sum(fp_nlls) / max(len(fp_nlls), 1),
                nll_run=mean_nll,
                page_hash=joint_hash,
                t_key=t_key,
                t_page_end=t_page_end,
                d=d,
                answer_logprobs=[-x for x in nlls],
                device=str(device),
            )
        )
    return results[0], results[1], joint_hash, True
