"""Asymmetric uniform page pack for K (per-channel) / V (per-token)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


Axis = Literal["channel", "token"]


@dataclass
class PackedTensor:
    """Packed quantized tensor with asymmetric uniform params."""

    codes: torch.Tensor  # uint8 codes (values in 0..2^bits-1); b=2 or b=4
    scale: torch.Tensor
    zero: torch.Tensor
    axis: Axis
    shape: Tuple[int, ...]
    bits: int
    group: int
    dtype_name: str

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.codes.cpu().contiguous().numpy().tobytes())
        h.update(self.scale.cpu().float().contiguous().numpy().tobytes())
        h.update(self.zero.cpu().float().contiguous().numpy().tobytes())
        meta = json.dumps(
            {
                "axis": self.axis,
                "shape": list(self.shape),
                "bits": self.bits,
                "group": self.group,
                "dtype": self.dtype_name,
            },
            sort_keys=True,
        ).encode()
        h.update(meta)
        return h.hexdigest()


@dataclass
class PackedPage:
    """One closed KV page for a single (layer, kv_head)."""

    layer: int
    kv_head: int
    page_id: int
    t_start: int
    t_end: int  # exclusive
    k: Optional[PackedTensor]
    v: Optional[PackedTensor]
    k_fp: Optional[torch.Tensor]  # kept for zero-pert / identity path
    v_fp: Optional[torch.Tensor]

    def descriptor(self) -> dict:
        return {
            "layer": self.layer,
            "kv_head": self.kv_head,
            "page_id": self.page_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "k_hash": None if self.k is None else self.k.content_hash(),
            "v_hash": None if self.v is None else self.v.content_hash(),
        }

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.descriptor(), sort_keys=True).encode())
        return h.hexdigest()


def _quantize_asymmetric(
    x: torch.Tensor,
    bits: int,
    axis: Axis,
    group: int,
) -> PackedTensor:
    """
    x: [L_p, head_dim]
    K: axis=channel → along L_p for each channel (head_dim), groups of `group` along L_p
    V: axis=token → along head_dim for each token, groups of `group` along head_dim
    """
    assert bits in (2, 4), f"supported bits={{2,4}}, got {bits}"
    assert x.ndim == 2
    L, D = x.shape
    qmax = (1 << bits) - 1
    orig_dtype = x.dtype
    xf = x.float()

    if axis == "channel":
        # group along sequence for each channel
        assert L % group == 0 or True  # allow last partial group
        n_groups = (L + group - 1) // group
        # pad seq to n_groups * group
        pad_L = n_groups * group - L
        if pad_L:
            xf = torch.nn.functional.pad(xf, (0, 0, 0, pad_L))
        xg = xf.view(n_groups, group, D)  # [G, g, D]
        xmin = xg.amin(dim=1)  # [G, D]
        xmax = xg.amax(dim=1)
        scale = (xmax - xmin).clamp_min(1e-8) / qmax
        zero = xmin
        # quantize
        q = ((xg - zero.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, qmax)
        codes = q.to(torch.uint8).reshape(-1, D)[:L].contiguous()
        return PackedTensor(
            codes=codes,
            scale=scale.contiguous(),
            zero=zero.contiguous(),
            axis=axis,
            shape=(L, D),
            bits=bits,
            group=group,
            dtype_name=str(orig_dtype).replace("torch.", ""),
        )
    elif axis == "token":
        # group along channels for each token
        n_groups = (D + group - 1) // group
        pad_D = n_groups * group - D
        if pad_D:
            xf = torch.nn.functional.pad(xf, (0, pad_D))
        xg = xf.view(L, n_groups, group)  # [L, G, g]
        xmin = xg.amin(dim=2)  # [L, G]
        xmax = xg.amax(dim=2)
        scale = (xmax - xmin).clamp_min(1e-8) / qmax
        zero = xmin
        q = ((xg - zero.unsqueeze(2)) / scale.unsqueeze(2)).round().clamp(0, qmax)
        codes = q.to(torch.uint8).reshape(L, -1)[:, :D].contiguous()
        return PackedTensor(
            codes=codes,
            scale=scale.contiguous(),
            zero=zero.contiguous(),
            axis=axis,
            shape=(L, D),
            bits=bits,
            group=group,
            dtype_name=str(orig_dtype).replace("torch.", ""),
        )
    else:
        raise ValueError(axis)


def dequantize(packed: PackedTensor) -> torch.Tensor:
    bits = packed.bits
    qmax = (1 << bits) - 1
    L, D = packed.shape
    group = packed.group
    codes = packed.codes.float()
    scale = packed.scale.float()
    zero = packed.zero.float()

    if packed.axis == "channel":
        n_groups = scale.shape[0]
        pad_L = n_groups * group - L
        c = codes
        if pad_L:
            c = torch.nn.functional.pad(c, (0, 0, 0, pad_L))
        cg = c.view(n_groups, group, D)
        xg = cg * scale.unsqueeze(1) + zero.unsqueeze(1)
        out = xg.reshape(-1, D)[:L]
    else:
        n_groups = scale.shape[1]
        pad_D = n_groups * group - D
        c = codes
        if pad_D:
            c = torch.nn.functional.pad(c, (0, pad_D))
        cg = c.view(L, n_groups, group)
        xg = cg * scale.unsqueeze(2) + zero.unsqueeze(2)
        out = xg.reshape(L, -1)[:, :D]

    dtype = getattr(torch, packed.dtype_name)
    return out.to(dtype)


def pack_page_kv(
    k_page: torch.Tensor,
    v_page: torch.Tensor,
    *,
    layer: int,
    kv_head: int,
    page_id: int,
    t_start: int,
    t_end: int,
    bits: int = 4,
    group: int = 64,
    intervene: Literal["k", "v", "kv", "none"] = "kv",
) -> PackedPage:
    """
    k_page/v_page: [L_p, head_dim] for one kv head.
    intervene selects which tensors are quantized; 'none' stores FP only (zero-pert path).
    """
    # Keep a CPU copy for hashing/serialization and a same-device copy for exact zero-pert replay
    k_fp = k_page.detach().contiguous()
    v_fp = v_page.detach().contiguous()
    # CPU mirrors used only for content hashing of codes path; FP tensors stay on original device
    k_p = v_p = None
    if intervene in ("k", "kv"):
        k_p = _quantize_asymmetric(k_fp, bits=bits, axis="channel", group=group)
    if intervene in ("v", "kv"):
        v_p = _quantize_asymmetric(v_fp, bits=bits, axis="token", group=group)
    return PackedPage(
        layer=layer,
        kv_head=kv_head,
        page_id=page_id,
        t_start=t_start,
        t_end=t_end,
        k=k_p,
        v=v_p,
        k_fp=k_fp,
        v_fp=v_fp,
    )


def materialize_page(
    page: PackedPage,
    *,
    mode: Literal["quant", "zero_pert", "fp"],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (K,V) tensors [L_p, head_dim] to write back into cache."""
    if mode == "fp":
        return page.k_fp.to(device=device, dtype=dtype), page.v_fp.to(device=device, dtype=dtype)
    if mode == "zero_pert":
        # full intervention path but restore original FP page bytes
        return page.k_fp.to(device=device, dtype=dtype), page.v_fp.to(device=device, dtype=dtype)
    # quant: use dequant where packed, else FP
    if page.k is not None:
        k = dequantize(page.k).to(device=device, dtype=dtype)
    else:
        k = page.k_fp.to(device=device, dtype=dtype)
    if page.v is not None:
        v = dequantize(page.v).to(device=device, dtype=dtype)
    else:
        v = page.v_fp.to(device=device, dtype=dtype)
    return k, v
