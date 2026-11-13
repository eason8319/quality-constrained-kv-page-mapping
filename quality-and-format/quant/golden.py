"""CPU golden codec. All encoding arithmetic is FP32 on BF16 source values."""
from dataclasses import dataclass
import torch

FORMATS = ("fp16", "bf16", "e8m0")
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


class NumericError(ValueError):
    def __init__(self, page, group, fmt, reason, count):
        self.record = dict(page=page, group=list(group), scale_fmt=fmt, reason=reason,
                           failed_groups=count)
        super().__init__(str(self.record))


def reject(mask, page, fmt, reason):
    if mask.any():
        loc = mask.nonzero()
        raise NumericError(page, loc[0].tolist(), fmt, reason, len(loc))


def encode_scale(raw, fmt, page="scale"):
    if fmt not in FORMATS:
        raise ValueError(fmt)
    reject(~torch.isfinite(raw) | (raw < 0), page, fmt, "invalid_fp32_step")
    if fmt in DTYPE:
        scale = raw.to(DTYPE[fmt])
        reject(~torch.isfinite(scale), page, fmt, "metadata_scale_overflow")
        return scale
    # frexp avoids log2 rounding an FP32 value just above a power of two down.
    mantissa, exp = torch.frexp(raw)
    exponent = torch.where(raw == 0, -127, exp - (mantissa == .5).int()).clamp_min(-127)
    reject(exponent > 127, page, fmt, "e8m0_overflow")
    return (exponent + 127).to(torch.uint8)


def decode_scale(field, fmt):
    if fmt == "e8m0":
        reject(field == 255, "decode", fmt, "reserved_exponent")
        return torch.ldexp(torch.ones_like(field, dtype=torch.float32), field.int()-127)
    return field.float()


def orient(x, direction):
    if direction not in ("K", "V"):
        raise ValueError(direction)
    return x.T if direction == "K" else x


def extrema(x, group):
    # Padding is excluded from reductions; the final group remains page-local.
    starts = torch.arange(0, x.shape[1], group)
    tail = (-x.shape[1]) % group
    lo = torch.nn.functional.pad(x, (0, tail), value=float("inf")).reshape(len(x), len(starts), group).amin(-1)
    hi = torch.nn.functional.pad(x, (0, tail), value=-float("inf")).reshape(len(x), len(starts), group).amax(-1)
    return lo, hi


@dataclass
class Encoded:
    codes: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    stats: dict


def expand(field, width, group):
    return field.repeat_interleave(group, dim=1)[:, :width]


def decode(codes, scales, offsets, direction, group, fmt, page="decode"):
    q = orient(codes, direction)
    step = decode_scale(scales, fmt)
    reject(~torch.isfinite(step) | (step < 0) | ~torch.isfinite(offsets), page, fmt, "invalid_metadata")
    product = q.float() * expand(step, q.shape[1], group)
    result = (product + expand(offsets.float(), q.shape[1], group)).bfloat16()
    invalid = ~torch.isfinite(result)
    if invalid.any():
        _, group_invalid = extrema(invalid.float(), group)
        reject(group_invalid.bool(), page, fmt, "reconstruction_overflow")
    return orient(result, direction).contiguous()


def encode(values, direction, bits, group, fmt, page="anonymous"):
    if values.dtype != torch.bfloat16 or values.ndim != 2 or values.numel() == 0:
        raise ValueError("expected a nonempty BF16 [token,channel] page")
    if bits not in (2,4,8) or group < 1:
        raise ValueError("invalid quantization geometry")
    x = orient(values.detach().cpu().float(), direction)
    lo, hi = extrema(x, group)
    reject(~torch.isfinite(lo) | ~torch.isfinite(hi), page, fmt, "nonfinite_input")
    offsets = lo.half()
    reject(~torch.isfinite(offsets), page, fmt, "metadata_offset_overflow")
    scales = encode_scale((hi-lo)/((1<<bits)-1), fmt, page)
    step = decode_scale(scales, fmt)
    flat = lo == hi
    zero = step == 0
    skip = expand(flat | zero, x.shape[1], group)
    denom = expand(torch.where(zero, 1., step), x.shape[1], group)
    codes = ((x-expand(offsets.float(), x.shape[1], group))/denom).round().clamp(0,(1<<bits)-1)
    codes = orient(codes.masked_fill(skip, 0).byte(), direction).contiguous()
    stats = dict(total_groups=lo.numel(), constant_groups=int(flat.sum()),
                 nonconstant_zero_step=int((~flat & zero).sum()), metadata_overflow_groups=0)
    decode(codes, scales, offsets, direction, group, fmt, page)
    return Encoded(codes, scales, offsets, lo, hi, stats)


def quantize_group(x_bf16, bits, scale_fmt):
    x = x_bf16.reshape(1, -1)
    enc = encode(x, "V", bits, x.numel(), scale_fmt)
    result = decode(enc.codes, enc.scales, enc.offsets, "V", x.numel(), scale_fmt)
    return enc.codes.flatten(), enc.scales.squeeze(), enc.offsets.squeeze(), result.flatten()
