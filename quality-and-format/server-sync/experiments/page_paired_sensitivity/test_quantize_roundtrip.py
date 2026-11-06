import torch
from quantize import pack_page_kv, materialize_page, dequantize

def main():
    torch.manual_seed(0)
    L, D = 64, 128
    k = torch.randn(L, D)
    v = torch.randn(L, D)
    page = pack_page_kv(k, v, layer=0, kv_head=0, page_id=0, t_start=0, t_end=64, bits=4, group=64, intervene="kv")
    k2, v2 = materialize_page(page, mode="quant", device=torch.device("cpu"), dtype=torch.float32)
    # rough bound: 4-bit should be close-ish
    print("k_err", (k2 - k).abs().mean().item(), "v_err", (v2 - v).abs().mean().item())
    print("hash", page.content_hash())
    # zero pert
    k0, v0 = materialize_page(page, mode="zero_pert", device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(k0, k.cpu()) and torch.equal(v0, v.cpu())
    print("zero_pert identity OK")

if __name__ == "__main__":
    main()
