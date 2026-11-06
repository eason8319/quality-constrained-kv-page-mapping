"""Probe DynamicCache attribute layout on this transformers version."""
import os, torch
os.environ.setdefault("HF_HOME", "/cluster/home/zengy/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
snap = "/cluster/home/zengy/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
ids = torch.tensor([[1,2,3,4]], device="cuda")
out = model(input_ids=ids, use_cache=True)
past = out.past_key_values
print("type", type(past), type(past).__name__)
print("dir sample", [a for a in dir(past) if not a.startswith("_")][:40])
for attr in ("key_cache", "value_cache", "layers", "to_legacy_cache"):
    print(attr, hasattr(past, attr))
if hasattr(past, "key_cache"):
    print("key_cache len", len(past.key_cache), "shape0", None if not past.key_cache else past.key_cache[0].shape)
if hasattr(past, "layers"):
    print("layers len", len(past.layers), "layer0", past.layers[0], dir(past.layers[0])[:30])
if hasattr(past, "to_legacy_cache"):
    leg = past.to_legacy_cache()
    print("legacy type", type(leg), "len", len(leg), "k0", leg[0][0].shape)
