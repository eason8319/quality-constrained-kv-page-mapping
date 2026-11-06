"""Fixed-seed early/mid/late × 2 KV-head sampler for Llama-3.1-8B."""
from __future__ import annotations

import random
from typing import List, Tuple


def sample_layer_heads(
    *,
    n_layers: int = 32,
    n_kv_heads: int = 8,
    seed: int = 20261105,
) -> List[Tuple[int, int]]:
    """
    Return 12 (layer, kv_head) combos:
    early/mid/late bands × 2 layers × 2 kv heads.
    """
    rng = random.Random(seed)
    # thirds of 32 layers: [0,11), [11,22), [22,32)
    bands = [
        list(range(0, 11)),
        list(range(11, 22)),
        list(range(22, n_layers)),
    ]
    out: List[Tuple[int, int]] = []
    for band in bands:
        layers = sorted(rng.sample(band, 2))
        for L in layers:
            heads = sorted(rng.sample(range(n_kv_heads), 2))
            for h in heads:
                out.append((L, h))
    assert len(out) == 12
    # stable order: by layer then head
    out = sorted(set(out), key=lambda x: (x[0], x[1]))
    # If collision reduced count, refill deterministically
    while len(out) < 12:
        L = rng.randrange(n_layers)
        h = rng.randrange(n_kv_heads)
        if (L, h) not in out:
            out.append((L, h))
        out = sorted(out, key=lambda x: (x[0], x[1]))
    return out[:12]


if __name__ == "__main__":
    print(sample_layer_heads())
