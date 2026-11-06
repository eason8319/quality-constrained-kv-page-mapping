"""Enumerate closed pages in a smoke/pilot prefix."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .prefix_builder import SmokePrefix, page_pos_bucket


def closed_pages(sp: SmokePrefix, L_p: int, recent_window: int = 128) -> List[Dict]:
    """All L_p-aligned pages in shared prefix (closed before any question)."""
    n = len(sp.shared_ids) // L_p
    assert len(sp.shared_ids) == n * L_p
    # Use Q甲 t_key for d (questions are similar length)
    t_key = len(sp.shared_ids) + len(sp.q_jia_ids) + sp.key_offset_jia
    out = []
    for i in range(n):
        t0, t1 = i * L_p, (i + 1) * L_p
        t_page_end = t1 - 1
        d = t_key - t_page_end
        kind = "other"
        if i == sp.page_a_index:
            kind = "fact_A"
        elif i == sp.page_b_index:
            kind = "fact_B"
        elif i == 0:
            kind = "intro_or_first"
        else:
            kind = "trailing_or_mid"
        out.append(
            {
                "page_index": i,
                "t0": t0,
                "t1": t1,
                "t_page_end": t_page_end,
                "d": d,
                "page_pos_bucket": page_pos_bucket(i, d, recent_window),
                "kind": kind,
            }
        )
    return out
