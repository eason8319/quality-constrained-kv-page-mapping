"""Build paired prefixes (smoke + 32-prefix pilot) with pages A/B and Q甲/乙."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .wordbank import CODEWORDS


PageBucket = Literal["first_token_page", "recent_closed", "earlier_history"]


@dataclass
class SmokePrefix:
    prefix_id: str
    shared_ids: List[int]
    q_jia_ids: List[int]
    q_yi_ids: List[int]
    ans_jia_ids: List[int]
    ans_yi_ids: List[int]
    page_a: Tuple[int, int]
    page_b: Tuple[int, int]
    t_page_end_a: int
    t_page_end_b: int
    key_offset_jia: int
    key_offset_yi: int
    text_shared: str
    text_jia: str
    text_yi: str
    fact_a: str
    fact_b: str
    # pilot extras
    swap_labels: bool = False
    n_trailing_pages: int = 1
    page_a_index: int = 1  # which L_p-page holds A (0-based among shared pages)
    page_b_index: int = 2


def _pad_to_page(tokenizer, text: str, page_len: int, pad_phrase: str = " The archive remains sealed. ") -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= page_len:
        return tokenizer.decode(ids[:page_len])
    out = text
    guard = 0
    while True:
        ids = tokenizer.encode(out, add_special_tokens=False)
        if len(ids) >= page_len:
            return tokenizer.decode(ids[:page_len])
        out = out + pad_phrase
        guard += 1
        if guard > 500:
            raise RuntimeError("pad_to_page failed to reach page_len")


def _exact_page_ids(tokenizer, text: str, L_p: int) -> List[int]:
    padded = _pad_to_page(tokenizer, text, L_p)
    return tokenizer.encode(padded, add_special_tokens=False)[:L_p]


def page_pos_bucket(page_index: int, d: int, recent_window: int = 128) -> PageBucket:
    """Protocol §2.1: first-token page listed separately; else recent if 0<d≤128."""
    if page_index == 0:
        return "first_token_page"
    if 0 < d <= recent_window:
        return "recent_closed"
    return "earlier_history"


def build_smoke_prefix(tokenizer, *, L_p: int = 64, prefix_id: str = "smoke000") -> SmokePrefix:
    return build_prefix(
        tokenizer,
        L_p=L_p,
        prefix_id=prefix_id,
        fact_a="APPLEWOOD",
        fact_b="RIVERBANK",
        swap_labels=False,
        n_trailing_pages=1,
        put_a_on_first_content_page=False,
    )


def build_prefix(
    tokenizer,
    *,
    L_p: int,
    prefix_id: str,
    fact_a: str,
    fact_b: str,
    swap_labels: bool = False,
    n_trailing_pages: int = 1,
    put_a_on_first_content_page: bool = False,
) -> SmokePrefix:
    """
    Layout (token pages of length L_p):
      [intro][optional filler…][page A][page B][trailing × n]
    If put_a_on_first_content_page: intro is shortened into page0 with fact A after first tokens
    via making page0 = intro+start and page A index 0 — we instead place fact A on page index 0
    by using a minimal system line on the same page *after* a few tokens (first-token page bucket).
    """
    label_a, label_b = ("Alpha", "Beta")
    if swap_labels:
        # swap physical archive names vs facts assignment for cross-prefix variety
        # Fact strings stay fact_a/fact_b; questions still ask Alpha vs Beta content roles:
        # When swapped, Alpha page gets fact_b and Beta page gets fact_a, and answers swap.
        page_fact_a, page_fact_b = fact_b, fact_a
        ans_a, ans_b = fact_b, fact_a
    else:
        page_fact_a, page_fact_b = fact_a, fact_b
        ans_a, ans_b = fact_a, fact_b

    if put_a_on_first_content_page:
        # Page 0 holds Archive Alpha fact (first-token page bucket). Keep a short header first.
        intro = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Reply with one codeword only.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"=== ARCHIVE {label_a.upper()} ===\n"
            f"Archive {label_a} stores the codeword: {page_fact_a}.\n"
        )
        intro_ids = _exact_page_ids(tokenizer, intro, L_p)
        page_a_ids = intro_ids
        page_a_index = 0

        page_b_body = (
            f"=== ARCHIVE {label_b.upper()} ===\n"
            f"Archive {label_b} stores the codeword: {page_fact_b}.\n"
            f"This fact belongs only to Archive {label_b}.\n"
        )
        page_b_ids = _exact_page_ids(tokenizer, page_b_body, L_p)
        page_b_index = 1
        pages = [page_a_ids, page_b_ids]
    else:
        intro = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a careful assistant. When asked for a codeword, reply with exactly "
            "that one word and nothing else.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Context documents follow. Use only information from the archives.\n"
        )
        intro_ids = _exact_page_ids(tokenizer, intro, L_p)
        page_a_body = (
            f"=== ARCHIVE {label_a.upper()} ===\n"
            f"Archive {label_a} stores the codeword: {page_fact_a}.\n"
            f"This fact belongs only to Archive {label_a}.\n"
        )
        page_b_body = (
            f"=== ARCHIVE {label_b.upper()} ===\n"
            f"Archive {label_b} stores the codeword: {page_fact_b}.\n"
            f"This fact belongs only to Archive {label_b}.\n"
        )
        page_a_ids = _exact_page_ids(tokenizer, page_a_body, L_p)
        page_b_ids = _exact_page_ids(tokenizer, page_b_body, L_p)
        page_a_index = 1
        page_b_index = 2
        pages = [intro_ids, page_a_ids, page_b_ids]

    trailing = []
    for i in range(max(1, n_trailing_pages)):
        rem = (
            f"=== NOTES part {i+1} ===\n"
            "Archives above are closed. Answer the forthcoming question using only "
            "those archives. Do not invent codewords. "
            f"Reference marker {prefix_id}-{i}.\n"
        )
        trailing.append(_exact_page_ids(tokenizer, rem, L_p))
    pages.extend(trailing)

    shared_ids: List[int] = []
    for p in pages:
        assert len(p) == L_p
        shared_ids.extend(p)

    page_a_span = (page_a_index * L_p, (page_a_index + 1) * L_p)
    page_b_span = (page_b_index * L_p, (page_b_index + 1) * L_p)

    q_jia = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"What codeword does Archive {label_a} store? Reply with one word only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_yi = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"What codeword does Archive {label_b} store? Reply with one word only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_jia_ids = tokenizer.encode(q_jia, add_special_tokens=False)
    q_yi_ids = tokenizer.encode(q_yi, add_special_tokens=False)
    ans_jia_ids = tokenizer.encode(ans_a, add_special_tokens=False)
    ans_yi_ids = tokenizer.encode(ans_b, add_special_tokens=False)

    return SmokePrefix(
        prefix_id=prefix_id,
        shared_ids=shared_ids,
        q_jia_ids=q_jia_ids,
        q_yi_ids=q_yi_ids,
        ans_jia_ids=ans_jia_ids,
        ans_yi_ids=ans_yi_ids,
        page_a=page_a_span,
        page_b=page_b_span,
        t_page_end_a=page_a_span[1] - 1,
        t_page_end_b=page_b_span[1] - 1,
        key_offset_jia=0,
        key_offset_yi=0,
        text_shared=tokenizer.decode(shared_ids),
        text_jia=q_jia,
        text_yi=q_yi,
        fact_a=ans_a,
        fact_b=ans_b,
        swap_labels=swap_labels,
        n_trailing_pages=n_trailing_pages,
        page_a_index=page_a_index,
        page_b_index=page_b_index,
    )


def build_pilot_prefixes(tokenizer, *, L_p: int = 64, n: int = 32, seed: int = 20261105) -> List[SmokePrefix]:
    """32 distinct prefixes with position / label variety."""
    assert len(CODEWORDS) >= 2 * n
    out: List[SmokePrefix] = []
    for i in range(n):
        fa = CODEWORDS[2 * i]
        fb = CODEWORDS[2 * i + 1]
        # Cycle trailing length: 1 → recent; 3 → larger d; 5 → often earlier_history for A
        n_trail = [1, 2, 3, 5][i % 4]
        swap = (i % 2 == 1)
        first_page = (i % 8 == 0)  # every 8th: A on first-token page
        pid = f"p{i:03d}"
        sp = build_prefix(
            tokenizer,
            L_p=L_p,
            prefix_id=pid,
            fact_a=fa,
            fact_b=fb,
            swap_labels=swap,
            n_trailing_pages=n_trail,
            put_a_on_first_content_page=first_page,
        )
        out.append(sp)
    return out


def summarize_prefix(sp: SmokePrefix) -> Dict:
    d = {
        "prefix_id": sp.prefix_id,
        "shared_len": len(sp.shared_ids),
        "page_a": list(sp.page_a),
        "page_b": list(sp.page_b),
        "t_page_end_a": sp.t_page_end_a,
        "t_page_end_b": sp.t_page_end_b,
        "ans_jia_ids": sp.ans_jia_ids,
        "ans_yi_ids": sp.ans_yi_ids,
        "fact_a": sp.fact_a,
        "fact_b": sp.fact_b,
        "swap_labels": sp.swap_labels,
        "n_trailing_pages": sp.n_trailing_pages,
        "page_a_index": sp.page_a_index,
        "page_b_index": sp.page_b_index,
    }
    return d


def enrich_meta(sp: SmokePrefix, question: str, recent_window: int = 128) -> Dict:
    q_ids = sp.q_jia_ids if question == "jia" else sp.q_yi_ids
    key_off = sp.key_offset_jia if question == "jia" else sp.key_offset_yi
    t_key = len(sp.shared_ids) + len(q_ids) + key_off
    d_a = t_key - sp.t_page_end_a
    d_b = t_key - sp.t_page_end_b
    return {
        "t_key": t_key,
        "d_a": d_a,
        "d_b": d_b,
        "bucket_a": page_pos_bucket(sp.page_a_index, d_a, recent_window),
        "bucket_b": page_pos_bucket(sp.page_b_index, d_b, recent_window),
    }


# --- Non-retrieval / longer-context templates ---

_NARRATIVE_FILLERS = [
    "The morning light spilled across the kitchen table as the coffee cooled. "
    "Birds argued in the courtyard maple. Nothing urgent demanded attention yet. ",
    "On the train, strangers scrolled their phones while the carriage swayed. "
    "A child asked about the river; the parent answered softly. ",
    "Later, the office hallway smelled of printer toner and rain-damp coats. "
    "Meetings stacked like plates; someone laughed too loudly near the door. ",
    "In the park, a dog chased a tennis ball through wet grass. "
    "Joggers passed in pairs. The city felt briefly generous. ",
    "Evening settled with a low hum of traffic. Streetlights blinked on one by one. "
    "A bakery closed its shutters; the scent of yeast lingered. ",
    "She reread an old email and decided not to reply. "
    "The cursor blinked. Outside, a scooter rattled over a drain cover. ",
    "He sketched a map of the neighborhood on a napkin, then folded it away. "
    "Plans for the weekend remained deliberately unfinished. ",
    "Rain resumed after midnight, soft against the window. "
    "Somewhere a radio played a song nobody could name. ",
]


def _fact_page_text(person: str, fact_word: str, scene: str) -> str:
    return (
        f"{scene} "
        f"{person} quietly noted a private keyword for later: {fact_word}. "
        f"It was easy to miss among ordinary details, but it mattered. "
    )


def _filler_page_text(seed_i: int, page_i: int, prefix_id: str) -> str:
    base = _NARRATIVE_FILLERS[(seed_i + page_i) % len(_NARRATIVE_FILLERS)]
    return (
        f"{base}"
        f"Daybook entry {prefix_id}-{page_i}: the afternoon continued without drama. "
        f"Someone mentioned the weather twice. A clock on the wall was three minutes fast. "
    )


def build_nonretr_long_prefix(
    tokenizer,
    *,
    L_p: int,
    prefix_id: str,
    fact_a: str,
    fact_b: str,
    n_pages: int,
    page_a_index: int,
    page_b_index: int,
    seed_i: int = 0,
) -> SmokePrefix:
    """
    Narrative / diary-style long shared context (no archive-retrieval framing).
    Fact words buried in prose at page_a_index / page_b_index.
    Total closed pages = n_pages (each exactly L_p tokens).
    """
    assert n_pages >= 4
    assert 0 <= page_a_index < n_pages and 0 <= page_b_index < n_pages
    assert page_a_index != page_b_index

    person_a, person_b = "Maya", "Jonah"
    scene_a = "While packing a suitcase near the window,"
    scene_b = "During a late walk past the closed bookstore,"

    pages: List[List[int]] = []
    for i in range(n_pages):
        if i == 0:
            # system + user open + start of narrative (first-token page)
            text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a careful reader. Answer with exactly one word from the diary "
                "when asked, and nothing else.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Below is an excerpt from a personal daybook. Read carefully.\n"
            )
            if page_a_index == 0:
                text += _fact_page_text(person_a, fact_a, scene_a)
            elif page_b_index == 0:
                text += _fact_page_text(person_b, fact_b, scene_b)
            else:
                text += _filler_page_text(seed_i, i, prefix_id)
        elif i == page_a_index:
            text = _fact_page_text(person_a, fact_a, scene_a)
        elif i == page_b_index:
            text = _fact_page_text(person_b, fact_b, scene_b)
        else:
            text = _filler_page_text(seed_i, i, prefix_id)
        pages.append(_exact_page_ids(tokenizer, text, L_p))

    shared_ids: List[int] = []
    for p in pages:
        assert len(p) == L_p
        shared_ids.extend(p)

    page_a_span = (page_a_index * L_p, (page_a_index + 1) * L_p)
    page_b_span = (page_b_index * L_p, (page_b_index + 1) * L_p)

    q_jia = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "In the daybook, what private keyword did Maya note? Reply with one word only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_yi = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "In the daybook, what private keyword did Jonah note? Reply with one word only."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_jia_ids = tokenizer.encode(q_jia, add_special_tokens=False)
    q_yi_ids = tokenizer.encode(q_yi, add_special_tokens=False)
    ans_jia_ids = tokenizer.encode(fact_a, add_special_tokens=False)
    ans_yi_ids = tokenizer.encode(fact_b, add_special_tokens=False)

    return SmokePrefix(
        prefix_id=prefix_id,
        shared_ids=shared_ids,
        q_jia_ids=q_jia_ids,
        q_yi_ids=q_yi_ids,
        ans_jia_ids=ans_jia_ids,
        ans_yi_ids=ans_yi_ids,
        page_a=page_a_span,
        page_b=page_b_span,
        t_page_end_a=page_a_span[1] - 1,
        t_page_end_b=page_b_span[1] - 1,
        key_offset_jia=0,
        key_offset_yi=0,
        text_shared=tokenizer.decode(shared_ids),
        text_jia=q_jia,
        text_yi=q_yi,
        fact_a=fact_a,
        fact_b=fact_b,
        swap_labels=False,
        n_trailing_pages=max(0, n_pages - max(page_a_index, page_b_index) - 1),
        page_a_index=page_a_index,
        page_b_index=page_b_index,
    )


def build_nonretr_long_prefixes(
    tokenizer,
    *,
    L_p: int = 64,
    n: int = 32,
    seed: int = 20261106,
) -> List[SmokePrefix]:
    """
    Non-retrieval longer-context prefixes.
    Target ~12–20 closed pages (768–1280 tokens). Seed default 20261106.
    """
    import random

    rng = random.Random(seed)
    assert len(CODEWORDS) >= 2 * n
    # Offset into wordbank so we don't reuse the exact pilot_32 pairs in the same order
    base = (seed % 17) * 2
    out: List[SmokePrefix] = []
    # Cycle lengths 12,14,16,18,20
    lengths = [12, 14, 16, 18, 20]
    for i in range(n):
        n_pages = lengths[i % len(lengths)]
        fa = CODEWORDS[(base + 2 * i) % len(CODEWORDS)]
        fb = CODEWORDS[(base + 2 * i + 1) % len(CODEWORDS)]
        # Place facts: vary early/mid/late; ensure distinct
        # early: 0 or 1; mid: n_pages//2 ±1; late: n_pages-3 .. n_pages-2 (leave last pages for recent)
        candidates_a = [0, 1, max(2, n_pages // 3), n_pages // 2, max(2, n_pages - 4)]
        page_a_index = candidates_a[i % len(candidates_a)] % n_pages
        # B offset from A
        offset = [2, 3, 4, n_pages // 2][i % 4]
        page_b_index = (page_a_index + offset) % n_pages
        if page_b_index == page_a_index:
            page_b_index = (page_a_index + 1) % n_pages
        # Prefer B not on last page only — OK if it is
        pid = f"nr{i:03d}"
        sp = build_nonretr_long_prefix(
            tokenizer,
            L_p=L_p,
            prefix_id=pid,
            fact_a=fa,
            fact_b=fb,
            n_pages=n_pages,
            page_a_index=page_a_index,
            page_b_index=page_b_index,
            seed_i=i + seed,
        )
        out.append(sp)
    return out


# --- Strong pairing templates (amplify need-vs-need-not D) ---

_STRONG_FILLERS = [
    "The courier depot hummed under fluorescent lights. Clerks stamped forms without looking up. ",
    "Rain streaked the glass partition. A radio murmured shipping delays for the northern line. ",
    "Someone spilled tea on a blank docket and cursed softly. The clock jumped a minute. ",
    "A trolley of sealed envelopes rattled past. Nobody checked the seals twice. ",
    "Outside, a delivery van idled. Inside, the filing cabinets smelled of dust and toner. ",
    "A trainee misfiled a folder then corrected it. The supervisor did not notice. ",
    "The afternoon shift began with a dull briefing about weather and route changes. ",
    "In the break room, a crossword sat unfinished beside a cold sandwich. ",
]


def _distractor_line(slot_name: str, distractor: str) -> str:
    return (
        f"A discarded sticky note near Slot {slot_name} showed a near-miss draft: {distractor}. "
        f"That draft was rejected and must not be used as the official registry string. "
    )


def _fact_registry_page(slot_name: str, phrase: str, scene: str) -> str:
    return (
        f"{scene} "
        f"Official Field Registry — Slot {slot_name}. "
        f"The exact multi-token registry string for Slot {slot_name} is: {phrase}. "
        f"Copy it character-for-character; do not substitute look-alikes. "
        f"Slot {slot_name} alone owns this string. "
    )


def build_strong_pair_prefix(
    tokenizer,
    *,
    L_p: int,
    prefix_id: str,
    phrase_a: str,
    phrase_b: str,
    distractor_a: str,
    distractor_b: str,
    n_trailing_pages: int = 4,
    page_a_index: int = 1,
    page_b_index: int = 2,
    n_mid_filler: int = 0,
    put_a_on_first_content_page: bool = False,
    seed_i: int = 0,
) -> SmokePrefix:
    """
    Strong pairing layout:
      [intro][optional mid fillers with near-miss distractors][page A][page B][trailing×n]
    Page A holds exact phrase_a (needed by Q_jia); page B holds phrase_b (needed by Q_yi).
    Filler/trailing pages plant near-miss distractors so corrupting the true fact page
    is more likely to drop need-question confidence (amplify D).
    Answers are multi-token rare registry strings (not single easy codewords).
    """
    slot_a, slot_b = "North", "South"
    scene_a = "While sealing the North courier pouch,"
    scene_b = "While logging the South outbound crate,"

    # Build page list with absolute indices
    # We construct pages in order 0..N-1 where N = 1 intro + mid + 2 facts (if not on intro) + trailing
    # Simpler approach: allocate a fixed page array sized by max(page_a, page_b)+1+trailing
    if put_a_on_first_content_page:
        page_a_index = 0
        if page_b_index == 0:
            page_b_index = 1

    n_pages = max(page_a_index, page_b_index) + 1 + max(1, n_trailing_pages)
    # Ensure room for mid fillers between intro and facts if requested
    if n_mid_filler > 0 and page_a_index < 1 + n_mid_filler:
        # bump fact indices
        shift = (1 + n_mid_filler) - page_a_index
        page_a_index += shift
        page_b_index += shift
        n_pages = max(page_a_index, page_b_index) + 1 + max(1, n_trailing_pages)

    pages: List[List[int]] = []
    for i in range(n_pages):
        if i == 0 and not put_a_on_first_content_page:
            text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a careful registry clerk. When asked for a registry string, "
                "reply with the exact multi-token string and nothing else.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Courier field logs follow. Near-miss drafts are invalid.\n"
            )
            text += _distractor_line("North", distractor_a)
        elif i == 0 and put_a_on_first_content_page:
            text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Reply with the exact registry string only.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
            )
            text += _fact_registry_page(slot_a, phrase_a, scene_a)
        elif i == page_a_index:
            text = _fact_registry_page(slot_a, phrase_a, scene_a)
        elif i == page_b_index:
            text = _fact_registry_page(slot_b, phrase_b, scene_b)
        else:
            # filler / trailing with distractors
            base = _STRONG_FILLERS[(seed_i + i) % len(_STRONG_FILLERS)]
            # Plant both near-misses so either question has a tempting wrong answer in context
            text = (
                f"{base}"
                f"Log fragment {prefix_id}-{i}. "
                f"{_distractor_line('North', distractor_a)}"
                f"{_distractor_line('South', distractor_b)}"
            )
        pages.append(_exact_page_ids(tokenizer, text, L_p))

    shared_ids: List[int] = []
    for p in pages:
        assert len(p) == L_p
        shared_ids.extend(p)

    page_a_span = (page_a_index * L_p, (page_a_index + 1) * L_p)
    page_b_span = (page_b_index * L_p, (page_b_index + 1) * L_p)

    q_jia = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the exact official registry string for Slot North? "
        "Reply with that string only, character-for-character."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_yi = (
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the exact official registry string for Slot South? "
        "Reply with that string only, character-for-character."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    q_jia_ids = tokenizer.encode(q_jia, add_special_tokens=False)
    q_yi_ids = tokenizer.encode(q_yi, add_special_tokens=False)
    ans_jia_ids = tokenizer.encode(phrase_a, add_special_tokens=False)
    ans_yi_ids = tokenizer.encode(phrase_b, add_special_tokens=False)

    return SmokePrefix(
        prefix_id=prefix_id,
        shared_ids=shared_ids,
        q_jia_ids=q_jia_ids,
        q_yi_ids=q_yi_ids,
        ans_jia_ids=ans_jia_ids,
        ans_yi_ids=ans_yi_ids,
        page_a=page_a_span,
        page_b=page_b_span,
        t_page_end_a=page_a_span[1] - 1,
        t_page_end_b=page_b_span[1] - 1,
        key_offset_jia=0,
        key_offset_yi=0,
        text_shared=tokenizer.decode(shared_ids),
        text_jia=q_jia,
        text_yi=q_yi,
        fact_a=phrase_a,
        fact_b=phrase_b,
        swap_labels=False,
        n_trailing_pages=max(1, n_trailing_pages),
        page_a_index=page_a_index,
        page_b_index=page_b_index,
    )


def build_strong_pair_prefixes(
    tokenizer,
    *,
    L_p: int = 64,
    n: int = 32,
    seed: int = 20261107,
) -> List[SmokePrefix]:
    """
    Strong-pairing pilot prefixes. Seed default 20261107 (distinct from pilot_32 / nonretr).
    Template family: strong_pair_registry.
    Varies trailing depth and fact page positions for bucket coverage.
    """
    from .wordbank import STRONG_PHRASE_PAIRS

    assert len(STRONG_PHRASE_PAIRS) >= 2, "need phrase pairs"
    out: List[SmokePrefix] = []
    trail_cycle = [3, 4, 5, 7, 8]  # deeper bury → earlier_history more often
    n_pairs = len(STRONG_PHRASE_PAIRS)
    for i in range(n):
        # Cycle through bank; pair i uses two consecutive entries (wrap).
        pa, da = STRONG_PHRASE_PAIRS[(2 * i) % n_pairs]
        pb, db = STRONG_PHRASE_PAIRS[(2 * i + 1) % n_pairs]
        if pa == pb:
            pb, db = STRONG_PHRASE_PAIRS[(2 * i + 2) % n_pairs]
        n_trail = trail_cycle[i % len(trail_cycle)]
        first_page = (i % 8 == 0)
        # Place B relative to A; with intro at 0, default A=1 B=2 unless first_page
        if first_page:
            page_a_index = 0
            page_b_index = 1 + (i % 2)  # 1 or 2
        else:
            page_a_index = 1
            page_b_index = 2
        # Occasionally insert mid filler between A and B by bumping B
        if i % 5 == 2 and not first_page:
            page_b_index = 3
        pid = f"sp{i:03d}"
        sp = build_strong_pair_prefix(
            tokenizer,
            L_p=L_p,
            prefix_id=pid,
            phrase_a=pa,
            phrase_b=pb,
            distractor_a=da,
            distractor_b=db,
            n_trailing_pages=n_trail,
            page_a_index=page_a_index,
            page_b_index=page_b_index,
            put_a_on_first_content_page=first_page,
            seed_i=i + seed,
        )
        out.append(sp)
    return out
