#!/usr/bin/env python3
"""
Generate a broader synthetic test set for the unredaction MVP.

Four plausible declassified-style documents (Soviet strategic forces, Chinese
strategic forces, Persian Gulf oil, DPRK nuclear program), each with a mix of
single-word and multi-word redactions across difficulty tiers:

  recoverable    -- the word is strongly implied by the local context
  theme-echo     -- the word also appears elsewhere in the document (the
                    keyness prior should credit it)
  low-information-- many plausible fillers fit; partial credit is the ceiling
  absent-entity  -- the answer never appears in the visible document (only
                    historical context can supply it; doc_context excludes it)

Everything else is DERIVED, not hand-maintained, so the JSON is always
self-consistent:
  - before/after  split exactly at the ground truth's occurrence, preserving
                  the original spacing so reconstruction is exact even when
                  the GT is adjacent to punctuation ("triad," not "triad ,")
  - min/max_chars from the GT length (+/- margin; wider for multi-word gaps)
  - doc_context   the full document text (GT excluded only for absent-entity)
  - no-adjacency-echo: the GT's content tokens must NOT appear within +-8
    words of the gap (that is the echo trap, not a fair test)

Run:  python gen_dataset.py          # writes redactions_broad.json
"""

import json
import re

from keyness import tokenize, _stem  # same normalizer unredact.py uses

_PUNCT_AFTER = ".,;:!?)\u2019\"'"  # chars that may abut the GT with no space


def _slug(gt):
    return re.sub(r"[^a-z0-9]+", "_", gt.lower()).strip("_")


def _gap_after(after):
    """Was the GT directly followed by punctuation (no space)? Mirrors the
    reconstruction in validate_dataset.py and the runtime before/after split."""
    return "" if after[:1] in _PUNCT_AFTER else " "


def _make_entry(doc_id, n, doc_text, gt, category, note=None, occurrence=1,
                absent=False):
    # locate the requested occurrence of the GT
    idx = -1
    start = 0
    for _ in range(occurrence):
        idx = doc_text.find(gt, start)
        if idx < 0:
            raise ValueError(f"GT {gt!r} (occurrence {occurrence}) not found in {doc_id}")
        start = idx + len(gt)

    raw_before = doc_text[:idx]
    raw_after = doc_text[idx + len(gt):]
    before = raw_before.rstrip()
    after = raw_after.lstrip()
    gap_before = " " if len(raw_before) - len(before) > 0 else ""
    gap_after = " " if len(raw_after) - len(after) > 0 else ""

    # reconstruction invariant: before + gap_before + gt + gap_after + after
    if before + gap_before + gt + gap_after + after != doc_text:
        raise ValueError(f"{doc_id} entry {n}: reconstruction mismatch for {gt!r}")

    # no-adjacency-echo: GT content tokens (len>=4) must not sit in the +-8
    # window -- same threshold unredact.py's echo guard applies to the window.
    window = " ".join(before.split()[-8:] + after.split()[:8])
    wtoks = {_stem(t) for t in tokenize(window) if len(_stem(t)) >= 4}
    gtoks = {_stem(t) for t in tokenize(gt) if len(_stem(t)) >= 4}
    leak = gtoks & wtoks
    if leak:
        raise ValueError(f"{doc_id} entry {n}: GT {gt!r} leaks into the "
                         f"adjacency window: {sorted(leak)}")

    multi = len(gt.split()) > 1
    lo = max(3, len(gt) - (4 if multi else 2))
    hi = len(gt) + (6 if multi else 3)
    if hi < lo + 2:
        hi = lo + 2

    # absent-entity: doc_context is the released doc with the GT removed --
    # keep the space that sat after the GT (drop the one before it).
    doc_ctx = (before + _gap_after(after) + after) if absent else doc_text

    return {
        "id": f"{doc_id}{n}_{_slug(gt)}",
        "before": before,
        "after": after,
        "doc_context": doc_ctx,
        "min_chars": lo,
        "max_chars": hi,
        "ground_truth": gt,
        "category": category,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Document corpus (authored to read like declassified estimates / memos)
# ---------------------------------------------------------------------------

DOCS = {
    "sov": (
        "Soviet strategic forces continue to provide the principal means of "
        "delivering nuclear weapons against the United States. The sea-based "
        "leg of the triad, consisting of ballistic missile submarines, has grown "
        "steadily since 1970, and we assess that the sea-based deterrent now "
        "accounts for the majority of survivable warheads. We believe that the "
        "Soviet Union deploys a force of approximately sixty Yankee-class "
        "submarines, of which roughly twenty are on patrol at any given time. "
        "Construction at the Severodvinsk shipyard has been expanded, and the new "
        "Delta III submarine is expected to enter service within the next two "
        "years. The Soviet Union has also improved the accuracy of its land-based "
        "missiles, particularly the SS-18, which we assess can now destroy "
        "hardened missile silos with a single warhead.",
        [
            ("triad", "recoverable"),
            ("sea-based", "theme-echo",
             "Second occurrence; the first 'sea-based' stays visible earlier in "
             "the same document, so the keyness prior should credit it.",
             2),
            ("patrol", "recoverable"),
            ("expanded", "recoverable"),
            ("hardened missile silos", "recoverable"),
        ],
    ),
    "chi": (
        "China has continued to modernize its strategic nuclear forces, though at "
        "a slower pace than previously estimated. The deployment of the DF-5 "
        "missile, with a range sufficient to reach the continental United States, "
        "has been completed, and work has begun on a solid-propellant successor "
        "that could be deployed by the end of the decade. Peking maintains a "
        "policy of no first use, and we judge that Chinese forces are configured "
        "for retaliation rather than preemption. The number of warheads available "
        "to the strategic rocket forces is expected to double within five years, "
        "as multiple independently targetable reentry vehicles enter the force.",
        [
            ("solid-propellant successor", "recoverable"),
            ("no first use", "recoverable"),
            ("retaliation", "recoverable"),
            ("double", "low-information",
             "Common verb; double/triple/grow/increase all fit, so partial "
             "credit is the ceiling."),
            ("reentry vehicles", "recoverable"),
        ],
    ),
    "gulf": (
        "Crude oil exports from the Persian Gulf are expected to remain the "
        "principal source of supply for Western Europe and Japan through the next "
        "decade. The most important single facility remains the Ras Tanura "
        "terminal in Saudi Arabia, which handles a significant share of total "
        "Gulf liftings. We assess that a closure of the Strait of Hormuz, however "
        "unlikely, would cut off more than one third of world oil exports within "
        "a matter of weeks. Alternative pipeline routes through Turkey and the "
        "Red Sea provide only limited redundancy, and we judge that the strategic "
        "petroleum reserves of the major consuming nations would be exhausted "
        "within approximately six months of a prolonged interruption.",
        [
            ("Strait of Hormuz", "recoverable"),
            ("terminal", "recoverable"),
            ("redundancy", "recoverable"),
            ("strategic petroleum reserves", "recoverable"),
            ("exhausted", "recoverable"),
        ],
    ),
    "dprk": (
        "The Democratic People's Republic of Korea continues to pursue a nuclear "
        "weapons capability despite repeated international sanctions. We assess "
        "that the regime possesses sufficient fissile material for a small number "
        "of weapons, but that a reliable warhead design has not yet been fully "
        "demonstrated. The medium-range missile program, centered on the Nodong "
        "system, has progressed further than previously believed, and we judge "
        "that a test of a longer-range delivery vehicle could occur without "
        "warning. The principal uncertainty concerns the degree of assistance "
        "provided by Iran and other foreign suppliers, which we cannot verify at "
        "this time.",
        [
            ("sanctions", "recoverable"),
            ("fissile material", "recoverable"),
            ("demonstrated", "low-information",
             "Many verbs fit (verified/proven/demonstrated); partial credit is "
             "the ceiling."),
            ("longer-range delivery vehicle", "recoverable"),
            ("uncertainty", "recoverable"),
            ("Iran", "absent-entity",
             "True absent-entity case: 'Iran' is not present in the visible "
             "document (doc_context excludes it); only historical context can "
             "supply it.",
             1, True),
        ],
    ),
}


def main():
    entries = []
    for doc_id, (text, reds) in DOCS.items():
        for i, spec in enumerate(reds, 1):
            gt, category = spec[0], spec[1]
            note = spec[2] if len(spec) > 2 else None
            occurrence = spec[3] if len(spec) > 3 else 1
            absent = spec[4] if len(spec) > 4 else False
            entries.append(_make_entry(doc_id, i, text, gt, category, note,
                                       occurrence=occurrence, absent=absent))

    data = {
        "_comment": "Broad synthetic test set: 4 plausible declassified-style "
                    "documents with 21 redactions across difficulty tiers "
                    "(recoverable / theme-echo / low-information / "
                    "absent-entity), mixing single-word and multi-word gaps. "
                    "Generated by gen_dataset.py; char ranges derive from the "
                    "ground-truth length, doc_context is the full document, and "
                    "the no-adjacency-echo rule is enforced at generation time. "
                    "Note: non-absent doc_contexts include the ground truth "
                    "itself (matching redactions.json's existing convention), so "
                    "the keyness prior can see the answer word in doc stats; "
                    "only the absent-entity entry models the realistic "
                    "released-document case where the answer is invisible.",
        "redactions": entries,
    }
    with open("redactions_broad.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # summary table
    print(f"wrote redactions_broad.json: {len(entries)} redactions")
    cats = {}
    multi = 0
    for e in entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
        multi += len(e["ground_truth"].split()) > 1
    print(f"  categories: {cats}")
    print(f"  multi-word gaps: {multi} / {len(entries)}")
    print(f"  GT length range: {min(len(e['ground_truth']) for e in entries)}-"
          f"{max(len(e['ground_truth']) for e in entries)} chars")
    print("\n  id | GT | range | category")
    for e in entries:
        print(f"  {e['id']:>22} | {e['ground_truth']!r:<32} | "
              f"[{e['min_chars']},{e['max_chars']}] | {e['category']}")


if __name__ == "__main__":
    main()
