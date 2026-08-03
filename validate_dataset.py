#!/usr/bin/env python3
"""Validate redactions_broad.json: schema, GT-in-range, doc_context
reconstruction, and the no-adjacency-echo rule. Offline (no model, no API)."""
import json
import sys

sys.path.insert(0, ".")
from keyness import tokenize, _stem  # noqa: E402

_PUNCT_AFTER = ".,;:!?)\u2019\"'"


def gap_after(after):
    """Space the GT had after it: none if directly followed by punctuation."""
    return "" if after[:1] in _PUNCT_AFTER else " "


d = json.load(open("redactions_broad.json", encoding="utf-8"))
reds = d["redactions"]
print("entries:", len(reds))
errors = []
for r in reds:
    rid = r["id"]
    gt = r["ground_truth"]
    if not (r["before"] and r["after"] and r["doc_context"] and gt):
        errors.append(f"{rid}: missing field")
    lo, hi = r["min_chars"], r["max_chars"]
    if not (lo <= len(gt) <= hi):
        errors.append(f"{rid}: GT len {len(gt)} not in [{lo},{hi}]")
    if gt in r["doc_context"]:
        full = r["before"] + " " + gt + gap_after(r["after"]) + r["after"]
        if full != r["doc_context"]:
            errors.append(f"{rid}: reconstruction mismatch")
    else:
        # absent-entity: doc_context should be the released text with GT removed
        expected = r["before"] + gap_after(r["after"]) + r["after"]
        if expected != r["doc_context"]:
            errors.append(f"{rid}: absent doc_context mismatch")
    win = " ".join(r["before"].split()[-8:] + r["after"].split()[:8])
    wtoks = {_stem(t) for t in tokenize(win) if len(_stem(t)) >= 4}
    gtoks = {_stem(t) for t in tokenize(gt) if len(_stem(t)) >= 4}
    leak = gtoks & wtoks
    if leak:
        errors.append(f"{rid}: GT leaks into adjacency window: {sorted(leak)}")
    if rid == "sov2_sea_based":
        if "sea-based leg" not in r["before"]:
            errors.append(f"{rid}: expected first occurrence visible in before")
print("ERRORS:", errors if errors else "none")
sys.exit(1 if errors else 0)
