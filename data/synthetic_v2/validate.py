#!/usr/bin/env python3
"""Offline validator for data/synthetic_v2.

The validator intentionally checks the paper taxonomy from index.typ:
Type 1 may retain another target occurrence in the visible document; Types 2
and 3 must have no target occurrence in visible document context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from keyness import tokenize, _stem  # noqa: E402

HERE = Path(__file__).resolve().parent


def _tokens(text):
    return {_stem(t) for t in tokenize(text) if len(_stem(t)) >= 4}


def _count(text, target):
    return text.count(target)


def main() -> int:
    errors = []
    datasets = {}
    for split in ("train", "test"):
        path = HERE / f"{split}.json"
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        datasets[split] = data
        if data.get("schema") != "unredact-synthetic-v2":
            errors.append(f"{split}: wrong schema")
        if data.get("split") != split:
            errors.append(f"{split}: split metadata mismatch")

    train_docs = set(datasets.get("train", {}).get("documents", []))
    test_docs = set(datasets.get("test", {}).get("documents", []))
    overlap = train_docs & test_docs
    if overlap:
        errors.append(f"train/test document overlap: {sorted(overlap)}")

    for split, data in datasets.items():
        for entry in data.get("redactions", []):
            rid = entry.get("id", "?")
            gt = entry.get("ground_truth", "")
            before = entry.get("before", "")
            after = entry.get("after", "")
            visible = entry.get("doc_context", "")
            paper_type = entry.get("paper_type")
            if not gt or not before or not after or not visible:
                errors.append(f"{rid}: missing required text field")
                continue
            reconstructed = before + entry.get("gap_before", "") + gt + entry.get("gap_after", "") + after
            source_path = HERE / "docs" / split / f"{entry['doc_id']}.txt"
            source = source_path.read_text(encoding="utf-8").rstrip("\n")
            if reconstructed != source:
                errors.append(f"{rid}: before/after/GT does not reconstruct frozen source")
            expected_visible = before + entry.get("gap_before", "") + entry.get("gap_after", "") + after
            if visible != expected_visible:
                errors.append(f"{rid}: doc_context is not the visible document")
            lo, hi = entry.get("min_chars"), entry.get("max_chars")
            if not isinstance(lo, int) or not isinstance(hi, int) or not (lo <= len(gt) <= hi):
                errors.append(f"{rid}: GT length {len(gt)} outside [{lo},{hi}]")
            if entry.get("box_width_pts", 0) <= 0 or entry.get("char_advance_pt", 0) <= 0:
                errors.append(f"{rid}: invalid simulated box measurement")
            count = _count(visible, gt)
            if paper_type == 1:
                if count < 1:
                    errors.append(f"{rid}: Type 1 target absent from visible context")
            elif paper_type in (2, 3):
                if count:
                    errors.append(f"{rid}: Type {paper_type} target still appears in visible context")
            else:
                errors.append(f"{rid}: unknown paper_type {paper_type!r}")

            # No target content token is allowed in the local +-8-word window.
            local = " ".join(before.split()[-8:] + after.split()[:8])
            leaked = _tokens(local) & _tokens(gt)
            if leaked:
                errors.append(f"{rid}: target token(s) leak into local adjacency: {sorted(leaked)}")

    print(f"train documents: {len(train_docs)}; test documents: {len(test_docs)}")
    print(f"train spans: {len(datasets.get('train', {}).get('redactions', []))}; "
          f"test spans: {len(datasets.get('test', {}).get('redactions', []))}")
    print("ERRORS:", errors if errors else "none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
