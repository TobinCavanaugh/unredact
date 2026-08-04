# Synthetic v2 virtual dataset

This is a **text-only** evaluation corpus. It intentionally stops before OCR,
PDF parsing, box detection, and span detection so those error sources do not
confound the unredaction model experiments.

## Canonical taxonomy

The labels follow `index.typ`:

1. **Juxtaposition / coreference** — the selected target occurrence is hidden,
   but another occurrence of the target remains elsewhere in the visible
   document. This is the direct in-document redundancy case.
2. **Total expungement** — the target is absent from the visible document and
   the document gives no useful way to identify it. Exact recovery is generally
   impossible from text alone.
3. **Constrained / situational** — the target is absent from the visible
   document, but grammar, facts, or document logic narrow the answer space.

The old `category` field is retained only as a compatibility shorthand:
`theme-echo` for Type 1, `absent-entity` for Type 2, and `recoverable` for
Type 3. New evaluations should use `paper_type`, `paper_type_name`, and
`evidence_mode`.

## Layout

```text
data/synthetic_v2/
  generate.py
  validate.py
  README.md
  manifest.json
  train.json
  test.json
  docs/
    train/*.txt
    test/*.txt
```

The split is by complete document, never by redaction span. Current size:

- train: 4 documents, 19 spans
- test: 2 documents, 10 spans

## Rebuild and validate

From the repository root:

```bash
python data/synthetic_v2/generate.py
python data/synthetic_v2/validate.py
```

Generation is deterministic. Complete documents are defined first in
`generate.py`; span specifications select occurrences afterward. For each span:

- `before` and `after` are the visible text around the selected occurrence;
- `doc_context` is the visible document (`before + after`), never a target-bearing
  full document;
- Type 1 may contain another target occurrence elsewhere;
- Types 2 and 3 must contain no target occurrence anywhere in `doc_context`;
- the target content tokens may not appear in the local ±8-word adjacency window;
- `min_chars` and `max_chars` are produced from a simulated monospace box width,
  font advance, and deterministic measurement factor/tolerance.

The JSON files are compatible with the current `unredact.py` loader. Ground
truth remains in the dataset for post-run evaluation only. The runner now
reports blind top-1 separately from candidate recall@K and the old
ground-truth-based semantic result, which is labeled an oracle diagnostic and
must not be treated as deployed performance.
