#!/usr/bin/env python3
"""Generate the taxonomy-aware virtual evaluation corpus.

This dataset deliberately stops before OCR and span detection. Complete
synthetic documents are frozen in this file first; span specifications select
occurrences afterward. The emitted train/test JSON remains compatible with
unredact.py's before/after/min_chars/max_chars/doc_context fields.

Canonical paper taxonomy (index.typ):
  Type 1: Juxtaposition / coreference -- answer appears elsewhere in-document.
  Type 2: Total expungement -- answer leaves no useful internal trace.
  Type 3: Constrained / situational -- answer is absent but document logic
           narrows the answer space.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOCS = OUT / "docs"

# Same monospace model as box_measure.py, but with a deterministic simulated
# measurement error. The range is derived from physical width, not hand-set
# around the answer as a text-level margin.
FONT_PT = 12.0
ADVANCE_EM = 0.6
MEASUREMENT_TOLERANCE = 0.08
MEASUREMENT_FACTORS = (1.00, 0.98, 1.02, 0.97, 1.03, 0.99)

_PUNCT_AFTER = ".,;:!?)\u2019\"'"

PAPER_TYPES = {
    1: {
        "name": "juxtaposition_coreference",
        "label": "Juxtaposition / coreference",
        "evidence_mode": "verbatim_elsewhere",
    },
    2: {
        "name": "total_expungement",
        "label": "Total expungement",
        "evidence_mode": "absent_unconstrained",
    },
    3: {
        "name": "constrained_situational",
        "label": "Constrained / situational",
        "evidence_mode": "document_entailment",
    },
}

# Frozen complete documents. Span selection happens below, after these strings
# are defined, so the target is never authored into a special redacted context.
DOCUMENTS = {
    "soviet_briefing": {
        "split": "train",
        "text": (
            "The assessment reviews Soviet strategic forces and their changing "
            "deployment patterns. The sea-based leg remains the most resilient "
            "part of the triad, while mobile launchers complicate collection. "
            "Analysts judge that the sea-based deterrent will remain central to "
            "the national posture through the decade. Field reporting indicates "
            "that mobile launchers have moved repeatedly between prepared sites. "
            "The command authority for the northern district has requested more "
            "secure communications. A later annex states that the command authority "
            "must approve any relocation of the reserve units. The force must remain "
            "survivable after a first strike, although the report gives no reliable "
            "estimate of the reserve crews. The only Nordic state with the required "
            "tracking station is Norway, according to a separate engineering note. "
            "The memorandum was signed by Colonel Petrov, but it provides no useful "
            "description of his assignment or personal history."
        ),
        "spans": [
            {"target": "sea-based", "paper_type": 1, "occurrence": 2},
            {"target": "mobile launchers", "paper_type": 1, "occurrence": 2},
            {"target": "command authority", "paper_type": 1, "occurrence": 2},
            {"target": "survivable", "paper_type": 3, "occurrence": 1,
             "note": "The post-strike requirement constrains the adjective, but the exact word is not repeated."},
            {"target": "Colonel Petrov", "paper_type": 2, "occurrence": 1,
             "note": "Unique signatory with no internal identifying evidence."},
        ],
    },
    "china_policy": {
        "split": "train",
        "text": (
            "Chinese strategic policy has emphasized restraint while the missile "
            "force expands. The first directive restates a no-first-use pledge and "
            "describes the policy as defensive. A later paragraph says that the "
            "no-first-use pledge is intended to reduce pressure during a crisis. "
            "The new solid-propellant missile can be stored for long periods with "
            "limited preparation. Intelligence officers expect another "
            "solid-propellant missile to enter testing before the end of the decade. "
            "The leadership describes retaliation as the purpose of the force, and "
            "the annex says that retaliation would follow any confirmed attack. "
            "The document does not identify the engineer who approved the design; "
            "the approval line names Liu Wen without a biography. The language of "
            "the directive is sufficiently unambiguous to rule out a launch on a "
            "mere warning."
        ),
        "spans": [
            {"target": "no-first-use", "paper_type": 1, "occurrence": 2},
            {"target": "solid-propellant", "paper_type": 1, "occurrence": 2},
            {"target": "retaliation", "paper_type": 1, "occurrence": 2},
            {"target": "unambiguous", "paper_type": 3, "occurrence": 1,
             "note": "The policy sentence constrains the required meaning, but the target is absent elsewhere."},
            {"target": "Liu Wen", "paper_type": 2, "occurrence": 1,
             "note": "A named approver is present only as an unexplained signature."},
        ],
    },
    "gulf_supply": {
        "split": "train",
        "text": (
            "Western planners continue to treat the Strait of Hormuz as the main "
            "chokepoint for Gulf exports. Any closure of the Strait of Hormuz would "
            "force governments to draw on stored supplies. The report compares "
            "pipeline routes through Turkey with alternate corridors along the Red Sea, "
            "but notes that neither route can replace the lost volume quickly. The "
            "annex later returns to pipeline routes after discussing port capacity. The "
            "strategic reserve is adequate for several weeks, and a later table "
            "estimates that the strategic reserve would be exhausted first in the "
            "smaller importing states. The most effective emergency measure would "
            "be rationing rather than a complete embargo. The cable was approved by "
            "General Farouk, whose role is not explained anywhere in the assessment."
        ),
        "spans": [
            {"target": "Strait of Hormuz", "paper_type": 1, "occurrence": 2},
            {"target": "pipeline routes", "paper_type": 1, "occurrence": 2},
            {"target": "strategic reserve", "paper_type": 1, "occurrence": 2},
            {"target": "rationing", "paper_type": 3, "occurrence": 1,
             "note": "The comparison with a complete embargo narrows the emergency measure."},
            {"target": "General Farouk", "paper_type": 2, "occurrence": 1,
             "note": "The signatory cannot be identified from the document itself."},
        ],
    },
    "dprk_program": {
        "split": "train",
        "text": (
            "Earlier cables refer to foreign suppliers without technical detail. "
            "The Korean program continues to seek a survivable nuclear capability. "
            "A preliminary cable mentions outside contractors but gives no technical "
            "details. Inspectors report that fissile material is stored at two declared "
            "locations, while another source estimates that fissile material may be "
            "held at an undisclosed site. The report attributes the procurement to "
            "outside contractors, and a footnote says that foreign suppliers have "
            "provided machine tools as well as technical advice. The medium-range "
            "system is approaching an operational test, but the warhead design "
            "remains uncertain. A reliable design must be demonstrated before the "
            "system can be treated as a strategic threat. The annex credits Deputy "
            "Minister Pak Yong-chol with the schedule but gives no further detail."
        ),
        "spans": [
            {"target": "fissile material", "paper_type": 1, "occurrence": 2},
            {"target": "foreign suppliers", "paper_type": 1, "occurrence": 2},
            {"target": "demonstrated", "paper_type": 3, "occurrence": 1,
             "note": "The requirement is strongly constrained by the preceding design discussion."},
            {"target": "Deputy Minister Pak Yong-chol", "paper_type": 2, "occurrence": 1,
             "note": "Unique official title and name with no redundant evidence."},
        ],
    },
    "south_asia_assessment": {
        "split": "test",
        "text": (
            "The assessment concerns a proposed transfer of long-range equipment in "
            "South Asia. Earlier reporting says that the equipment is intended for "
            "a regional partner, while a later paragraph discusses training schedules "
            "and fuel requirements. The map annex later repeats the phrase regional "
            "partner. The border units are described in the preliminary map, while "
            "the border force is named only in a later annex, although the annex "
            "also refers to a border force near the frontier and uses a different "
            "scale. The only South Asian state with the required launch facilities "
            "is India. The decision was recorded by Ambassador Rao, whose identity "
            "is not otherwise explained. The transfer would be defensive if the "
            "recipient accepted inspection, but the report does not name the final "
            "negotiating team."
        ),
        "spans": [
            {"target": "regional partner", "paper_type": 1, "occurrence": 2},
            {"target": "border force", "paper_type": 1, "occurrence": 2},
            {"target": "India", "paper_type": 3, "occurrence": 1,
             "note": "The facility and regional constraint narrow the country answer; the name is not repeated."},
            {"target": "Ambassador Rao", "paper_type": 2, "occurrence": 1,
             "note": "The document gives no internal evidence identifying the ambassador."},
            {"target": "defensive", "paper_type": 3, "occurrence": 1,
             "note": "The inspection condition constrains the policy characterization."},
        ],
    },
    "arctic_shipping": {
        "split": "test",
        "text": (
            "The seasonal shipping review records that sea ice delayed the eastern "
            "convoy during the first survey. A second table projects that sea ice "
            "will remain the principal delay through the next cycle. The northern "
            "route is open for only a short period, and the appendix says that the "
            "northern route requires advance weather reporting. The safest response "
            "to a late freeze would be an icebreaker escort rather than a larger "
            "cargo manifest. The report names Captain Sorensen as the officer who "
            "approved the route, but does not describe his experience. The fleet "
            "could continue if the weather window remained stable, although the "
            "estimate excludes unreported mechanical failures."
        ),
        "spans": [
            {"target": "sea ice", "paper_type": 1, "occurrence": 2},
            {"target": "northern route", "paper_type": 1, "occurrence": 2},
            {"target": "icebreaker escort", "paper_type": 3, "occurrence": 1,
             "note": "The contrast with a cargo manifest and late freeze constrains the operational response."},
            {"target": "Captain Sorensen", "paper_type": 2, "occurrence": 1,
             "note": "Unique approving officer with no other identifying information."},
            {"target": "stable", "paper_type": 3, "occurrence": 1,
             "note": "The continuation condition constrains the required weather property."},
        ],
    },
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _find_occurrence(text: str, target: str, occurrence: int) -> int:
    start = 0
    found = -1
    for _ in range(occurrence):
        found = text.find(target, start)
        if found < 0:
            raise ValueError(f"target {target!r} occurrence {occurrence} not found")
        start = found + len(target)
    return found


def _bounds(target: str, span_number: int) -> dict:
    factor = MEASUREMENT_FACTORS[(span_number - 1) % len(MEASUREMENT_FACTORS)]
    advance_pt = ADVANCE_EM * FONT_PT
    width_pts = len(target) * advance_pt * factor
    estimated_chars = width_pts / advance_pt
    lo = max(1, math.floor(estimated_chars * (1.0 - MEASUREMENT_TOLERANCE)))
    hi = max(lo, math.ceil(estimated_chars * (1.0 + MEASUREMENT_TOLERANCE)))
    return {
        "min_chars": lo,
        "max_chars": hi,
        "box_width_pts": round(width_pts, 3),
        "font_pt": FONT_PT,
        "char_advance_pt": round(advance_pt, 3),
        "measurement_factor": factor,
        "measurement_tolerance": MEASUREMENT_TOLERANCE,
    }


def _entry(doc_id: str, spec: dict, number: int, source: str) -> dict:
    target = spec["target"]
    occurrence = spec.get("occurrence", 1)
    idx = _find_occurrence(source, target, occurrence)
    raw_before = source[:idx]
    raw_after = source[idx + len(target):]
    before = raw_before.rstrip()
    after = raw_after.lstrip()
    gap_before = raw_before[len(before):]
    gap_after = raw_after[:len(raw_after) - len(after)]
    if before + gap_before + target + gap_after + after != source:
        raise AssertionError(f"{doc_id}: reconstruction failed for {target!r}")

    # Preserve the spaces around the removed span. The exact original target
    # spacing remains in gap_* fields for reconstruction.
    visible = before + gap_before + gap_after + after
    info = PAPER_TYPES[spec["paper_type"]]
    bounds = _bounds(target, number)
    entry = {
        "id": f"{doc_id}_span{number}_{slug(target)}",
        "doc_id": doc_id,
        "paper_type": spec["paper_type"],
        "paper_type_name": info["name"],
        "evidence_mode": info["evidence_mode"],
        "target_occurrence": occurrence,
        "before": before,
        "after": after,
        "doc_context": visible,
        "ground_truth": target,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "category": {
            1: "theme-echo",
            2: "absent-entity",
            3: "recoverable",
        }[spec["paper_type"]],
        "note": spec.get("note"),
        **bounds,
    }
    return entry


def main() -> None:
    for split in ("train", "test"):
        (DOCS / split).mkdir(parents=True, exist_ok=True)
    for doc_id, spec in DOCUMENTS.items():
        path = DOCS / spec["split"] / f"{doc_id}.txt"
        path.write_text(spec["text"] + "\n", encoding="utf-8")

    grouped = {"train": [], "test": []}
    for doc_id, spec in DOCUMENTS.items():
        source = spec["text"]
        for number, span in enumerate(spec["spans"], 1):
            grouped[spec["split"]].append(_entry(doc_id, span, number, source))

    for split, entries in grouped.items():
        payload = {
            "schema": "unredact-synthetic-v2",
            "split": split,
            "taxonomy": "index.typ",
            "ground_truth_policy": (
                "ground_truth is evaluation-only; doc_context is the visible "
                "document with the selected occurrence removed. Type 1 permits "
                "a separate occurrence elsewhere; Types 2 and 3 require the "
                "target to be absent from the visible document."
            ),
            "documents": sorted({e["doc_id"] for e in entries}),
            "redactions": entries,
        }
        (OUT / f"{split}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    manifest = {
        "schema": "unredact-synthetic-v2-manifest",
        "taxonomy_source": "index.typ",
        "documents": {
            doc_id: {
                "split": spec["split"],
                "path": str(Path("docs") / spec["split"] / f"{doc_id}.txt"),
                "spans": len(spec["spans"]),
            }
            for doc_id, spec in sorted(DOCUMENTS.items())
        },
        "counts": {
            split: len(entries) for split, entries in grouped.items()
        },
        "paper_types": PAPER_TYPES,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote synthetic v2 dataset")
    print(f"  train: {len(grouped['train'])} spans across "
          f"{sum(1 for s in DOCUMENTS.values() if s['split'] == 'train')} documents")
    print(f"  test:  {len(grouped['test'])} spans across "
          f"{sum(1 for s in DOCUMENTS.values() if s['split'] == 'test')} documents")
    for split, entries in grouped.items():
        counts = {}
        for e in entries:
            counts[e["paper_type"]] = counts.get(e["paper_type"], 0) + 1
        print(f"  {split} paper types: {counts}")


if __name__ == "__main__":
    main()
