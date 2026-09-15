# Unredact

[![Typst Paper](https://img.shields.io/badge/Paper-index.pdf-red.svg)](index.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unredact** is an experimental framework for reverse box-redaction in monospace declassified documents using the physical side-channel leak of bounding box character lengths combined with non-autoregressive Diffusion Language Models (DLMs) and semantic candidate verification.

📄 **Read the Paper:** [**index.pdf**](index.pdf) (Written in Typst, ACL/TRACL template)

---

## 🎯 The Core Concept

In monospace documents (such as FOIA releases, CIA CREST reading room files, and typewriter-era diplomatic telegrams), font glyphs have fixed advance widths. While a black box overlay destroys visual pixel data, its physical dimensions directly leak the character length bounds $[L_{\min}, L_{\max}]$.

```text
Visible Context (Prefix / Suffix) ──┐
                                     ├──> DLM Infilling (LLaDA) ──> Character-Range Pruning ──> Semantic Ranking
Leaked Box Geometry [L_min, L_max] ─┘
```

### Redaction Taxonomy
1. **Type 1 (Juxtaposition / Coreference):** Redacted entity appears elsewhere in document. *Highly recoverable.*
2. **Type 2 (Total Expungement):** Concept is purged from document, but constrained by grammar/local context. *Grammatically constrained.*
3. **Type 3 (Constrained / Situational):** Absent entity requiring external domain knowledge. *Partial/Role-based recovery.*

## Quick start (tonight)

1. Install deps:  `python -m pip install -r requirements.txt`
2. Get a free Mercury API key (diffusion LLM, ~100M free tokens):
   sign up at https://inceptionlabs.ai, then store it in a `.env` file in this
   directory (the script auto-loads it, no extra packages needed):

   ```
   INCEPTION_API_KEY=sk-...
   ```

   (Or set it as an environment variable: `export INCEPTION_API_KEY=...` on
   bash, `set INCEPTION_API_KEY=...` on Windows cmd. The `.env` file is
   git-ignored so your key stays local.)

   For `--semantic` (which downloads all-MiniLM-L6-v2 from Hugging Face),
   add your HF token the same way to avoid rate-limit warnings on first
   download -- huggingface_hub reads it straight from the environment, and
   the script loads `.env` before the model is fetched:

   ```
   HF_TOKEN=hf_...
   ```
3. Edit `redactions.json`:
   - `before` / `after`: visible text around the box. Use the FULL document
     context, split exactly at the redaction gap.
   - `min_chars` / `max_chars`: how many characters fit in the box. Derive
     these from physical measurements with `box_measure.py` (below).
   - `ground_truth` (optional): the actual redacted text, when you know it.
     The script then reports exact-match and similarity scores per redaction
     and prints a summary at the end.
4. Run:

   ```
   python unredact.py --backend mercury --mode chat   # tighter length control
   python unredact.py --backend mercury --mode fim    # diffusion-native infill
   ```

   FIM mode pools candidates across local context windows and token budgets,
   filters FIM artifacts (empty/punctuation/multi-line output and verbatim
   copies of the surrounding text), and -- when FIM yields nothing inside the
   box's char range -- blends in chat-mode candidates so the pool always has
   length-verified options to rank. Note: mercury-edit-2's FIM endpoint is
   unreliable on prose (it either nails short gaps or returns empty/multi-line/
   out-of-range junk); that is why the fallback exists. Generate-and-verify is
   the design, not a workaround.

   Score semantic fit (optional, recommended):

   ```
   pip install sentence-transformers        # Windows CPU: install torch --index-url https://download.pytorch.org/whl/cpu first
   python unredact.py --backend mercury --mode chat --semantic
   ```

   `--semantic` computes an **oracle diagnostic contrastive slot score** using
   `all-MiniLM-L6-v2`: it embeds `before + candidate + after` vs
   `before + ground_truth + after` and reports how much closer the candidate is
   to the ground truth than a control token is. This is useful for analyzing
   ranking quality, but it uses the answer and is not blind deployment
   performance. Embeddings use a **local
   window around the slot** (last/first `--semantic-window` words, default 8)
   so the shared context doesn't saturate the cosine; char-range verification
   still uses the full context. Scores run 1.0 (identical to GT) down to 0.0
   (no better than the control `zzzzzz`; change it with `--semantic-control`)
   and can go negative (worse than the control). If the control anchors too
   close to the GT (cosine > 0.98), the metric is unreliable and scoring is
   skipped for that box with a warning. Each box is classified
   **exact / near-miss / off-target** (threshold `--semantic-threshold`,
   default 0.7) and a graded recovery summary is printed.

   **Adjacency-echo guard (on by default, `--echo-penalty`):** the contrastive
   anchor cannot punish a candidate that simply reproduces a word sitting
   right next to the gap -- an echoed word is always closer to the GT slot
   than a nonsense control. So any candidate that is mostly composed of
   adjacent-window words (the "Indian"/"Chinese" failure mode, where the
   model copies the first word of the after-text) gets `echo_penalty` (default
   0.5) subtracted from its score, and echoed candidates also receive **no
   keyness-prior credit** (so the prior can't re-boost an echo). Verified
   against the test set: none of the ground truths appear in their adjacent
   window (each was redacted *out* of it), so the guard never touches a
   correct answer. Disable with `--echo-penalty 0`.

   No API key yet? Test the whole pipeline offline:

   ```
   python unredact.py --dry-run
   ```

## Local LLaDA backend

The native masked-diffusion LLaDA server runs on a GPU machine while this
repository remains the evaluator. The server is a small unauthenticated HTTP
service, so keep it on a trusted private network and do not port-forward it to
the public internet.

### Start the server

On the GPU machine, install the server dependencies and a CUDA-enabled PyTorch
build, then start the server. The default binds to all interfaces on port 8000
so another machine on the LAN can reach it:

```bash
python -m pip install -r llada_server_requirements.txt
# Install the CUDA-enabled PyTorch build appropriate for your machine separately.
python llada_server.py --model GSAI-ML/LLaDA-8B-Base --host 0.0.0.0 --port 8000
```

The server exposes:

- `GET /health` — readiness and model/device information.
- `POST /generate` — visible `before`/`after` text plus character bounds and
  sampling settings; returns a list of candidate strings and diagnostics.

It never receives `ground_truth`, document IDs, paper taxonomy, or evaluator
metadata. Returned candidates still require client-side character-range
verification.

### Run the evaluator

On the evaluator machine, use the server's reachable host and port. For a
server on the same machine, the default is `127.0.0.1:8000`; for a GPU desktop
on the LAN, replace the host with that machine's private IP:

```bat
set LLADA_HOST=GPU-DESKTOP-IP
set LLADA_PORT=8000
run_llada_tests.bat redactions.json llada_low low_confidence
```

The batch runner accepts the following arguments:

```text
run_llada_tests.bat [redactions] [tag] [remasking] [host] [port]
```

Environment variables are also supported:

```bat
set LLADA_URL=http://gpu-desktop.example:8000
run_llada_tests.bat redactions.json llada_random random
```

`LLADA_URL` takes precedence over `LLADA_HOST` and `LLADA_PORT`. The runner
checks `/health`, sends a tiny `/generate` smoke request, then executes eight
serial 64-step candidate requests and logs the result. Use `low_confidence` or
`random` for the remasking strategy.

To invoke the evaluator directly:

```bash
python unredact.py --backend llada \--llada-host GPU-DESKTOP-IP --llada-port 8000
 \
  --llada-steps 64 --llada-temperature 0.8 \
  --llada-remasking low_confidence --candidates 8 \
  --redactions redactions.json
```

Or provide the complete endpoint URL:

```bash
python unredact.py --backend llada --llada-url http://127.0.0.1:8000 \
  --llada-steps 64 --llada-temperature 0.8 \
  --llada-remasking low_confidence --candidates 8 \
  --redactions redactions.json
```

Useful validation commands, which do not require the LLaDA server, are:

```bash
python validate_dataset.py
python data/synthetic_v2/validate.py
python unredact.py --dry-run --redactions redactions.json
python unredact.py --dry-run --redactions redactions.json --n-eff
python log_run.py n_eff --redactions redactions_broad.json --dry-run --n-eff --candidates 8
typst compile index.typ index.pdf
```

`--llada-steps` defaults to 64 and the client derives nearby token-span
requests from each redaction's character upper bound. LLaDA generates a fixed
token span rather than stopping at a character boundary, so all candidates are
filtered locally against the authoritative character range.

## Keyness prior (document bias in ranking)

The pipeline is generate -> verify (char range) -> score (contrastive
semantic embedding). A **document-keyness prior** adds a third, independent
term to that scorer so candidate ranking is biased toward what this document
is *about* -- and away from what it is *saturated with*:

```
final_score = semantic_contrastive + prior_weight * keyness_prior(candidate)
```

The prior is **two-sided**: it rewards thematic terms while penalizing
entities that dominate the document, reducing the salience-trap failure mode.

- **Positive** weight on the document's thematic terms and entity shortlist
  (words whose frequency is anomalously high relative to general English --
  `missile`, `IRBM`, `India`, `China` -- computed with a small embedded
  frequency table; no external dependency).
- **Negative** weight on the document's own dominant entities  (the *salience trap*: an earlier Pakistan/Israel case exposed how a saturated entity can dominate the context; that case is excluded from the active six-case benchmark).

It never filters or corrupts generation -- it only nudges selection, so the
common-word ground truths ("active", "large") are untouched.

```
# A/B against the baseline (baseline = same command without --prior-weight):
python unredact.py --backend mercury --mode fim --semantic --prior-weight 0.5
```

Note: with `--prior-weight` active, the `GRADED RECOVERY` summary reports the
**blended final score** (contrastive + prior), which can exceed 1.0 -- it is
labeled "mean best-candidate final score" in that case, not raw contrastive.

Requirements: `--semantic` (the prior blends into semantic scores) and a
`doc_context` field in `redactions.json` holding the full document text the
redaction comes from (whole-document *statistics* only -- generation stays
windowed, since FIM chokes on long context). Tune the blend with
`--prior-weight` (try 0.3-0.5; 0 = off) and see what the extractor found in
the `[keyness]` line per redaction.

## Constraint-tightness ablation (does the leaked length do work?)

The whole pipeline leans on the box's leaked char range `[min_chars, max_chars]`.
This ablation quantifies how much of the recovery actually comes from that
leak -- the data for the "constraint-tightness vs accuracy" chart.

- **`--loosen <delta>`** -- the full-pipeline version: widen every box by
  `delta` characters on **both** sides (`[min-delta, max+delta]`) before
  generation *and* verification (chat length hints loosen too). Run it at
  several deltas and watch recovery fall:

  ```
  python unredact.py --redactions redactions_broad.json --backend mercury \
    --semantic --prior-weight 0.5 --loosen 3
  ```

- **`--tightness-sweep 0,3,5,10`** -- the selection-stage version: generates
  ONE candidate pool per redaction (at the tight bounds), then re-verifies,
  re-ranks and re-grades that same pool at every widening. Zero extra API
  calls, deterministic, and it isolates exactly what the length constraint
  contributes at selection. Prints a `COMPARISON: recovery vs looseness`
  table at the end:

  ```
  python unredact.py --redactions redactions_broad.json --backend mercury \
    --semantic --prior-weight 0.5 --tightness-sweep 0,3,5,10
  ```

**What the data says (logged runs, broad set):** the pool-reuse sweep is
*flat* -- recovery does not fall as the bounds loosen. Because generation is
already length-conditioned and the semantic scorer ranks the right fill first,
widening the eligible set barely moves recovery. The leaked length's real
contribution is at **generation**, not selection. The decisive comparison is
chat mode with vs without the length hint:

  ```
  # Plain-LLM baseline: no leaked length in the prompt.
  python log_run.py no_len_hint --redactions redactions_broad.json \
    --backend mercury --mode chat --semantic --no-length-hint
  ```

## Evaluation logging (track improvement over time)

Every run can be logged as a full text log + a graphable sidecar:

  ```
  python log_run.py <tag> [unredact.py args...]
  ```

Writes into `logs/`: `<ts>_<tag>.txt` (full text), `<ts>_<tag>.json`
(structured sidecar: per-redaction rows incl. `control_sim` and optional
`N_eff` difficulty diagnostics + summary), and `runs.csv` (one row per run --
the improvement-progress file). See `logs/README.md`. The sidecar is produced
by unredact.py's `--json-results` flag, so any run can be logged, not just via
the wrapper.

### Ground-truth-free difficulty (`N_eff`)

Use `--n-eff` to estimate how concentrated the generated candidate support is
for each box, without reading `ground_truth`:

```bash
python log_run.py n_eff --redactions redactions_broad.json \
  --backend mercury --mode fim --candidates 8 --n-eff
```

The diagnostic computes `N_eff = exp(H)` from the empirical frequencies of
cleaned candidate draws. Repeated draws count: a pool that repeatedly returns
one normalized fill has low `N_eff`, while a pool spread across many fills has
higher `N_eff`. The main reported value is computed after the leaked character
range is applied; the sidecar also records the unfiltered pool. This is an
**observed-support proxy**, not the model's true conditional entropy or a
calibrated probability of recovery: the backend exposes no token likelihoods,
small pools miss unseen outcomes, and generation/order effects remain. Ordinary
ranking still deduplicates repeated strings; only the diagnostic preserves their
sampling multiplicity.
Empty in-range pools are reported per redaction and excluded from aggregate
mean/median. Treat `N_eff` as a hypothesis-generating hardness signal; validate
any relationship with blind recovery on repeated runs rather than claiming the
thresholds in the roadmap as established facts.

## Token usage (Mercury free tier)

FIM is the cheap path (~30 tokens/call); chat is ~7-10x more expensive
(mercury-2 reasons before answering, ~140 reasoning tokens/call uncapped).
The pipeline already trims spend:

- FIM stops early: once 3 usable candidates are pooled, the ground truth is
  hit, or after one window sweep if nothing fits the box's char range.
- Chat is capped at 3 candidates per box in every mode (fallback and explicit
  `--mode chat`); `--candidates` mainly scales FIM calls.
- Chat requests use `reasoning: {"effort": "low"}`, which cuts reasoning
  tokens roughly in half (~140 -> ~65) with no quality loss measured.

A full 6-box run with `--candidates 6` typically costs ~2-4k tokens. Tune
spend with `--candidates` (FIM calls per box). Pass `--full-pool` on eval
runs to lift the early-stop caps and gather the whole candidate pool for
paper analysis (at higher token cost).

## Measuring boxes

The infill pipeline is text-only; convert measurements to a character range
with `box_measure.py` (monospace typewriter assumed: Courier advance = 0.6 em,
e.g. 12pt = 7.2pt per char = 10 CPI):

```
python box_measure.py --width-pts 50 --font-pt 12   # box width in PDF points
python box_measure.py --width-in 0.7 --font-pt 10   # width in inches
python box_measure.py --width-mm 18 --font-pt 12    # width in millimeters
python box_measure.py --cpi 10 --width-pts 72       # when chars/inch is known
```

It prints the estimate plus the `min_chars`/`max_chars` snippet to paste into
`redactions.json`.

## Backends

| name      | status  | notes                                        |
|-----------|---------|----------------------------------------------|
| mercury   | working | FIM (diffusion-native infill) or chat mode   |
| echo      | working | dummy candidates for offline pipeline tests  |
| llada     | working | local LLaDA masked-infill HTTP server |
| openai    | stub    |                                              |
| anthropic | stub    |                                              |

Add a backend by subclassing the interface: `generate(red, n) -> list[str]`,
then register it in the `BACKENDS` dict in `unredact.py`.
