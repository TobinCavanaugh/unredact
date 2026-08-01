# Unredact MVP (manual-first, pure text)

Length-constrained infilling of redacted text using a diffusion LLM.

The idea: a redaction box leaks its length. We estimate how many characters
fit in the box (from the box width and font metrics), generate candidate
infills from a diffusion LLM given the surrounding document text, then keep
only the candidates whose length fits. That generate-and-verify loop is the
core contribution.

**Key finding so far: context is everything -- but the two Mercury modes want
it differently.** Chat mode recovers far better when given the full surrounding
text split at the gap. FIM mode (diffusion-native infill) chokes on long
context (it returns a bare space when handed full paragraphs), so it is fed a
short local window around the gap instead, and we pool candidates across
window sizes and token budgets (the generate half of generate-and-verify).

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

   `--semantic` computes a **contrastive slot score** using `all-MiniLM-L6-v2`:
   it embeds `before + candidate + after` vs `before + ground_truth + after`
   and reports how much closer the candidate is to the ground truth than a
   control token is (raw cosine is dominated by shared context, so the
   control-anchored score is what discriminates). Embeddings use a **local
   window around the slot** (last/first `--semantic-window` words, default 8)
   so the shared context doesn't saturate the cosine; char-range verification
   still uses the full context. Scores run 1.0 (identical to GT) down to 0.0
   (no better than the control `zzzzzz`; change it with `--semantic-control`)
   and can go negative (worse than the control). If the control anchors too
   close to the GT (cosine > 0.98), the metric is unreliable and scoring is
   skipped for that box with a warning. Each box is classified
   **exact / near-miss / off-target** (threshold `--semantic-threshold`,
   default 0.7) and a graded recovery summary is printed.

   No API key yet? Test the whole pipeline offline:

   ```
   python unredact.py --dry-run
   ```

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

A full 7-box run with `--candidates 6` typically costs ~2-4k tokens. Tune
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
| llada     | stub    | local LLaDA-8B later (needs torch/transformers) |
| openai    | stub    |                                              |
| anthropic | stub    |                                              |

Add a backend by subclassing the interface: `generate(red, n) -> list[str]`,
then register it in the `BACKENDS` dict in `unredact.py`.
