#!/usr/bin/env python3
"""
Manual-first MVP: pure-text, length-constrained infilling of redacted text.

This stage of the pipeline is text-only. For each redaction you provide:
  - `before` / `after`: the visible text around the box -- use the FULL
    surrounding document context, split right at the gap. More context is
    the single biggest factor in recovery quality.
  - `min_chars` / `max_chars`: how many characters fit in the box. Derive
    these from physical measurements with the separate box_measure.py tool.

Pipeline per redaction:
  1. Load context + character range from the JSON input
  2. Ask a diffusion-LLM backend to generate candidate infills
  3. Keep candidates whose length falls within [min_chars, max_chars]
  4. If `ground_truth` is known, report exact match + similarity

Backends:
  mercury (default)  Inception Labs Mercury (a diffusion LLM), free tier API.
  echo               No-network dummy backend for testing the pipeline (--dry-run).
  llada              Local LLaDA masked-infill HTTP server on the GPU desktop.
  openai / anthropic STUBS -- drop-in points for other models later.

Usage:
  python unredact.py --backend mercury --redactions redactions.json
  python unredact.py --dry-run        # test the whole pipeline with no API key
"""

import argparse
import difflib
import html
import json
import os
import sys
from datetime import datetime, timezone

# Reuse the keyness tokenizer/stemmer (stdlib-only module) so the echo guard
# and the prior normalize words the same way.
from keyness import tokenize, _stem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# NOTE: no trailing /v1 here -- endpoints below include it.
MERCURY_BASE = "https://api.inceptionlabs.ai"
MERCURY_FIM_MODEL = "mercury-edit-2"   # v1/fim/completions  -> choices[0].text
MERCURY_CHAT_MODEL = "mercury-2"       # v1/chat/completions -> choices[0].message.content

# ---------------------------------------------------------------------------
# .env support (no third-party dependency)
# ---------------------------------------------------------------------------


def load_dotenv(path=None):
    """Minimal .env loader: KEY=VALUE lines, '#' comments, quotes stripped.

    Never overrides variables already set in the environment, and silently
    ignores a missing file. Searches the script's own directory first, then
    the current working directory (so it works regardless of CWD). Uses
    utf-8-sig to tolerate a UTF-8 BOM added by Windows editors. Good enough
    for an API key.
    """
    candidates = [path] if path else []
    candidates += [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), ".env"]
    for p in candidates:
        if not p:
            continue
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except FileNotFoundError:
            continue


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def load_redactions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    redactions = data.get("redactions", [])
    for r in redactions:
        missing = [k for k in ("before", "after") if not str(r.get(k, "")).strip()]
        if missing:
            raise ValueError(
                f"Redaction {r.get('id', '?')} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        r["min_chars"] = max(1, int(r.get("min_chars", 1)))
        r["max_chars"] = int(r.get("max_chars", r["min_chars"] + 4))
        if r["max_chars"] < r["min_chars"]:
            r["max_chars"] = r["min_chars"]
    return redactions


# ---------------------------------------------------------------------------
# Length verification (pure text)
# ---------------------------------------------------------------------------


def verify_candidate(text, red):
    """Check a candidate's length against the box's character range."""
    n_chars = len(text)
    in_range = red["min_chars"] <= n_chars <= red["max_chars"]
    return in_range, {
        "text": text,
        "chars": n_chars,
        "char_range": (red["min_chars"], red["max_chars"]),
        "fits_chars": in_range,
    }


def rank_candidates(candidates, red):
    """Verify every candidate, then rank: in-range first, then closest to the
    center of the char range."""
    verified = [verify_candidate(c, red) for c in candidates]
    mid = (red["min_chars"] + red["max_chars"]) / 2.0
    verified.sort(key=lambda v: (not v[0], abs(v[1]["chars"] - mid)))
    return verified


# ---------------------------------------------------------------------------
# Semantic scoring (contextual slot similarity)
# ---------------------------------------------------------------------------


class SemanticScorer:
    """Contextual slot-similarity using all-MiniLM-L6-v2 (lazy-loaded).

    Measures how well a candidate fits the sentence slot by comparing
    embeddings of (before + candidate + after) vs (before + ground_truth + after).
    The model is only loaded when --semantic is used, so the base pipeline
    has no heavy dependency.

    Raw sentence cosine is dominated by the shared context (two sentences that
    differ in one word can still score >0.9 even for nonsense candidates), so we
    report a CONTRASTIVE score: how much closer the candidate is to the GT than
    a control token is. 1.0 = identical to GT, 0.0 = no better than the control.
    """

    def __init__(self, control="zzzzzz", window=8, echo_penalty=0.5):
        self._model = None
        self.control = control
        self.window = window
        self.echo_penalty = echo_penalty
        self.last_echo = set()
        self.last_control_sim = None
        self._warned_unreliable = False
        self._warned_noisy = False

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                sys.exit(
                    "--semantic requires 'sentence-transformers'.\n"
                    "Install with: python -m pip install sentence-transformers\n"
                    "(Windows CPU-only: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cpu first)"
                )
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def batch_slot_scores(self, before, candidates, after, gt):
        """Return {candidate_text: (contrastive_score, raw_cosine)}.

        Uses a LOCAL window around the slot (last `self.window` words of
        `before` plus first `self.window` words of `after`) for the embeddings.
        Full-context slots saturate the cosine toward 1.0 for every candidate,
        which would also push control_sim -> 1.0 and disable the metric; the
        windowed comparison keeps the anchor meaningful. The char-range
        verification in run() still uses the full context; only the semantic
        comparison is windowed.

        Anchors each candidate against the control:
        score = (raw - control_sim) / (1 - control_sim), so 1.0 means identical
        to the GT, 0.0 means no better than the control, and negatives mean
        worse than the control.
        """
        self._ensure()
        self.last_control_sim = None  # reset so empty-candidates boxes don't leak a stale anchor
        self.last_echo = set()
        if not candidates:
            return {}
        b_local = " ".join(before.split()[-self.window:]) if before else ""
        a_local = " ".join(after.split()[:self.window]) if after else ""
        # Adjacency-echo guard: the contrastive anchor cannot punish a candidate
        # that simply reproduces a word sitting right next to the gap (an echoed
        # word is always closer to the GT slot than a nonsense control). So we
        # knock `echo_penalty` off the score of any candidate that is mostly
        # composed of adjacent-window words -- the "Indian"/"Chinese" failure
        # mode. A long phrase that merely shares a few function words is left
        # alone (shared/total ratio must be >= 0.5). Only content-length tokens
        # (>=4 chars) count as echo: function words ("the", "of", "and") are
        # ubiquitous in any window and would wrongly flag legit multi-word
        # answers like "in the region".
        adj = {s for t in tokenize(f"{b_local} {a_local}") if len(s := _stem(t)) >= 4}
        gt_slot = f"{b_local} {gt} {a_local}".strip()
        ctrl_slot = f"{b_local} {self.control} {a_local}".strip()
        gt_emb, ctrl_emb = self._model.encode([gt_slot, ctrl_slot], normalize_embeddings=True)
        control_sim = float(ctrl_emb @ gt_emb)
        self.last_control_sim = control_sim
        if control_sim > 0.98:
            # Control indistinguishable from GT in embedding space: the contrastive
            # anchor is unreliable, so skip scoring for this redaction entirely
            # (and avoid wasting encodes on the candidate slots).
            if not self._warned_unreliable:
                print(
                    f"  [semantic] warning: control sim {control_sim:.2f} is too high "
                    "to anchor a contrastive score; scoring skipped for this redaction. "
                    "Pass --semantic-control with a clearly unrelated token.",
                    file=sys.stderr,
                )
                self._warned_unreliable = True
            return {}
        if control_sim > 0.95 and not self._warned_noisy:
            print(
                f"  [semantic] note: control sim {control_sim:.2f} is high; scores "
                "may be noisy. Consider --semantic-control with an unrelated token.",
                file=sys.stderr,
            )
            self._warned_noisy = True
        cand_embs = self._model.encode(
            [f"{b_local} {c} {a_local}".strip() for c in candidates],
            normalize_embeddings=True)
        denom = max(1.0 - control_sim, 1e-4)
        out = {}
        for c, e in zip(candidates, cand_embs):
            raw = float(e @ gt_emb)
            score = (raw - control_sim) / denom
            if self.echo_penalty > 0:
                c_toks = {_stem(t) for t in tokenize(c)}
                matched = c_toks & adj
                if matched and len(matched) * 2 >= max(len(c_toks), 1):
                    score -= self.echo_penalty
                    self.last_echo.add(c)
            out[c] = (score, raw)
        return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class MercuryBackend:
    """Inception Labs Mercury -- a diffusion LLM with a free API tier.

    FIM mode (default): true diffusion-native infilling via prompt+suffix.
    Chat mode:          explicit length instruction, better length adherence.
    """

    name = "mercury"

    def __init__(self, mode="fim", full_pool=False, no_length_hint=False,
                 fim_only=False):
        self.mode = mode
        self.full_pool = full_pool
        self.no_length_hint = no_length_hint
        self.fim_only = fim_only
        # Per-call diagnostics are deliberately kept separate from candidates:
        # a response that hit max_tokens must never enter the evaluation pool,
        # but we still need to quantify how often that happened.
        self.last_generation = {}
        self._generation_stats = None

    def _begin_generation(self, red):
        self._generation_stats = {
            "api_attempts": 0,
            "http_responses": 0,
            "truncated": 0,
            "fim_truncated": 0,
            "chat_truncated": 0,
            "empty_responses": 0,
            "artifact_filtered": 0,
            "usable_candidates": 0,
            "fim_usable": 0,
            "chat_usable": 0,
            "fallback_chat": False,
            "fim_requested_budgets": [],
        }
        return self._generation_stats

    def _finish_generation(self, red, candidates):
        stats = dict(self._generation_stats or {})
        stats["returned_candidates"] = len(candidates)
        stats["in_range_candidates"] = sum(
            red["min_chars"] <= len(c) <= red["max_chars"] for c in candidates)
        self.last_generation = stats
        self._generation_stats = None
        return candidates

    def generate(self, red, n):
        import requests

        stats = self._begin_generation(red)
        load_dotenv()  # also supports running as a library, not just the CLI
        key = os.environ.get("INCEPTION_API_KEY") or os.environ.get("MERCURY_API_KEY")
        if not key:
            sys.exit(
                "No API key found. Set INCEPTION_API_KEY (get one at "
                "https://inceptionlabs.ai -- free tier includes ~100M tokens)."
            )

        if self.mode == "fim":
            cands = self._generate_fim(requests, red, key, n)
            if any(red["min_chars"] <= len(c) <= red["max_chars"] for c in cands):
                return self._finish_generation(red, cands)
            if self.fim_only:
                print("  [mercury] FIM-only: no in-range candidate; "
                      "skipping chat fallback", file=sys.stderr)
                return self._finish_generation(red, cands)
            # FIM produced nothing inside the box's char range (empty output,
            # artifacts, or out-of-range junk): blend in chat candidates so the
            # pool always has length-verified options to rank. This is the
            # generate-and-verify design -- mercury-edit-2's FIM endpoint is
            # unreliable on prose, and the verification half catches it.
            print("  [mercury] FIM candidates empty/out-of-range; "
                  "adding chat candidates", file=sys.stderr)
            stats["fallback_chat"] = True
            # Chat is ~10x more expensive per call than FIM (mercury-2 spends
            # ~140 tokens reasoning per request), so cap the fallback pool
            # unless --full-pool asks for the whole thing.
            cap = n if self.full_pool else max(1, n // 2)
            chat = self._generate_chat(requests, red, key, cap)
            return self._finish_generation(red, cands + [c for c in chat if c not in cands])
        return self._finish_generation(red, self._generate_chat(requests, red, key, n))

    @staticmethod
    def _is_artifact(text, before, after):
        """Drop the junk FIM models love to emit: trivial whitespace,
        punctuation-only output, multi-line garbage, or long verbatim copies
        of the surrounding context (suffix regurgitation)."""
        t = text.strip()
        if not t or len(t) < 2:
            return True
        if "\n" in t:
            return True
        if all(c in ".\u2033;:!?()[]\"' " for c in t):
            return True
        if len(t) >= 12:
            # Only treat long candidates as regurgitation if they echo text
            # immediately around the gap (the FIM failure mode). A legitimate
            # long phrase that repeats elsewhere in the document is preserved.
            gap = " ".join(before.split()[-30:] + after.split()[:30]).lower()
            if t.lower() in gap:
                return True
        return False

    def _generate_fim(self, requests, red, key, n):
        """Diffusion-native fill-in-the-middle with candidate pooling.

        FIM gives no length control and chokes on long context (it returns a
        bare space when given full paragraphs), so we vary the local context
        window and the token budget across requests, then filter artifacts.
        To cap token spend we also stop early: once 3 usable candidates are
        pooled, the ground truth is hit, or after one window sweep if nothing
        produced fits the box's char range. The char-range + semantic
        verification in run() does the real selection; this is the generate
        half of generate-and-verify.
        """
        before, after = red["before"], red["after"]
        b_words, a_words = before.split(), after.split()
        windows = [4, 8, 12]
        lo, hi = red["min_chars"], red["max_chars"]
        # Experimental tight schedule: max_tokens is an output-token budget,
        # while the box bounds are characters. Use a conservative chars/token
        # heuristic and a small cushion rather than allowing the model to run
        # up to ~2x the character count. This is intentionally an ablation:
        # if the model needs to spill into continuation text before stopping,
        # truncation may rise; the diagnostics make that visible.
        budgets = [
            max(8, lo // 4 + 2),
            max(12, hi // 3 + 4),
            max(16, hi // 2 + 6),
        ]
        endpoint = "/v1/fim/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        candidates = []
        calls = 0
        for i in range(n):
            w = windows[i % len(windows)]
            # Decouple window from budget (offsets by block) so short-window /
            # large-budget and long-window / small-budget combos get tried.
            mt = budgets[(i + i // len(windows)) % len(budgets)]
            self._generation_stats["fim_requested_budgets"].append(mt)
            payload = {
                "model": MERCURY_FIM_MODEL,
                "prompt": " ".join(b_words[-w:]),
                "suffix": " ".join(a_words[:w]),
                "max_tokens": mt,
                "temperature": 0.9,
            }
            text = ""
            ok = False  # valid HTTP response: empty text is final, not retry-worthy
            for _attempt in range(2):  # retry once on transient failures only
                calls += 1  # count actual API attempts, retries included
                self._generation_stats["api_attempts"] += 1
                try:
                    resp = requests.post(MERCURY_BASE + endpoint, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    stats = self._generation_stats
                    stats["http_responses"] += 1
                    text = data["choices"][0]["text"]
                    finish = data["choices"][0].get("finish_reason")
                    text = (text or "").strip()
                    ok = True
                    if finish == "length":
                        # A length-terminated completion is incomplete by
                        # definition. Never let it contaminate recall/ranking.
                        stats["truncated"] += 1
                        stats["fim_truncated"] += 1
                        text = ""
                        print("  [mercury] candidate truncated by max_tokens budget "
                              "(discarded)", file=sys.stderr)
                    elif not text:
                        stats["empty_responses"] += 1
                except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                    print(f"  [mercury] candidate {i + 1} failed: {e}", file=sys.stderr)
                    text = ""
                if text or ok:
                    break
            if text and text not in candidates:
                if self._is_artifact(text, before, after):
                    self._generation_stats["artifact_filtered"] += 1
                else:
                    candidates.append(text)
                    self._generation_stats["usable_candidates"] += 1
                    self._generation_stats["fim_usable"] += 1
            # Token budget: the first three calls already sweep every window
            # size, so stop once we have a usable pool (or have already hit
            # the ground truth), and bail out of FIM after that first sweep
            # if nothing produced fits the box's char range -- the same
            # criterion the chat fallback uses -- rather than burning all n
            # calls on a box where FIM is dead or out of range.
            enough = not self.full_pool and len(candidates) >= 3
            if enough or (
                not self.full_pool
                and i == len(windows) - 1
                and not any(lo <= len(c) <= hi for c in candidates)
            ):
                break
        if candidates:
            print(f"  [mercury] FIM pooled {len(candidates)} usable candidate(s) "
                  f"(wanted {n}, {calls} attempt(s)); artifacts filtered", file=sys.stderr)
        return candidates

    def _generate_chat(self, requests, red, key, n):
        """Chat mode: explicit length instruction, better length adherence."""
        before, after = red["before"], red["after"]
        if self.no_length_hint:
            # Plain-LLM baseline: no leaked length in the prompt.
            # Measures how much recovery actually comes from the char-range
            # leak vs. just context + model knowledge.
            hint = (
                "Reply with ONLY the missing text, nothing else. Do not include "
                "surrounding spaces, quotes, or punctuation."
            )
        else:
            hint = (
                f"The missing text is between {red['min_chars']} and "
                f"{red['max_chars']} characters long (including spaces). "
                "Reply with ONLY the missing text, nothing else. Do not include "
                "surrounding spaces, quotes, or punctuation."
            )
        prompt = (
            f"{hint}\n\nDocument text before the redaction:\n\"{before}\"\n\n"
            f"Document text after the redaction:\n\"{after}\"\n\nMissing text:"
        )
        payload = {
            "model": MERCURY_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            # mercury-2 is a reasoning model: it spends ~140 tokens "thinking"
            # before answering (that is ~45% of every call's cost), so give it
            # a generous budget or it returns empty content
            # (finish_reason="length" with content=null), and cap reasoning
            # effort to "low" -- probed to cut reasoning tokens ~139 -> 63.
            "max_tokens": 1024,
            "reasoning": {"effort": "low"},
        }
        endpoint = "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        candidates = []
        for i in range(n):
            text = ""
            ok = False  # valid HTTP response: empty content is final
            for _attempt in range(2):  # retry once on transient failures only
                self._generation_stats["api_attempts"] += 1
                try:
                    resp = requests.post(MERCURY_BASE + endpoint, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    stats = self._generation_stats
                    stats["http_responses"] += 1
                    text = data["choices"][0]["message"]["content"]
                    finish = data["choices"][0].get("finish_reason")
                    text = (text or "").strip()
                    ok = True
                    if finish == "length":
                        # Chat reasoning may consume the budget before a final
                        # answer, so discard this response just like FIM.
                        stats["truncated"] += 1
                        stats["chat_truncated"] += 1
                        text = ""
                        print("  [mercury] response truncated by max_tokens budget "
                              "(discarded)", file=sys.stderr)
                    elif not text:
                        stats["empty_responses"] += 1
                except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                    print(f"  [mercury] candidate {i + 1} failed: {e}", file=sys.stderr)
                    text = ""
                if text or ok:
                    break
            if text and text not in candidates:
                candidates.append(text)
                self._generation_stats["usable_candidates"] += 1
                self._generation_stats["chat_usable"] += 1
            # Token budget: chat costs ~10x FIM per call, and a few candidates
            # is usually enough once verified downstream (unless --full-pool).
            if not self.full_pool and len(candidates) >= 3:
                break
        return candidates


class EchoBackend:
    """Dummy backend: fabricates candidates across the char range. No network."""

    name = "echo"

    def generate(self, red, n):
        lo, hi = red["min_chars"], red["max_chars"]
        mid = (lo + hi) // 2
        lengths = [max(1, lo - 2), lo, mid, hi, hi + 2]  # below / in / above range
        return ["W" * lengths[i % len(lengths)] for i in range(n)]


def clean_llada_candidate(candidate):
    """Normalize or reject artifacts emitted by the raw base checkpoint.

    LLaDA-8B-Base is not instruction-tuned and can emit HTML entities,
    replacement characters, or a continuation on a second line. These are
    transport/model artifacts, not useful evidence for the redacted span.
    HTML entities are decoded (``&nbsp;`` becomes a normal space); a Unicode
    replacement character is rejected rather than silently guessed through.
    """
    if not isinstance(candidate, str):
        return None
    text = html.unescape(candidate)
    if "\ufffd" in text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None
    # The generated middle is a prose span, so a second line is continuation
    # garbage (e.g. ``active\\nOutput:``). Keep only the first line.
    text = lines[0].replace("\xa0", " ")
    text = " ".join(text.split())
    if not text or text.lower() in {"output", "output:"}:
        return None
    return text


class LladaBackend:
    """Client for the local LLaDA masked-infill server.

    The evaluator sends only visible context and generation settings. Ground
    truth, document context, IDs, and all grading metadata stay client-side.
    We send one request per candidate so nearby token-span lengths can be
    tried without changing the desktop server. Authoritative character-range
    filtering remains local in rank_candidates().
    """

    name = "llada"

    def __init__(self, base_url="http://127.0.0.1:8000", timeout=300.0,
                 steps=64, temperature=0.8, remasking="low_confidence",
                 gen_length=None, block_length=None):
        self.base_url = base_url.rstrip("/")
        self.endpoint = (self.base_url if self.base_url.endswith("/generate")
                         else self.base_url + "/generate")
        self.timeout = timeout
        self.steps = steps
        self.temperature = temperature
        self.remasking = remasking
        self.gen_length = gen_length
        self.block_length = block_length
        self.last_generation = {}

    @staticmethod
    def _estimated_gen_length(red):
        """Estimate LLaDA tokens from the leaked character upper bound.

        The server fills a fixed token span and deliberately does not stop at
        a character boundary. A tighter estimate is therefore essential: the
        previous server default used roughly max_chars/3 + 4, which made an
        8-character box receive about 7 tokens and caused long continuations.
        This leaves a small one-token cushion without pretending tokens equal
        characters; the local verifier remains authoritative.
        """
        return max(2, min(64, (red["max_chars"] + 3) // 4 + 1))

    def _requested_lengths(self, red, n):
        if self.gen_length is not None:
            return [self.gen_length] * n
        base = self._estimated_gen_length(red)
        # Try tight, nominal, and one-token-wider spans. This is still entirely
        # client-side and gives short boxes a chance to land exactly on one
        # word without forcing every candidate to use the widest span.
        schedule = [max(2, base - 1), base, base + 1, base]
        return [schedule[i % len(schedule)] for i in range(n)]

    def generate(self, red, n):
        import requests

        if n < 1:
            raise ValueError("LLaDA candidate count must be at least 1")

        candidates = []
        raw_count = 0
        empty_count = 0
        artifact_count = 0
        replacement_char_count = 0
        html_entity_count = 0
        continuation_count = 0
        http_responses = 0
        server_elapsed = []
        server_lengths = []
        requested_lengths = self._requested_lengths(red, n)

        for gen_length in requested_lengths:
            # Construct this explicitly: never send ground_truth, doc_context,
            # paper type, IDs, or any other evaluator-only field to the model.
            payload = {
                "before": red["before"],
                "after": red["after"],
                "min_chars": red["min_chars"],
                "max_chars": red["max_chars"],
                "candidates": 1,
                "steps": self.steps,
                "temperature": self.temperature,
                "remasking": self.remasking,
                "gen_length": gen_length,
                "block_length": self.block_length or gen_length,
            }
            try:
                response = requests.post(self.endpoint, json=payload,
                                         timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"LLaDA request failed at {self.endpoint}: {exc}. "
                    "Is the desktop server running and reachable?"
                ) from exc
            except ValueError as exc:
                raise RuntimeError(
                    f"LLaDA returned invalid JSON from {self.endpoint}: {exc}"
                ) from exc

            raw_candidates = data.get("candidates")
            if not isinstance(raw_candidates, list):
                raise RuntimeError(
                    "LLaDA response did not contain a 'candidates' list: "
                    f"{data!r}"
                )
            http_responses += 1
            raw_count += len(raw_candidates)
            server_elapsed.append(data.get("elapsed_ms"))
            server_lengths.append(gen_length)
            for raw in raw_candidates:
                if isinstance(raw, str):
                    if "\ufffd" in raw:
                        replacement_char_count += 1
                    if "&" in raw and ";" in raw:
                        html_entity_count += 1
                    if "\n" in raw or "\r" in raw:
                        continuation_count += 1
                cleaned = clean_llada_candidate(raw)
                if cleaned is None:
                    if isinstance(raw, str) and raw.strip():
                        artifact_count += 1
                    else:
                        empty_count += 1
                    continue
                if cleaned not in candidates:
                    candidates.append(cleaned)
                else:
                    artifact_count += 1

        # The server performs one masked-diffusion attempt per request. Keep
        # diagnostics compatible with the existing generation logger.
        valid_elapsed = [x for x in server_elapsed if isinstance(x, (int, float))]
        self.last_generation = {
            "api_attempts": len(requested_lengths),
            "http_responses": http_responses,
            "truncated": 0,
            "fim_truncated": 0,
            "chat_truncated": 0,
            "empty_responses": empty_count,
            "artifact_filtered": artifact_count,
            "replacement_char_artifacts": replacement_char_count,
            "html_entity_candidates": html_entity_count,
            "continuation_candidates": continuation_count,
            "usable_candidates": len(candidates),
            "fim_usable": 0,
            "chat_usable": 0,
            "fallback_chat": False,
            "returned_candidates": len(candidates),
            "raw_candidates": raw_count,
            "in_range_candidates": sum(
                red["min_chars"] <= len(c) <= red["max_chars"]
                for c in candidates
            ),
            "server_model": data.get("model"),
            "server_device": data.get("device"),
            "server_elapsed_ms_total": sum(valid_elapsed) if valid_elapsed else None,
            "server_gen_lengths": server_lengths,
        }
        return candidates


class StubBackend:
    """Placeholder for future backends (OpenAI, Anthropic, ...)."""

    def __init__(self, real_name):
        self.real_name = real_name

    def generate(self, red, n):
        raise NotImplementedError(
            f"Backend '{self.real_name}' is a stub. Implement it by subclassing "
            "the backend interface (generate(red, n) -> list[str]) and wiring it "
            "into BACKENDS below."
        )


BACKENDS = {
    "mercury": MercuryBackend,
    "echo": EchoBackend,
    "llada": LladaBackend,
    "openai": lambda: StubBackend("openai"),
    "anthropic": lambda: StubBackend("anthropic"),
}


# ---------------------------------------------------------------------------
# Scoring + grading helpers (shared by run() and the tightness sweep)
# ---------------------------------------------------------------------------


def score_candidates(red, candidates, scorer, prior_weight, priors):
    """Compute semantic + keyness-blended scores for one redaction's pool.

    Returns (semantic, blended, prior, echoed):
      semantic  {text: (contrastive, raw)} -- empty when no gt / no scorer
      blended   {text: contrastive + prior_weight * prior(text)} -- empty
                unless the prior is active; echoed candidates get zero prior
                credit (the prior's thematic terms overlap with adjacency, so
                re-crediting them would undo the echo penalty)
      prior     the KeynessPrior for this redaction's doc_context (or None)
      echoed    texts the adjacency-echo guard penalized on this call
    """
    semantic = {}
    blended = {}
    echoed = set()
    gt = red.get("ground_truth")
    if gt and scorer is not None:
        semantic = scorer.batch_slot_scores(
            red["before"], candidates, red["after"], gt)
        echoed = set(scorer.last_echo)
    prior = None
    if prior_weight > 0 and priors:
        prior = priors.get(red.get("doc_context", ""))
        if prior is not None and semantic:
            blended = {c: sc + prior_weight * (0.0 if c in echoed else prior.score(c))
                       for c, (sc, _raw) in semantic.items()}
    return semantic, blended, prior, echoed


def grade_one(red, ranked, semantic, blended, semantic_threshold):
    """Pick the best candidate for a redaction and grade it against the GT.

    Returns (result, verdict, char_sim) where result is the (rid, gt, best,
    exact, score, classification) tuple used by the summaries, verdict is the
    human-readable "[result]" line, and char_sim is the best candidate's
    character similarity to the GT (also recorded in the JSON sidecar). The
    scorer (echo-penalized contrastive, plus the keyness blend when the prior
    is active) selects the best in-range candidate; char-range closeness is
    only a fallback when no in-range candidate was scored (e.g. the control
    anchor was unreliable).
    """
    gt = red.get("ground_truth")
    if not gt:
        return None, "", 0.0
    in_range = [info["text"] for _fits, info in ranked if info["fits_chars"]]
    if semantic and in_range and any(t in semantic for t in in_range):
        ranked_best = max(
            in_range,
            key=lambda t: blended.get(t, semantic.get(t, (-1e9, 0))[0]))
    elif in_range:
        ranked_best = ranked[0][1]["text"] if ranked else ""
    else:
        ranked_best = ranked[0][1]["text"] if ranked else ""
    exact = any(info["text"].strip().lower() == gt.lower() for _f, info in ranked)
    match = next((info["text"] for _f, info in ranked
                  if info["text"].strip().lower() == gt.lower()), ranked_best)
    char_sim = difflib.SequenceMatcher(None, match.strip().lower(), gt.lower()).ratio()
    sem = semantic.get(match) if semantic else None
    use_score = blended.get(match) if blended else (sem[0] if sem else None)
    if use_score is None:
        classification = "EXACT" if exact else "?"
        raw_out = None
    else:
        raw_out = sem[1] if sem else None
        classification = "EXACT" if exact else (
            "near-miss" if use_score >= semantic_threshold else "off-target")
    result = (red.get("id", "?"), gt, match, exact, use_score, classification)
    if exact:
        verdict = "EXACT MATCH"
    elif use_score is not None:
        verdict = f"{classification} (sc {use_score:.2f}" + (
            f", raw {raw_out:.2f})" if raw_out is not None else ")")
    else:
        verdict = "no exact match"
    verdict += f" | best candidate: \"{match}\" | char-sim to GT: {char_sim:.0%}"
    return result, verdict, char_sim


def blind_best(ranked):
    """Select a candidate without ground truth or oracle semantic scores.

    The deployed baseline is deliberately simple: choose the in-range
    candidate closest to the leaked character-range midpoint (the same
    non-oracle ordering used by rank_candidates). This gives us a clean
    baseline before adding a genuinely blind reranker such as consensus or
    local likelihood.
    """
    return next((info["text"] for _fits, info in ranked if info["fits_chars"]),
                ranked[0][1]["text"] if ranked else "")


def blind_grade(red, ranked):
    """Return blind top-1 result plus candidate recall@K for one redaction."""
    gt = red.get("ground_truth")
    best = blind_best(ranked)
    exact = bool(gt) and best.strip().lower() == gt.strip().lower()
    recall = bool(gt) and any(
        info["text"].strip().lower() == gt.strip().lower()
        for _fits, info in ranked
    )
    return {
        "id": red.get("id", "?"),
        "doc_id": red.get("doc_id"),
        "paper_type": red.get("paper_type"),
        "paper_type_name": red.get("paper_type_name"),
        "ground_truth": gt,
        "best": best,
        "exact": exact,
        "candidate_recall": recall,
        "candidate_count": len(ranked),
        "in_range_count": sum(1 for _fits, info in ranked if info["fits_chars"]),
        "char_sim": (difflib.SequenceMatcher(
            None, best.strip().lower(), gt.strip().lower()).ratio()
            if gt else 0.0),
    }


def _write_json(path, payload):
    """Write the structured sidecar for a run (used by --json-results)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _generation_totals(stats_list):
    """Aggregate backend generation-health counters for a sidecar."""
    return {
        "api_attempts": sum(s.get("api_attempts", 0) for s in stats_list),
        "http_responses": sum(s.get("http_responses", 0) for s in stats_list),
        "truncated": sum(s.get("truncated", 0) for s in stats_list),
        "fim_truncated": sum(s.get("fim_truncated", 0) for s in stats_list),
        "chat_truncated": sum(s.get("chat_truncated", 0) for s in stats_list),
        "empty_responses": sum(s.get("empty_responses", 0) for s in stats_list),
        "artifact_filtered": sum(s.get("artifact_filtered", 0) for s in stats_list),
        "replacement_char_artifacts": sum(
            s.get("replacement_char_artifacts", 0) for s in stats_list),
        "html_entity_candidates": sum(
            s.get("html_entity_candidates", 0) for s in stats_list),
        "continuation_candidates": sum(
            s.get("continuation_candidates", 0) for s in stats_list),
        "usable_candidates": sum(s.get("usable_candidates", 0) for s in stats_list),
        "fim_usable": sum(s.get("fim_usable", 0) for s in stats_list),
        "chat_usable": sum(s.get("chat_usable", 0) for s in stats_list),
        "fallback_chat": sum(bool(s.get("fallback_chat")) for s in stats_list),
        "fim_requested_budget_min": min(
            (b for s in stats_list for b in s.get("fim_requested_budgets", [])),
            default=None),
        "fim_requested_budget_max": max(
            (b for s in stats_list for b in s.get("fim_requested_budgets", [])),
            default=None),
        "fim_requested_budget_mean": (lambda bs: sum(bs) / len(bs) if bs else None)(
            [b for s in stats_list for b in s.get("fim_requested_budgets", [])]),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(redactions, backend, candidates_per_box, scorer=None, semantic_threshold=0.7,
        prior_weight=0.0, priors=None, out=sys.stdout, json_path=None, meta=None):
    results = []  # legacy oracle results: (rid, gt, best, exact, sem, classification)
    blind_results = []
    details = []  # per-redaction dicts for the JSON sidecar (--json-results)
    generation_stats = []
    printed_priors = set()

    for red in redactions:
        print("=" * 70, file=out)
        print(f"redaction: {red.get('id', '?')}  "
              f"chars [{red['min_chars']}, {red['max_chars']}]", file=out)
        if red.get("note"):
            print(f"  note: {red['note']}", file=out)
        gt = red.get("ground_truth")
        if gt:
            print(f"  ground truth: \"{gt}\"", file=out)
        try:
            cands = backend.generate(red, candidates_per_box)
        except NotImplementedError as e:
            print(f"  !! {e}", file=out)
            continue
        ranked = rank_candidates(cands, red)
        gen_stats = dict(getattr(backend, "last_generation", {}) or {})
        generation_stats.append(gen_stats)
        blind = blind_grade(red, ranked)
        blind_results.append(blind)
        print(f"  [blind] best candidate: \"{blind['best']}\" "
              f"({'EXACT' if blind['exact'] else 'not exact'}); "
              f"candidate_recall@K={'hit' if blind['candidate_recall'] else 'miss'}",
              file=out)

        semantic, blended, prior, echoed = score_candidates(
            red, [info["text"] for _fits, info in ranked], scorer, prior_weight, priors)
        ctrl = scorer.last_control_sim if scorer is not None else None
        if scorer is not None and scorer.last_control_sim is not None:
            print(f"  [semantic] control_sim={scorer.last_control_sim:.2f} "
                  f"(anchor for contrastive scores)", file=out)
        if scorer is not None and echoed:
            print(f"  [semantic] echo-penalized {len(echoed)} candidate(s) "
                  f"(reproduce adjacent text; -{scorer.echo_penalty:.2f}): "
                  f"{', '.join(sorted(echoed))}", file=out)
        if prior is not None and red.get("doc_context") not in printed_priors:
            print(f"  [keyness] {prior.describe()}", file=out)
            printed_priors.add(red.get("doc_context"))

        for fits, info in ranked:
            mark = "FIT " if fits else "    "
            flag = "" if info["fits_chars"] else "  (char range!)"
            if gt and info["text"].strip().lower() == gt.lower():
                flag += "  == GT"
            if info["text"] in semantic:
                score, raw = semantic[info["text"]]
                flag += f"  cos={raw:.2f} sc={score:.2f}"
            if info["text"] in blended:
                flag += f" fin={blended[info['text']]:.2f}"
            if gt and scorer is not None and info["text"] in scorer.last_echo:
                flag += "  (echo)"
            print(f"  [{mark}] chars={info['chars']}{flag}  |{info['text']}|", file=out)

        if gt:
            result, verdict, char_sim = grade_one(red, ranked, semantic, blended,
                                                  semantic_threshold)
            results.append(result)
            print(f"  [result] {verdict}", file=out)
            sem = semantic.get(result[2]) if semantic else None
            details.append({
                **blind,
                "oracle_best": result[2],
                "oracle_exact": result[3],
                "oracle_score": result[4],
                "oracle_classification": result[5],
                "oracle_raw": sem[1] if sem else None,
                "control_sim": ctrl,
                "generation": gen_stats,
            })

    if results:
        # Summary stats used by both the printed GRADED RECOVERY line and the
        # JSON sidecar (computed once). With the prior active the scores are the
        # blended finals (contrastive + prior), not raw contrastive.
        semis = [r[4] for r in results if r[4] is not None]
        exacts = sum(1 for r in results if r[3])
        near = sum(1 for r in results if r[5] == "near-miss")
        off = sum(1 for r in results if r[5] == "off-target")
        metric = ("mean best-candidate final score"
                  if prior_weight > 0 and priors else
                  "mean best-candidate contrastive score")
        print("=" * 70, file=out)
        print("ORACLE DIAGNOSTIC (uses ground truth; not blind performance)", file=out)
        for rid, gt, best, exact, sem, cls in results:
            extra = f"  sc={sem:.2f}" if sem is not None else ""
            print(f"  {rid}: GT \"{gt}\" -> best \"{best}\" ({cls}){extra}", file=out)
        if semis:
            print("=" * 70, file=out)
            print(f"ORACLE DIAGNOSTIC: exact={exacts}  near-miss={near}  off-target={off}  "
                  f"{metric}={sum(semis) / len(semis):.2f}", file=out)
        blind_exact = sum(1 for r in blind_results if r["exact"])
        blind_recall = sum(1 for r in blind_results if r["candidate_recall"])
        generation_totals = _generation_totals(generation_stats)
        by_type = {}
        for row in blind_results:
            key = str(row["paper_type"]) if row["paper_type"] is not None else "untyped"
            bucket = by_type.setdefault(key, {"n": 0, "blind_exact": 0,
                                              "candidate_recall": 0})
            bucket["n"] += 1
            bucket["blind_exact"] += int(row["exact"])
            bucket["candidate_recall"] += int(row["candidate_recall"])
        print("=" * 70, file=out)
        print(f"BLIND BASELINE: exact={blind_exact}/{len(blind_results)}  "
              f"candidate_recall@K={blind_recall}/{len(blind_results)}  "
              "(selection uses char range + generator order only)", file=out)
        for key, bucket in sorted(by_type.items()):
            print(f"  paper_type={key}: blind_exact={bucket['blind_exact']}/{bucket['n']}  "
                  f"candidate_recall@K={bucket['candidate_recall']}/{bucket['n']}", file=out)
        print("GENERATION DIAGNOSTICS (truncated responses are discarded)", file=out)
        print("  " + "  ".join(f"{k}={v}" for k, v in generation_totals.items()), file=out)
        if json_path:
            _write_json(json_path, {
                "schema": "unredact-run/v3",
                "meta": meta or {},
                "summary": {
                    "n": len(results),
                    "blind_exact": blind_exact,
                    "candidate_recall_at_k": blind_recall,
                    "blind_exact_rate": (blind_exact / len(blind_results)) if blind_results else None,
                    "candidate_recall_rate": (blind_recall / len(blind_results)) if blind_results else None,
                    "oracle_exact": exacts,
                    "oracle_near_miss": near,
                    "oracle_off_target": off,
                    "oracle_mean_score": (sum(semis) / len(semis)) if semis else None,
                    "oracle_metric_label": metric,
                    "by_paper_type": by_type,
                    "generation": generation_totals,
                },
                "redactions": details,
            })


def run_tightness_sweep(redactions, backend, candidates_per_box, deltas,
                        scorer=None, semantic_threshold=0.7, prior_weight=0.0,
                        priors=None, out=sys.stdout, json_path=None, meta=None):
    """Constraint-tightness ablation: recovery vs how loose the box's leaked
    char range (a selection-stage diagnostic for the length leak).

    Each delta widens every box's [min_chars, max_chars] by `delta` characters
    on both sides, and each redaction is re-verified, re-ranked and re-graded
    at that width. Candidate pools are generated ONCE per redaction (at the
    tight bounds) and re-verified at every delta, so the sweep isolates what
    the leaked length constraint contributes at the *selection* stage -- it is
    deterministic and costs zero extra API calls. (The generation-side effect
    of looser length hints is measured separately with --loosen <delta>.)
    """
    print("=" * 70, file=out)
    print("TIGHTNESS SWEEP (ORACLE DIAGNOSTIC): recovery vs constraint looseness "
          f"(deltas = {', '.join(str(d) for d in deltas)} chars each side)", file=out)
    print("One candidate pool per redaction (generated at the tight bounds) is "
          "re-verified at every delta -- isolates the selection-stage effect "
          "of the length leak; zero extra API calls.", file=out)
    if scorer is None:
        print("note: without --semantic, grading is exact-match only "
              "(no near/off classification)", file=out)

    # Generate + score each redaction ONCE; only the verification changes.
    prepared = []  # (red, cands, semantic, blended, generation stats)
    generation_stats = []
    printed_priors = set()
    for red in redactions:
        try:
            cands = backend.generate(red, candidates_per_box)
        except NotImplementedError as e:
            print(f"  !! {e}", file=out)
            continue
        gen_stats = dict(getattr(backend, "last_generation", {}) or {})
        generation_stats.append(gen_stats)
        semantic, blended, prior, _echoed = score_candidates(
            red, cands, scorer, prior_weight, priors)
        if prior is not None and red.get("doc_context") not in printed_priors:
            print(f"  [keyness] {prior.describe()}", file=out)
            printed_priors.add(red.get("doc_context"))
        prepared.append((red, cands, semantic, blended, gen_stats))

    all_by_delta = {}   # delta -> [result, ...]
    rows_by_delta = {}  # delta -> [sidecar row dicts]
    for d in deltas:
        results = []
        rows = []
        for red, cands, semantic, blended, gen_stats in prepared:
            # Widen the box by `d` chars on both sides and re-verify/re-rank
            # the SAME pool through the standard rank_candidates -- keeps the
            # sweep's selection identical to a plain run() with those bounds.
            wide = dict(red)
            wide["min_chars"] = max(1, red["min_chars"] - d)
            wide["max_chars"] = red["max_chars"] + d
            ranked = rank_candidates(cands, wide)
            if wide.get("ground_truth"):
                result, _verdict, _char_sim = grade_one(
                    wide, ranked, semantic, blended, semantic_threshold)
                results.append(result)
                rows.append({
                    "id": result[0], "ground_truth": result[1],
                    "best": result[2], "exact": result[3],
                    "score": result[4],
                    "classification": result[5],
                    "generation": gen_stats,
                })

        all_by_delta[d] = results
        rows_by_delta[d] = rows
        print("-" * 70, file=out)
        print(f"delta={d}: bounds widened by {d} chars each side "
              f"([min-{d}, max+{d}])", file=out)
        for result in results:
            rid, gt, best, exact, sem, cls = result
            extra = f"  sc={sem:.2f}" if sem is not None else ""
            mark = "EXACT" if exact else (cls or "?")
            print(f"  {rid:20s} {mark:12s} |{best}|{extra}", file=out)
        exacts = sum(1 for r in results if r[3])
        near = sum(1 for r in results if r[5] == "near-miss")
        off = sum(1 for r in results if r[5] == "off-target")
        semis = [r[4] for r in results if r[4] is not None]
        mean = sum(semis) / len(semis) if semis else float("nan")
        print(f"  ORACLE GRADED: exact={exacts}  near-miss={near}  off-target={off}  "
              f"mean={'%.2f' % mean if semis else 'n/a'}", file=out)

    print("=" * 70, file=out)
    print("COMPARISON: recovery vs looseness (chars added to each side)", file=out)
    print(f"{'DELTA':>6} {'EXACT':>6} {'NEAR':>6} {'OFF':>6} {'MEAN_FINAL':>11}", file=out)
    for d in deltas:
        results = all_by_delta[d]
        exacts = sum(1 for r in results if r[3])
        near = sum(1 for r in results if r[5] == "near-miss")
        off = sum(1 for r in results if r[5] == "off-target")
        semis = [r[4] for r in results if r[4] is not None]
        mean = sum(semis) / len(semis) if semis else float("nan")
        print(f"{d:>6} {exacts:>6} {near:>6} {off:>6} "
              f"{('%.2f' % mean) if semis else 'n/a':>11}", file=out)
    print("Note: this pool-reuse sweep isolates the SELECTION-stage effect of the "
          "length leak. Empirically the curve is flat (the semantic scorer already "
          "ranks the right fill first, so widening the eligible set barely moves "
          "recovery) -- the leaked length mainly conditions GENERATION. Compare "
          "with the chat baseline runs (with vs without --no-length-hint).", file=out)
    if json_path:
        _write_json(json_path, {
            "schema": "unredact-tightness-sweep/v3-oracle",
            "meta": meta or {},
            "deltas": [
                {
                    "delta": d,
                    "n": len(all_by_delta[d]),
                    "exact": sum(1 for r in all_by_delta[d] if r[3]),
                    "near_miss": sum(1 for r in all_by_delta[d] if r[5] == "near-miss"),
                    "off_target": sum(1 for r in all_by_delta[d] if r[5] == "off-target"),
                    "mean_score": (lambda s: (sum(s) / len(s)) if s else None)(
                        [r[4] for r in all_by_delta[d] if r[4] is not None]),
                }
                for d in deltas
            ],
            "generation": _generation_totals(generation_stats),
            "per_delta": {str(d): rows_by_delta[d] for d in deltas},
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_unicode_stdio():
    """Keep Windows consoles/pipes from crashing on model Unicode output.

    LLaDA's raw base checkpoint can emit characters outside CP1252 (for
    example U+25BC). Preserve UTF-8 in files/pipes while replacing only what a
    legacy console cannot display.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # Nonstandard file-like streams used by library callers/tests may
            # not support reconfigure; their owner controls encoding.
            pass


def main(argv=None):
    _configure_unicode_stdio()
    ap = argparse.ArgumentParser(description="Length-constrained unredaction MVP (pure text).")
    ap.add_argument("--redactions", default="redactions.json",
                    help="JSON input file (default: redactions.json)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="mercury",
                    help="Infilling backend (default: mercury)")
    ap.add_argument("--mode", choices=["fim", "chat"], default="fim",
                    help="Mercury API mode (default: fim)")
    ap.add_argument("--llada-url", default=os.environ.get("LLADA_URL"),
                    help="Complete LLaDA server URL, including scheme and port "
                         "(default: LLADA_URL, otherwise --llada-host/--llada-port)")
    ap.add_argument("--llada-host",
                    default=os.environ.get("LLADA_HOST", "127.0.0.1"),
                    help="LLaDA server hostname or IP (default: LLADA_HOST or 127.0.0.1)")
    ap.add_argument("--llada-port", type=int,
                    default=os.environ.get("LLADA_PORT") or "8000",
                    help="LLaDA server port (default: LLADA_PORT or 8000)")
    ap.add_argument("--llada-timeout", type=float, default=300.0,
                    help="LLaDA request timeout in seconds (default: 300)")
    ap.add_argument("--llada-steps", type=int, default=64,
                    help="LLaDA masked-sampling steps (default: 64)")
    ap.add_argument("--llada-temperature", type=float, default=0.8,
                    help="LLaDA sampling temperature (default: 0.8; 0 is deterministic)")
    ap.add_argument("--llada-remasking", choices=["low_confidence", "random"],
                    default="low_confidence",
                    help="LLaDA remasking strategy (default: low_confidence)")
    ap.add_argument("--llada-gen-length", type=int, default=None,
                    help="Optional LLaDA generation length in tokens; otherwise "
                         "let the server infer it from max_chars")
    ap.add_argument("--llada-block-length", type=int, default=None,
                    help="Optional LLaDA block length in tokens")
    ap.add_argument("--candidates", type=int, default=6,
                    help="Candidates to generate per redaction (default: 6); "
                         "chat mode caps at 3 unless --full-pool")
    ap.add_argument("--semantic", action="store_true",
                    help="Score candidates with contextual embedding similarity "
                         "(requires sentence-transformers)")
    ap.add_argument("--semantic-threshold", type=float, default=0.7,
                    help="Contrastive-score threshold for 'near-miss' classification (default 0.7)")
    ap.add_argument("--semantic-control", default="zzzzzz",
                    help="Control token used to anchor the contrastive score "
                         "(default: 'zzzzzz')")
    ap.add_argument("--semantic-window", type=int, default=8,
                    help="Words of context kept on each side of the slot for "
                         "embedding (default 8; larger saturates the cosine)")
    ap.add_argument("--full-pool", action="store_true",
                    help="Lift early-stop token caps: gather the full candidate "
                         "pool (FIM and chat) for eval/paper analysis")
    ap.add_argument("--prior-weight", type=float, default=0.0,
                    help="Blend a document-keyness prior into semantic scores: "
                         "final = contrastive + prior_weight * prior(candidate) "
                         "(default 0 = off; requires --semantic and a doc_context "
                         "field in the redactions file)")
    ap.add_argument("--echo-penalty", type=float, default=0.5,
                    help="Points subtracted from the semantic score of any "
                         "candidate that mostly reproduces words adjacent to the "
                         "gap (the echo failure mode; default 0.5; 0 disables). "
                         "Only applies with --semantic.")
    ap.add_argument("--loosen", type=int, default=0,
                    help="Ablation: widen every box's char range by this many "
                         "characters on both sides (min-loosen, max+loosen) "
                         "before running -- looser length leak, for BOTH "
                         "generation (chat length hints) and verification. "
                         "Run 0 / 3 / 5 / 10 and compare recovery.")
    ap.add_argument("--tightness-sweep", default="",
                    help="Ablation: comma-separated deltas (e.g. 0,3,5,10) -- "
                         "re-verify ONE candidate pool per redaction at each "
                         "widening of the box's char range and print a "
                         "recovery-vs-looseness table (zero extra API calls; "
                         "the table needs --semantic for near/off grading).")
    ap.add_argument("--no-length-hint", action="store_true",
                    help="Chat-mode baseline: strip the leaked char-range from "
                    "the prompt (plain LLM baseline). Measures how much "
                    "recovery actually comes from the length leak.")
    ap.add_argument("--fim-only", action="store_true",
                    help="FIM ablation: never use Mercury chat as a fallback "
                    "when FIM produces no in-range candidate. Keeps the FIM "
                    "budget experiment isolated from chat performance.")

    ap.add_argument("--json-results", default="",
                    help="Write a structured JSON sidecar (per-redaction rows + "
                         "summary, or per-delta rows for --tightness-sweep) to "
                         "this path. Used by log_run.py to build graphable logs "
                         "(logs/*.json + logs/runs.csv).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use the no-network 'echo' backend to test the pipeline")
    args = ap.parse_args(argv)
    if args.candidates < 1:
        sys.exit("--candidates must be at least 1")
    if args.llada_steps < 1:
        sys.exit("--llada-steps must be at least 1")
    if args.llada_timeout <= 0:
        sys.exit("--llada-timeout must be greater than 0")
    if not 1 <= args.llada_port <= 65535:
        sys.exit("--llada-port must be between 1 and 65535")
    if not args.llada_url:
        args.llada_url = f"http://{args.llada_host}:{args.llada_port}"
    if args.llada_gen_length is not None and args.llada_gen_length < 1:
        sys.exit("--llada-gen-length must be at least 1")
    if args.llada_block_length is not None and args.llada_block_length < 1:
        sys.exit("--llada-block-length must be at least 1")

    load_dotenv()
    redactions = load_redactions(args.redactions)
    if not redactions:
        sys.exit(f"No redactions found in {args.redactions}")

    if args.loosen > 0:
        for r in redactions:
            r["min_chars"] = max(1, r["min_chars"] - args.loosen)
            r["max_chars"] = r["max_chars"] + args.loosen
        print(f"note: --loosen {args.loosen} -- every box widened by "
              f"{args.loosen} chars on each side (generation hints and "
              f"verification both loosened)", file=sys.stderr)

    backend_name = "echo" if args.dry_run else args.backend
    if backend_name == "llada":
        backend = LladaBackend(
            base_url=args.llada_url,
            timeout=args.llada_timeout,
            steps=args.llada_steps,
            temperature=args.llada_temperature,
            remasking=args.llada_remasking,
            gen_length=args.llada_gen_length,
            block_length=args.llada_block_length,
        )
    else:
        backend = BACKENDS[backend_name]()
    if backend_name == "mercury":
        backend.mode = args.mode
        backend.full_pool = args.full_pool
        backend.no_length_hint = args.no_length_hint
        backend.fim_only = args.fim_only
        if args.fim_only and args.mode != "fim":
            print("warning: --fim-only only affects FIM mode", file=sys.stderr)
        if args.no_length_hint and args.mode != "chat":
            print("warning: --no-length-hint only affects chat-mode prompts; "
                  "FIM mode has no length hint to strip", file=sys.stderr)

    if args.prior_weight > 0 and not args.semantic:
        print("warning: --prior-weight blends into semantic scores; pass --semantic "
              "or the prior has no effect", file=sys.stderr)
    priors = {}
    if args.prior_weight > 0:
        from keyness import KeynessPrior
        for red in redactions:
            ctx = red.get("doc_context", "")
            if ctx and ctx not in priors:
                priors[ctx] = KeynessPrior(ctx)
        if not priors:
            print("warning: --prior-weight given but no redaction has doc_context; "
                  "the prior will have no effect", file=sys.stderr)

    scorer = (SemanticScorer(args.semantic_control, args.semantic_window,
                             args.echo_penalty) if args.semantic else None)

    # Sidecar metadata for --json-results: everything needed to reproduce the
    # run and to chart it against other runs in logs/runs.csv.
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": (sys.argv[1:] if argv is None else list(argv)),
        "redactions_file": args.redactions,
        "backend": backend_name,
        "mode": args.mode if backend_name == "mercury" else None,
        "llada_url": args.llada_url if backend_name == "llada" else None,
        "llada_timeout": args.llada_timeout if backend_name == "llada" else None,
        "llada_steps": args.llada_steps if backend_name == "llada" else None,
        "llada_temperature": args.llada_temperature if backend_name == "llada" else None,
        "llada_remasking": args.llada_remasking if backend_name == "llada" else None,
        "llada_gen_length": args.llada_gen_length if backend_name == "llada" else None,
        "llada_block_length": args.llada_block_length if backend_name == "llada" else None,
        "candidates": args.candidates,
        "full_pool": args.full_pool,
        "semantic": args.semantic,
        "semantic_threshold": args.semantic_threshold,
        "semantic_control": args.semantic_control,
        "semantic_window": args.semantic_window,
        "echo_penalty": args.echo_penalty,
        "prior_weight": args.prior_weight,
        "loosen": args.loosen,
        "tightness_sweep": args.tightness_sweep,
        "no_length_hint": args.no_length_hint,
        "fim_only": args.fim_only,
        "dry_run": args.dry_run,
        "evaluation": "blind_top1_plus_candidate_recall_and_oracle_diagnostic",
    }
    try:
        import subprocess as _sp
        meta["git_commit"] = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        meta["git_commit"] = None

    json_path = args.json_results or None
    if args.tightness_sweep:
        deltas = [int(x) for x in args.tightness_sweep.split(",") if x.strip()]
        if not deltas or any(d < 0 for d in deltas):
            sys.exit("--tightness-sweep expects non-negative ints, e.g. 0,3,5,10")
        run_tightness_sweep(redactions, backend, args.candidates, deltas, scorer,
                            args.semantic_threshold, args.prior_weight, priors,
                            json_path=json_path, meta=meta)
        return
    run(redactions, backend, args.candidates, scorer, args.semantic_threshold,
        args.prior_weight, priors, json_path=json_path, meta=meta)


if __name__ == "__main__":
    main()
