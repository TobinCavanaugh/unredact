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
  llada / openai / anthropic   STUBS -- drop-in points for other models later.

Usage:
  python unredact.py --backend mercury --redactions redactions.json
  python unredact.py --dry-run        # test the whole pipeline with no API key
"""

import argparse
import difflib
import json
import os
import sys

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

    def __init__(self, mode="fim", full_pool=False):
        self.mode = mode
        self.full_pool = full_pool

    def generate(self, red, n):
        import requests

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
                return cands
            # FIM produced nothing inside the box's char range (empty output,
            # artifacts, or out-of-range junk): blend in chat candidates so the
            # pool always has length-verified options to rank. This is the
            # generate-and-verify design -- mercury-edit-2's FIM endpoint is
            # unreliable on prose, and the verification half catches it.
            print("  [mercury] FIM candidates empty/out-of-range; "
                  "adding chat candidates", file=sys.stderr)
            # Chat is ~10x more expensive per call than FIM (mercury-2 spends
            # ~140 tokens reasoning per request), so cap the fallback pool
            # unless --full-pool asks for the whole thing.
            cap = n if self.full_pool else max(1, n // 2)
            chat = self._generate_chat(requests, red, key, cap)
            return cands + [c for c in chat if c not in cands]
        return self._generate_chat(requests, red, key, n)

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
        gt = red.get("ground_truth")
        budgets = [
            max(16, lo // 3),
            max(24, hi // 2),
            max(32, hi * 2),
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
                try:
                    resp = requests.post(MERCURY_BASE + endpoint, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["text"]
                    finish = data["choices"][0].get("finish_reason")
                    text = (text or "").strip()
                    ok = True
                    if finish == "length":
                        if text:
                            print("  [mercury] candidate truncated by max_tokens budget "
                                  "(accepted)", file=sys.stderr)
                        else:
                            print("  [mercury] response truncated by max_tokens budget", file=sys.stderr)
                except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                    print(f"  [mercury] candidate {i + 1} failed: {e}", file=sys.stderr)
                    text = ""
                if text or ok:
                    break
            if text and text not in candidates and not self._is_artifact(text, before, after):
                candidates.append(text)
            # Token budget: the first three calls already sweep every window
            # size, so stop once we have a usable pool (or have already hit
            # the ground truth), and bail out of FIM after that first sweep
            # if nothing produced fits the box's char range -- the same
            # criterion the chat fallback uses -- rather than burning all n
            # calls on a box where FIM is dead or out of range.
            exact_hit = (not self.full_pool and bool(gt) and any(
                c.strip().lower() == gt.lower() for c in candidates))
            enough = not self.full_pool and len(candidates) >= 3
            if exact_hit or enough or (
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
                try:
                    resp = requests.post(MERCURY_BASE + endpoint, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    finish = data["choices"][0].get("finish_reason")
                    text = (text or "").strip()
                    ok = True
                    if not text and finish == "length":
                        print("  [mercury] response truncated by max_tokens budget", file=sys.stderr)
                except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                    print(f"  [mercury] candidate {i + 1} failed: {e}", file=sys.stderr)
                    text = ""
                if text or ok:
                    break
            if text and text not in candidates:
                candidates.append(text)
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


class StubBackend:
    """Placeholder for future backends (local LLaDA, OpenAI, Anthropic, ...)."""

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
    "llada": lambda: StubBackend("llada"),
    "openai": lambda: StubBackend("openai"),
    "anthropic": lambda: StubBackend("anthropic"),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(redactions, backend, candidates_per_box, scorer=None, semantic_threshold=0.7,
        prior_weight=0.0, priors=None, out=sys.stdout):
    results = []  # (rid, gt, best, exact, sem, classification)
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

        semantic = {}  # text -> (contrastive_score, raw_cosine)
        if gt and scorer is not None:
            semantic = scorer.batch_slot_scores(
                red["before"], [info["text"] for _, info in ranked], red["after"], gt)
            if scorer.last_control_sim is not None:
                print(f"  [semantic] control_sim={scorer.last_control_sim:.2f} "
                      f"(anchor for contrastive scores)", file=out)
            if gt and scorer is not None and scorer.last_echo:
                print(f"  [semantic] echo-penalized {len(scorer.last_echo)} candidate(s) "
                      f"(reproduce adjacent text; -{scorer.echo_penalty:.2f}): "
                      f"{', '.join(sorted(scorer.last_echo))}", file=out)

        # Keyness prior (two-sided): final = contrastive + prior_weight * prior(candidate)
        prior = None
        if prior_weight > 0 and priors:
            prior = priors.get(red.get("doc_context", ""))
            if prior is not None and red.get("doc_context") not in printed_priors:
                print(f"  [keyness] {prior.describe()}", file=out)
                printed_priors.add(red.get("doc_context"))
        blended = {}  # text -> final score when the prior is active
        if prior is not None and semantic:
            # Echoed candidates get no prior credit: the prior's thematic terms
            # overlap with adjacency (the model copied "Indian"/"Chinese" from
            # the surrounding text), so re-crediting them would undo the echo
            # penalty that batch_slot_scores already applied to the contrastive.
            echoed = getattr(scorer, "last_echo", set())
            blended = {c: sc + prior_weight * (0.0 if c in echoed else prior.score(c))
                       for c, (sc, _raw) in semantic.items()}

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
            # The scorer (echo-penalized contrastive, plus the keyness blend
            # when the prior is active) selects the best in-range candidate;
            # char-range closeness is only a fallback when no in-range
            # candidate was scored (e.g. the control anchor was unreliable).
            in_range = [info["text"] for fits, info in ranked if info["fits_chars"]]
            if semantic and in_range and any(t in semantic for t in in_range):
                ranked_best = max(
                    in_range,
                    key=lambda t: blended.get(t, semantic.get(t, (-1e9, 0))[0]))
            elif in_range:
                ranked_best = ranked[0][1]["text"] if ranked else ""
            else:
                ranked_best = ranked[0][1]["text"] if ranked else ""
            exact = any(info["text"].strip().lower() == gt.lower() for _, info in ranked)
            match = next((info["text"] for _, info in ranked
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
            results.append((red.get("id", "?"), gt, match, exact, use_score, classification))
            if exact:
                verdict = "EXACT MATCH"
            elif use_score is not None:
                verdict = f"{classification} (sc {use_score:.2f}" + (
                    f", raw {raw_out:.2f})" if raw_out is not None else ")")
            else:
                verdict = "no exact match"
            print(f"  [result] {verdict} | best candidate: \"{match}\" "
                  f"| char-sim to GT: {char_sim:.0%}", file=out)

    if results:
        print("=" * 70, file=out)
        print("GROUND TRUTH SUMMARY", file=out)
        for rid, gt, best, exact, sem, cls in results:
            extra = f"  sc={sem:.2f}" if sem is not None else ""
            print(f"  {rid}: GT \"{gt}\" -> best \"{best}\" ({cls}){extra}", file=out)
        if any(sem is not None for _r, _g, _b, _e, sem, _c in results):
            semis = [sem for _r, _g, _b, _e, sem, _c in results if sem is not None]
            exacts = sum(1 for r in results if r[3])
            near = sum(1 for r in results if r[5] == "near-miss")
            off = sum(1 for r in results if r[5] == "off-target")
            print("=" * 70, file=out)
            # With the prior active the summary scores are the blended finals
            # (contrastive + prior), not raw contrastive -- label them honestly.
            metric = ("mean best-candidate final score"
                      if prior_weight > 0 and priors else
                      "mean best-candidate contrastive score")
            print(f"GRADED RECOVERY: exact={exacts}  near-miss={near}  off-target={off}  "
                  f"{metric}={sum(semis) / len(semis):.2f}", file=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Length-constrained unredaction MVP (pure text).")
    ap.add_argument("--redactions", default="redactions.json",
                    help="JSON input file (default: redactions.json)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="mercury",
                    help="Infilling backend (default: mercury)")
    ap.add_argument("--mode", choices=["fim", "chat"], default="fim",
                    help="Mercury API mode (default: fim)")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="Use the no-network 'echo' backend to test the pipeline")
    args = ap.parse_args(argv)

    load_dotenv()
    redactions = load_redactions(args.redactions)
    if not redactions:
        sys.exit(f"No redactions found in {args.redactions}")

    backend_name = "echo" if args.dry_run else args.backend
    backend = BACKENDS[backend_name]()
    if backend_name == "mercury":
        backend.mode = args.mode
        backend.full_pool = args.full_pool

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
    run(redactions, backend, args.candidates, scorer, args.semantic_threshold,
        args.prior_weight, priors)


if __name__ == "__main__":
    main()
