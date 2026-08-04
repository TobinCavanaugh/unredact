"""
Document-keyness prior for candidate ranking.

Computes, for a document, words whose frequency is anomalously high relative
to general English (keyness / TF-IDF style), plus an entity shortlist from
capitalization, and exposes a scoring function used to nudge candidate
ranking in unredact.py:

    final_score = semantic_contrastive + prior_weight * keyness_prior(candidate)

Two-sided by design:

  + positive weight on the document's thematic terms / entity shortlist
  - negative weight on the document's OWN dominant entities (the salience
    trap: the model anchors on "Pakistan" when "Pakistan" saturates the
    context, even when the answer is another country). The dominant penalty
    fully overrides the positive term credit so a salience-trap candidate
    always ranks below genuine candidates.

Dependency-free: the general-English baseline is a small embedded table of
common-word frequencies (log10 per billion tokens, i.e. Zipf scale). Coarse
values are fine -- keyness only needs the *ordering* (common words high,
rare words low). Words absent from the table are treated as rare, which is
exactly what makes document-specific vocabulary like "missile" or "IRBM"
score as key. (The wordfreq package, the usual choice, is unmaintained and
breaks on modern Python, so we avoid the dependency.)
"""

import re
from collections import Counter

# ---------------------------------------------------------------------------
# General-English word frequencies (Zipf scale: log10 occurrences per billion
# tokens). Approximate values -- relative ordering is what matters.
# NOTE: proper nouns (India, China, Pakistan, ...) are deliberately NOT here:
# they must flow into the entity shortlist and keyness, not be treated as
# common English words.
# ---------------------------------------------------------------------------

COMMON_WORDS = {
    # function words / top of the frequency list
    "the": 7.8, "of": 7.6, "and": 7.6, "to": 7.5, "in": 7.4, "a": 7.3,
    "is": 7.2, "that": 7.1, "for": 7.0, "it": 7.0, "as": 6.9, "was": 6.9,
    "with": 6.8, "be": 6.8, "by": 6.7, "on": 6.7, "not": 6.6, "he": 6.5,
    "i": 6.5, "at": 6.5, "this": 6.4, "his": 6.4, "they": 6.4, "from": 6.3,
    "had": 6.3, "or": 6.3, "have": 6.3, "but": 6.3, "which": 6.2, "you": 6.2,
    "were": 6.2, "her": 6.2, "all": 6.2, "she": 6.1, "one": 6.1, "do": 6.1,
    "there": 6.1, "their": 6.0, "if": 6.0, "will": 6.0, "would": 5.9,
    "about": 5.9, "who": 5.9, "so": 5.9, "no": 5.9, "when": 5.8, "out": 5.8,
    "up": 5.8, "then": 5.8, "what": 5.8, "my": 5.8, "them": 5.7, "me": 5.7,
    "can": 5.7, "some": 5.7, "more": 5.7, "also": 5.6, "after": 5.6,
    "two": 5.6, "get": 5.6, "other": 5.6, "only": 5.6, "over": 5.6,
    "new": 5.5, "time": 5.5, "its": 5.5, "than": 5.5, "made": 5.5,
    "such": 5.5, "most": 5.5, "many": 5.4, "where": 5.4, "been": 5.4,
    "first": 5.4, "between": 5.4, "through": 5.4, "should": 5.4,
    "people": 5.3, "great": 5.3, "little": 5.3, "before": 5.3, "years": 5.3,
    "own": 5.3, "good": 5.3, "how": 5.3, "because": 5.3, "back": 5.2,
    "could": 5.2, "way": 5.2, "any": 5.2, "our": 5.2, "even": 5.2, "too": 5.2,
    "may": 5.2, "still": 5.2, "must": 5.2, "however": 5.1, "very": 5.1,
    "just": 5.1, "these": 5.1, "those": 5.1, "well": 5.0, "much": 5.0,
    "an": 6.4, "we": 6.0, "us": 5.6, "am": 5.3, "has": 6.0, "did": 5.8,
    "does": 5.2, "being": 5.4, "each": 5.4, "into": 5.5, "while": 5.0,
    "until": 4.8, "under": 5.0, "same": 5.0, "few": 5.0, "both": 5.1,
    "off": 5.1, "down": 5.2, "again": 4.9, "here": 5.0, "why": 5.0,
    # common verbs / auxiliaries
    "said": 5.0, "know": 4.9, "make": 4.9, "see": 4.9, "go": 4.8,
    "think": 4.8, "come": 4.8, "take": 4.8, "want": 4.8, "use": 4.7,
    "find": 4.7, "give": 4.7, "tell": 4.7, "become": 4.7, "leave": 4.6,
    "seem": 4.6, "feel": 4.6, "put": 4.6, "bring": 4.6, "begin": 4.6,
    "show": 4.6, "hear": 4.6, "keep": 4.5, "hold": 4.5, "turn": 4.5,
    "mean": 4.5, "let": 4.5, "set": 4.5, "ask": 4.5, "help": 4.5,
    "follow": 4.4, "live": 4.4, "move": 4.4, "look": 4.4, "run": 4.4,
    "work": 4.4, "need": 4.4, "try": 4.4, "call": 4.4, "form": 4.3,
    "develop": 4.3, "require": 4.3, "provide": 4.3, "include": 4.3,
    "consider": 4.2, "appear": 4.2, "believe": 4.2, "acquire": 3.9,
    # common nouns / adjectives (generic, not domain-specific)
    "year": 5.4, "state": 5.2, "country": 5.0, "world": 5.0,
    "system": 5.0, "government": 4.9, "program": 4.8, "military": 4.7,
    "national": 4.7, "force": 4.6, "part": 4.6, "development": 4.5,
    "area": 4.5, "population": 4.3, "range": 4.3, "weapon": 4.1,
    "nuclear": 4.0, "capable": 4.1, "potential": 4.4, "large": 4.6,
    "small": 4.5, "active": 4.3, "major": 4.4, "northern": 3.8,
    "southern": 3.8, "western": 3.9, "eastern": 3.9, "industrial": 3.8,
    "strategic": 3.9, "missile": 3.7, "ballistic": 3.2, "tactical": 3.6,
    "deterrent": 3.4, "warhead": 3.3, "threat": 4.3, "coverage": 3.6,
    "percentage": 3.8, "including": 4.7, "against": 5.0, "target": 4.3,
    "capability": 3.8, "assistance": 3.8, "purchase": 4.0, "complete": 4.3,
    "outside": 4.3, "source": 4.5, "possible": 4.6, "probably": 4.7,
    "prefer": 4.0, "shorter": 3.5, "purpose": 4.1, "deliver": 4.0,
    "delivery": 3.9, "deployed": 3.7, "deploy": 3.7, "intend": 4.0,
    "strongly": 4.0, "motivated": 3.8, "motivation": 3.7,
}

DEFAULT_ZIPF = 3.0   # words absent from the table are "rare" in general English
ZIPF_CEIL = 8.0      # upper end of the Zipf scale; drives the keyness gradient
ENTITY_BONUS = 0.02  # extra keyness credit for appearing as a capitalized entity


def tokenize(text):
    """Lowercase word tokens: letters with optional internal apostrophes."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def _stem(tok):
    """Strip a trailing possessive 's so 'Pakistan's' and 'Pakistan' match."""
    return tok[:-2] if tok.endswith("'s") else tok


class KeynessPrior:
    """Two-sided document prior: thematic-term keyness (+), dominant-entity
    salience penalty (-). Built from the full document text."""

    def __init__(self, doc_text, top_k=15, dominant_threshold=2):
        self.keyness = {}      # every word (lowercase) -> normalized keyness 0..1
        self.terms = {}        # top-k subset, for display only
        self.entities = {}     # entity (stemmed, case-preserved) -> count
        self.dominant = set()  # lowercase stemmed entities >= threshold times
        self.top_k = top_k
        self.dominant_threshold = dominant_threshold
        self._build(doc_text)

    def _build(self, doc_text):
        tokens = tokenize(doc_text)
        counts = Counter(tokens)
        n = max(len(tokens), 1)

        # entity shortlist: capitalized tokens that aren't common words, with
        # possessive forms stemmed so "Pakistan's" counts as "Pakistan". The
        # token regex already constrains shape (letters + optional 's), so no
        # isalpha() gate (that would reject possessives). Sentence-initial
        # caps are kept: common words there are already excluded by
        # COMMON_WORDS ("The", "We", "It"), and the occasional false positive
        # ("Massive") never reaches the dominance threshold -- while dropping
        # the filter is what lets the doc's own dominant entity ("Pakistan")
        # actually be detected.
        for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", doc_text or ""):
            tok = m.group()
            if tok[0].isupper() and len(tok) >= 2 and tok.lower() not in COMMON_WORDS:
                stem = _stem(tok)
                self.entities[stem] = self.entities.get(stem, 0) + 1
        entity_lower = {e.lower() for e in self.entities}  # keys are already stemmed
        # lowercase so score()'s lowercase tokens match (case-insensitive)
        self.dominant = {w.lower() for w, c in self.entities.items()
                         if c >= self.dominant_threshold}

        # raw keyness: doc frequency x how rare the word is in general English
        raw = {}
        for w, tf in counts.items():
            zipf = COMMON_WORDS.get(w, DEFAULT_ZIPF)
            raw[w] = (tf / n) * max(0.0, ZIPF_CEIL - zipf)
            if _stem(w) in entity_lower:
                raw[w] += ENTITY_BONUS

        mx = max(raw.values()) if raw else 1.0
        self.keyness = {w: v / mx for w, v in raw.items()}
        top = sorted(self.keyness.items(), key=lambda kv: kv[1], reverse=True)
        self.terms = dict(top[:self.top_k])

    def score(self, candidate):
        """Prior contribution in roughly [-1, 1]: mean thematic keyness of the
        candidate's words. If the candidate reproduces a dominant entity, the
        penalty fully overrides the positive credit (-1.0) so the salience-trap
        answer always ranks below genuine candidates."""
        words = tokenize(candidate)
        if not words:
            return 0.0
        if any(_stem(w) in self.dominant for w in words):
            return -1.0
        return sum(self.keyness.get(w, 0.0) for w in words) / len(words)

    def describe(self):
        ents = ", ".join(sorted(self.entities)) or "(none)"
        dom = ", ".join(w.capitalize() for w in sorted(self.dominant)) or "(none)"
        terms = ", ".join(f"{w}({v:.2f})" for w, v in self.terms.items())
        return (f"top terms: {terms} | entities: {ents} | dominant: {dom}")
