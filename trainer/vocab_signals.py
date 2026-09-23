"""Deterministic vocabulary-knowledge signals from one turn of the learner's own speech.

The only thing that needs judgment is *whether the learner asked about a word this turn, and
which one* — that comes from the tutor's structured output (see tutor.py: `asked_about`).
Everything here is pure, no model or I/O involved: matching her sentence against the target
vocabulary list, and applying the exclusion rule (asking about one word says nothing about
whether she knew the others in that same sentence — she never asks about more than one at once).
"""
import re

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

KNOWN_QUALITY = 4     # SM-2 quality fed to Store.review_vocab for a word used without asking
UNKNOWN_QUALITY = 2    # SM-2 quality for a word the learner explicitly asked about


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def match_target_words(text: str, target_words: set[str]) -> list[str]:
    """Exact, case-insensitive whole-word match against `target_words` only - no stemming or
    lemmatization, so an inflected form ("gegangen") will not match its base form ("gehen").
    Undercounting is preferred over a fuzzy match that could silently misattribute knowledge.
    Returns matches in first-appearance order, deduplicated."""
    by_lower = {w.lower(): w for w in target_words}
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokenize(text):
        canonical = by_lower.get(tok.lower())
        if canonical is not None and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def compute_vocab_signals(text: str, target_words: set[str], asked_about: str | None) -> dict[str, int]:
    """The word -> SM-2 quality mapping for this one turn.

    - If `asked_about` is set: ONLY that word gets a signal (UNKNOWN_QUALITY). Every other target
      word matched in the same sentence is dropped entirely - neither pass nor fail - because the
      learner only ever asks about one word at a time, so their silence about the rest is not
      evidence they knew them.
    - If `asked_about` is None: every matched target word gets KNOWN_QUALITY.

    `asked_about` is recorded regardless of whether it's itself in `target_words` (the learner can
    ask about any word); the caller is responsible for sourcing its gloss separately in that case.
    """
    if asked_about:
        return {asked_about: UNKNOWN_QUALITY}
    return {w: KNOWN_QUALITY for w in match_target_words(text, target_words)}
