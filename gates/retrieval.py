"""Lexical retrieval over the exemplar bank. Stdlib only, no embeddings.

Why not embeddings: `gates` ships with zero runtime dependencies so it drops
into whatever environment the host already has (`PLAN.md` §6), and the two ways
to get embeddings both break something. A vector library is a dependency; an
embedding endpoint is a network call per line, which spends more than the
retrieval saves and makes the gate's cost depend on a second service.

Why lexical is not a consolation prize here: log lines are short, and the
distinction the scanner draws is largely one of surface vocabulary. "falling
back", "skipped", "partial", "could not" against "deprecated", "registered",
"100%|". BM25 over the exemplar bank picks those up directly, deterministically,
and in microseconds.

Retrieval is **balanced by construction** — k signals and k noise exemplars,
never whichever happens to score highest. Returning the nearest neighbours
unbalanced would bias the prompt toward whichever class the log happens to
resemble, and since precision is the floor, the nearest *negative* is usually
the more valuable of the two.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .exemplars import EXEMPLARS, NOISE, SIGNALS, Exemplar

_TOKEN = re.compile(r"[a-z0-9_]+")

# Okapi BM25's usual constants. Not tuned: the bank is ~20 short documents, and
# tuning them against it would be fitting noise.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Index:
    """A BM25 index over a fixed set of exemplars."""

    documents: tuple[Exemplar, ...]

    def __post_init__(self) -> None:
        self._tokens = [tokenize(d.line + " " + d.why) for d in self.documents]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._frequencies = [Counter(t) for t in self._tokens]
        document_count = len(self.documents)
        containing: Counter[str] = Counter()
        for tokens in self._tokens:
            containing.update(set(tokens))
        self._idf = {
            term: math.log(
                1 + (document_count - count + 0.5) / (count + 0.5)
            )
            for term, count in containing.items()
        }

    def score(self, query: str) -> list[tuple[Exemplar, float]]:
        terms = tokenize(query)
        scored: list[tuple[Exemplar, float]] = []
        for i, document in enumerate(self.documents):
            total = 0.0
            length = self._lengths[i] or 1
            for term in terms:
                frequency = self._frequencies[i].get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + _K1 * (
                    1 - _B + _B * length / (self._avg_length or 1)
                )
                total += self._idf.get(term, 0.0) * frequency * (_K1 + 1) / denominator
            scored.append((document, total))
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def top(self, query: str, k: int) -> list[Exemplar]:
        return [doc for doc, score in self.score(query)[:k] if score > 0]


_SIGNAL_INDEX = Index(SIGNALS)
_NOISE_INDEX = Index(NOISE)
_ALL_INDEX = Index(EXEMPLARS)


def _fill(retrieved: list[Exemplar], pool: tuple[Exemplar, ...], k: int) -> list[Exemplar]:
    """Top a class up to ``k`` from its head, without repeating a hit.

    Short queries score against only part of the bank, which would otherwise
    leave the classes unbalanced. Balance is worth more here than the marginal
    relevance of the last example: the ratio the model sees is itself a claim
    about the base rate, and a block of five signals and one noise line teaches
    the wrong prior however apt each individual line is.
    """
    out = list(retrieved[:k])
    for exemplar in pool:
        if len(out) >= k:
            break
        if exemplar not in out:
            out.append(exemplar)
    return out


def select(query: str, *, k_each: int = 3) -> list[Exemplar]:
    """The exemplars most worth showing for this log, balanced across classes."""
    signals = _fill(_SIGNAL_INDEX.top(query, k_each), SIGNALS, k_each)
    noise = _fill(_NOISE_INDEX.top(query, k_each), NOISE, k_each)
    # Interleave so neither class leads: models weight the first example more
    # than the last, and which class leads should not be an accident of order.
    out: list[Exemplar] = []
    for pair in zip(signals, noise):
        out.extend(pair)
    out.extend(signals[len(noise):])
    out.extend(noise[len(signals):])
    return out


def render(exemplars: list[Exemplar]) -> str:
    """The few-shot block, in the shape the scanner's own output takes."""
    if not exemplars:
        return ""
    lines = ["Worked examples of the judgement, from other runs:", ""]
    for exemplar in exemplars:
        verdict = "REPORT" if exemplar.signal else "IGNORE"
        lines.append(f"  {verdict}  {exemplar.line}")
        lines.append(f"          -> {exemplar.why}")
    return "\n".join(lines)
