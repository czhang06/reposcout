"""BM25 keyword search with a code-aware tokenizer.

Vector search is good at meaning ("reset a task" ~ "clear a task instance") but weak at
exact identifiers. BM25 is the opposite. `clear_task_instances` is indexed as the whole
token plus its parts (clear, task, instances), and `DagRun` as dagrun + dag + run, so
both "clear_task_instances" and "clear task instances" match. Tokens are stemmed
(Snowball, as Postgres full-text uses) so "clearing" matches "clear".
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import snowballstemmer

_STEM = snowballstemmer.stemmer("english")

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it its of on or so that the "
    "this to was what when where which why with self none true false return def class "
    "import".split()
)


def tokenize(text: str) -> list[str]:
    out = []
    for word in _WORD.findall(text):
        lower = word.lower()
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        if lower not in STOPWORDS:
            out.append(lower)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    # Stem plain words; keep snake_case identifiers exact.
    return [t if "_" in t else _STEM.stemWord(t) for t in out]


class BM25:
    """Okapi BM25 over a fixed corpus. k1/b are the usual defaults."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter] = [Counter(tokenize(d)) for d in docs]
        self.doc_len = np.array([sum(tf.values()) for tf in self.doc_tfs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Inverted index: term -> (doc indices, term frequencies)
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.doc_tfs):
            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
        self.postings = {
            t: (np.array([i for i, _ in p]), np.array([c for _, c in p], dtype=np.float32))
            for t, p in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.doc_tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avg_len or 1.0))
        for t in set(tokenize(query)):
            if t not in self.postings:
                continue
            idx, tf = self.postings[t]
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s

    def top_k(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        k = min(k, int((s > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top]
