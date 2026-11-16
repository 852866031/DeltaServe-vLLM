# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The routing decision: which adapter, how sure, or the base model.

e5 embeddings sit in a narrow cone, so the raw cosine to a centroid says little
about how good a match is. Two signals do carry information (measured on task
scores, see the adapter-route experiments):

- the **centred score**: cosine after subtracting the mean of the pool's
  centroids from both the request and the centroids;
- the **margin**: the gap between the best and the second-best adapter.

``confidence = z(centred top-1 score) + z(margin)``. The z-scores, and the
threshold, come from the pool's own **reference samples** — a few training-
sample embeddings kept per adapter during indexing: the threshold is the
confidence below which the lowest ``100 - keep_percent`` % of them fall. So one
user setting (``keep_percent``) yields a threshold that follows the pool as it
changes. A ``RoutingModel`` is rebuilt whenever the pool's snapshot changes
(about 2 ms for 32 adapters).

A request then has one of three outcomes:

- confidence below the threshold → the base model;
- one adapter clearly ahead → that adapter;
- several within ``close_band`` of the best → one of them, chosen by ``prefer``
  (already in a GPU slot, then most used): among close candidates the choice
  makes no measurable difference to the output, but it saves a slot swap.

The threshold assumes requests look like the training data. They never quite
do, and the narrower the training set the bigger the gap: with a few dozen
templated samples per adapter, genuine requests score several units below
every reference sample while unrelated ones are tens of units away. ``slack``
lowers the threshold by that many confidence units for such pools; the
decision log is there to set it from real traffic.

Small pools: centring needs several centroids (with one, the mean IS the
centroid). Below ``min_pool`` adapters the confidence is the raw score against
the reference samples' own raw scores — a much weaker signal. Without any
reference samples nothing is gated (always top-1).

Pure numpy.
"""

from dataclasses import dataclass, field

import numpy as np

from deltaserve.adapter_routing.index import CentroidIndex, l2_normalize

CENTRED, RAW, UNGATED = "centred", "raw", "ungated"


@dataclass
class Choice:
    adapter: str | None           # None = base model
    score: float | None = None    # raw cosine of the best adapter
    confidence: float | None = None
    margin: float | None = None
    candidates: list = field(default_factory=list)   # the close set, best first
    low_confidence: bool = False


class RoutingModel:
    def __init__(self, index: CentroidIndex, keep_percent: float = 80.0,
                 close_band: float = 0.05, min_pool: int = 4,
                 slack: float = 0.0) -> None:
        self.names = list(index.names)
        self.keep_percent, self.close_band = float(keep_percent), float(close_band)
        self.slack = float(slack)
        C = index._matrix.astype(np.float32) if self.names else np.zeros((0, 0), np.float32)
        self.C = C
        refs = [index.refs[n] for n in self.names if n in index.refs]
        R = np.concatenate(refs) if refs else None
        self.n_refs = 0 if R is None else len(R)
        self.mode, self.threshold = UNGATED, None
        self.mu = None
        self.stats = (0.0, 1.0, 0.0, 1.0)
        if len(self.names) >= max(2, int(min_pool)) and R is not None:
            self.mode = CENTRED
            self.mu = C.mean(0)
            self.Cc = l2_normalize(C - self.mu)
            s1, m = self._signals(R)
            self.stats = (float(s1.mean()), float(s1.std()) or 1.0,
                          float(m.mean()), float(m.std()) or 1.0)
            conf = self._confidence(s1, m)
        elif self.names and R is not None:
            self.mode = RAW
            s1 = (R @ C.T).max(1)
            self.stats = (float(s1.mean()), float(s1.std()) or 1.0, 0.0, 1.0)
            conf = (s1 - self.stats[0]) / self.stats[1]
        else:
            conf = None
        if conf is not None and self.keep_percent < 100.0:
            self.threshold = float(
                np.percentile(conf, 100.0 - self.keep_percent)) - self.slack

    def _signals(self, E: np.ndarray):
        S = l2_normalize(E - self.mu) @ self.Cc.T
        top2 = np.partition(S, -2, axis=1)[:, -2:]
        return top2[:, 1], top2[:, 1] - top2[:, 0]

    def _confidence(self, s1, margin):
        a, b, c, d = self.stats
        return (s1 - a) / b + (margin - c) / d

    def describe(self) -> dict:
        return {"mode": self.mode, "adapters": len(self.names),
                "reference_samples": self.n_refs, "threshold": self.threshold,
                "keep_percent": self.keep_percent, "slack": self.slack,
                "close_band": self.close_band}

    def decide(self, emb: np.ndarray, prefer=None,
               raw_floor: float = -1.0) -> Choice:
        """``prefer(name) -> sortable``: higher is preferred among close
        candidates (e.g. (in a GPU slot, usage)). ``raw_floor``: a manual
        absolute cut on the raw score, applied first."""
        if not self.names:
            return Choice(None)
        e = l2_normalize(np.asarray(emb, np.float32).reshape(1, -1))
        raw = (e @ self.C.T)[0]
        if self.mode == CENTRED:
            s = (l2_normalize(e - self.mu) @ self.Cc.T)[0]
        else:
            s = raw
        order = np.argsort(-s)
        best = int(order[0])
        margin = float(s[best] - s[order[1]]) if len(order) > 1 else None
        conf = None
        if self.mode == CENTRED:
            conf = float(self._confidence(s[best], margin))
        elif self.mode == RAW:
            conf = float((raw[best] - self.stats[0]) / self.stats[1])
        score = float(raw[best])
        close = [self.names[i] for i in order if s[i] >= s[best] - self.close_band] \
            if self.mode == CENTRED else [self.names[best]]
        if score < raw_floor:
            return Choice(None, score, conf, margin, close)
        if self.threshold is not None and conf is not None and conf < self.threshold:
            return Choice(None, score, conf, margin, close, low_confidence=True)
        pick = close[0]
        if len(close) > 1 and prefer is not None:
            # stable: the best-scoring one wins among equally preferred
            pick = max(close, key=lambda n: (prefer(n), -close.index(n)))
        return Choice(pick, score, conf, margin, close)


def admission_report(index: CentroidIndex, name: str, centroid, refs=None,
                     min_pool: int = 4) -> dict:
    """What adding ``name`` would do to the pool ``index`` (the active
    adapters), measured on reference samples — no encoding, no GPU:

    - ``self_recall``: share of the candidate's own reference samples that the
      new pool sends to it (can it be told apart?);
    - ``stolen``: per existing adapter, the share of ITS reference samples the
      candidate would take (``stolen_max`` the worst case, ``stolen_from`` whom);
    - ``nearest`` / ``nearest_cosine``: the existing adapter closest to it.

    Scores are centred like the router's when the new pool is big enough."""
    centroid = l2_normalize(np.asarray(centroid, np.float32).reshape(1, -1))[0]
    others = [n for n in index.names if n != name]
    report = {"pool_before": len(others), "self_recall": None, "stolen": {},
              "stolen_max": 0.0, "stolen_from": None, "nearest": None,
              "nearest_cosine": None}
    if not others:
        return report
    C = np.vstack([index.centroid_of(n) for n in others] + [centroid])
    if len(C) >= max(2, int(min_pool)):
        mu = C.mean(0)
        Cc = l2_normalize(C - mu)
        tr = lambda x: l2_normalize(x - mu)          # noqa: E731
    else:
        Cc, tr = C, (lambda x: x)
    sims = Cc[:-1] @ Cc[-1]
    j = int(np.argmax(sims))
    report["nearest"], report["nearest_cosine"] = others[j], round(float(sims[j]), 4)
    me = len(C) - 1
    if refs is not None and len(refs):
        top = (tr(np.asarray(refs, np.float32)) @ Cc.T).argmax(1)
        report["self_recall"] = round(float((top == me).mean()), 4)
    for i, n in enumerate(others):
        r = index.refs.get(n)
        if r is None or not len(r):
            continue
        S = tr(r) @ Cc.T
        before = (S[:, :-1].argmax(1) == i)
        taken = (S.argmax(1) == me) & before
        share = float(taken.sum() / max(1, before.sum()))
        if share > 0:
            report["stolen"][n] = round(share, 4)
        if share > report["stolen_max"]:
            report["stolen_max"], report["stolen_from"] = round(share, 4), n
    return report
