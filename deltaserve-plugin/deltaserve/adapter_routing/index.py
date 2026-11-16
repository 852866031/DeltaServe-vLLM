# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter centroids (ARIADNE): the index and the resumable indexing job.

An adapter's centroid is the mean of the L2-normalised sentence embeddings of
its training samples, re-normalised. A request is routed to the adapter whose
centroid has the highest cosine similarity with the request's embedding.

Indexing is a running sum, so its whole state is ``(sum, count, cursor)``:
``Indexer`` hands out the next chunk of samples and accepts their embeddings;
after every accepted chunk the state is checkpointed. Any executor (CPU, GPU)
can take the next chunk, a chunk whose result never comes back is simply handed
out again, and a restart continues from the checkpoint.

Pure numpy; no torch / vLLM import.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _atomic_savez(path: str | os.PathLike, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def select_samples(data_path: str, max_samples: int, seed: int = 0) -> list[str]:
    """The texts an adapter's centroid is built from: the non-empty lines of
    its finetuning corpus (one sample per line), at most ``max_samples`` of
    them, drawn without replacement in a fixed (seeded) order."""
    with open(data_path, encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    texts = [t for t in texts if t]
    if len(texts) > max_samples:
        order = np.random.default_rng(seed).permutation(len(texts))[:max_samples]
        texts = [texts[i] for i in order]
    return texts


def fingerprint(texts: list[str], encoder: str, prefix: str,
                max_tokens: int) -> str:
    """Identifies WHAT is being indexed and HOW; a checkpoint with another
    fingerprint belongs to a different job and is not resumed."""
    h = hashlib.sha256()
    h.update(json.dumps([encoder, prefix, int(max_tokens), len(texts)]).encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class IndexingState:
    fingerprint: str
    total: int
    cursor: int = 0
    count: int = 0
    sum: np.ndarray | None = None     # float64 [dim]; None until the first chunk

    def save(self, path) -> None:
        _atomic_savez(
            path, fingerprint=np.array(self.fingerprint),
            meta=np.array([self.total, self.cursor, self.count], dtype=np.int64),
            sum=np.zeros(0) if self.sum is None else self.sum)

    @classmethod
    def load(cls, path) -> "IndexingState | None":
        try:
            with np.load(path, allow_pickle=False) as z:
                total, cursor, count = (int(v) for v in z["meta"])
                s = z["sum"].astype(np.float64)
                return cls(str(z["fingerprint"]), total, cursor, count,
                           s if s.size else None)
        except (OSError, KeyError, ValueError):
            return None


class Indexer:
    """One adapter's indexing job, resumable chunk by chunk."""

    def __init__(self, texts: list[str], fp: str,
                 checkpoint_path: str | os.PathLike | None = None) -> None:
        self.texts = texts
        self.checkpoint_path = checkpoint_path
        self.resumed_from = 0
        state = IndexingState.load(checkpoint_path) if checkpoint_path else None
        if state is not None and (state.fingerprint != fp
                                  or state.total != len(texts)
                                  or not 0 <= state.cursor <= len(texts)):
            state = None      # another dataset / encoder: start over
        if state is None:
            state = IndexingState(fingerprint=fp, total=len(texts))
        else:
            self.resumed_from = state.cursor
        self.state = state

    @property
    def done(self) -> bool:
        return self.state.cursor >= self.state.total

    @property
    def progress(self) -> tuple[int, int]:
        return self.state.cursor, self.state.total

    def next_chunk(self, n: int) -> tuple[int, list[str]]:
        """The next ``n`` samples to encode and the cursor they start at. Does
        not advance: the same chunk is returned until it is committed."""
        c = self.state.cursor
        return c, self.texts[c:c + max(1, int(n))]

    def commit(self, start: int, embeddings: np.ndarray) -> bool:
        """Add a chunk's embeddings ([n, dim], one row per sample handed out at
        ``start``). False (and nothing changes) for a chunk that is not the
        current one — a late result of a chunk that was given up on."""
        st = self.state
        emb = np.asarray(embeddings, dtype=np.float64)
        if start != st.cursor or emb.ndim != 2 or emb.shape[0] == 0 \
                or start + emb.shape[0] > st.total:
            return False
        emb = l2_normalize(emb)
        st.sum = emb.sum(0) if st.sum is None else st.sum + emb.sum(0)
        st.count += emb.shape[0]
        st.cursor += emb.shape[0]
        if self.checkpoint_path:
            st.save(self.checkpoint_path)
        return True

    def centroid(self) -> np.ndarray:
        if not self.done or self.state.sum is None:
            raise RuntimeError("indexing is not finished")
        return l2_normalize(self.state.sum / self.state.count).astype(np.float32)


class CentroidIndex:
    """One centroid per routable adapter, with the encoder that built them."""

    def __init__(self, encoder: str = "", prefix: str = "",
                 max_tokens: int = 0) -> None:
        self.encoder, self.prefix, self.max_tokens = encoder, prefix, int(max_tokens)
        self.names: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def add(self, name: str, centroid: np.ndarray) -> None:
        c = l2_normalize(np.asarray(centroid, dtype=np.float32).reshape(1, -1))
        if len(self.names) and c.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"centroid dim {c.shape[1]} != index dim {self._matrix.shape[1]}")
        if name in self.names:
            self._matrix[self.names.index(name)] = c[0]
            return
        self._matrix = c if not len(self.names) else np.vstack([self._matrix, c])
        self.names.append(name)

    def remove(self, name: str) -> bool:
        if name not in self.names:
            return False
        i = self.names.index(name)
        self.names.pop(i)
        self._matrix = np.delete(self._matrix, i, axis=0)
        return True

    def scores(self, embedding: np.ndarray) -> np.ndarray:
        """Cosine of one embedding against every centroid, in ``names`` order."""
        if not self.names:
            return np.zeros(0, dtype=np.float32)
        e = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        return self._matrix @ e[0]

    def route(self, embedding: np.ndarray,
              threshold: float = -1.0) -> tuple[str | None, float]:
        """(best adapter, its score); the adapter is None — fall back to the
        base model — when the index is empty or the best score is below
        ``threshold``."""
        s = self.scores(embedding)
        if not s.size:
            return None, float("nan")
        i = int(np.argmax(s))
        best = float(s[i])
        return (self.names[i] if best >= threshold else None), best

    def save(self, path) -> None:
        _atomic_savez(
            path, names=np.array(self.names, dtype=str), matrix=self._matrix,
            meta=np.array(json.dumps({"encoder": self.encoder,
                                      "prefix": self.prefix,
                                      "max_tokens": self.max_tokens})))

    @classmethod
    def load(cls, path) -> "CentroidIndex":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            idx = cls(meta["encoder"], meta["prefix"], meta["max_tokens"])
            idx.names = [str(n) for n in z["names"]]
            idx._matrix = z["matrix"].astype(np.float32)
        return idx
