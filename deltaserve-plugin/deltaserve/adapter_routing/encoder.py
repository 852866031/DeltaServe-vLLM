# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The sentence encoder (e5-large-v2 by default) and the CPU indexer process.

``SentenceEncoder`` is the plain ``transformers`` form of what
sentence-transformers does for the e5 family: prefix, truncate, mean-pool the
last hidden state over the attention mask, L2-normalise.

``CpuIndexerProcess`` runs one in a child process so its threads, its GIL and
its memory stay out of the vLLM processes. The child runs under ``SCHED_IDLE``,
can be pinned to a core set, and takes the thread count with every chunk (the
scheduler's level decides it). Its interface is asynchronous — ``submit`` a chunk,
``poll`` for the embeddings — which is what ``runner.IndexingRunner`` expects
of any executor.

torch / transformers are imported only when an encoder is actually loaded.
"""

import multiprocessing as mp
import os

import numpy as np


def parse_cpu_list(spec: str | None) -> set[int] | None:
    """'8-15,24' → {8..15, 24}; None / '' → None (no restriction)."""
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def apply_idle_priority() -> str:
    """Put the calling process in the lowest scheduling class. Returns which
    mechanism took effect."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        return "SCHED_IDLE"
    except (AttributeError, OSError):
        pass
    try:
        os.nice(19)
        return "nice 19"
    except OSError:
        return "none"


def pin_to_cpus(cpus: set[int] | None) -> set[int]:
    """Restrict the calling process to ``cpus`` (intersected with what it may
    use); returns the resulting set."""
    allowed = os.sched_getaffinity(0)
    if cpus:
        target = allowed & set(cpus)
        if target:
            os.sched_setaffinity(0, target)
            return target
    return allowed


class SentenceEncoder:
    def __init__(self, model: str, prefix: str = "query: ",
                 max_tokens: int = 512, device: str = "cpu") -> None:
        self.model_name, self.prefix = model, prefix
        self.max_tokens, self.device = int(max_tokens), device
        self._tok = self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(
            self.device).eval()

    def unload(self) -> None:
        self._tok = self._model = None

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """[n, dim] float32, L2-normalised, in the order of ``texts``."""
        import torch

        self.load()
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = [self.prefix + t for t in texts[i:i + batch_size]]
                enc = self._tok(batch, padding=True, truncation=True,
                                max_length=self.max_tokens,
                                return_tensors="pt").to(self.device)
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                out.append(pooled.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def _cpu_indexer_main(conn, model, prefix, max_tokens, cpus) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""      # this process never uses a GPU
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from deltaserve.backward_process import arm_parent_death_signal

        arm_parent_death_signal("cpu-indexer")
    except Exception:  # noqa: BLE001 — never block the worker on this
        pass
    conn.send(("hello", apply_idle_priority(), sorted(pin_to_cpus(cpus))))
    enc = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "stop":
            return
        if op == "unload":
            enc = None
            import gc

            gc.collect()
            continue
        _, seq, texts, threads = msg
        try:
            import torch

            torch.set_num_threads(max(1, int(threads)))
            if enc is None:
                enc = SentenceEncoder(model, prefix, max_tokens, "cpu")
            conn.send(("ok", seq, enc.encode(texts)))
        except Exception as e:  # noqa: BLE001 — reported to the parent
            conn.send(("error", seq, f"{type(e).__name__}: {e}"))


class CpuIndexerProcess:
    """Executor backed by a child process that encodes on the CPU."""

    device = "cpu"

    def __init__(self, model: str, prefix: str, max_tokens: int,
                 cpus: set[int] | None = None) -> None:
        self._args = (model, prefix, int(max_tokens), cpus)
        self._proc = self._conn = None
        self._seq = 0
        self._inflight: int | None = None
        self.loaded = False
        self.info: tuple | None = None        # (priority mechanism, cpu list)

    def _start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_cpu_indexer_main,
                                 args=(child, *self._args),
                                 name="deltaserve-cpu-indexer", daemon=True)
        cur = mp.current_process()
        was = cur._config.get("daemon", False)   # the engine may be daemonic
        cur._config["daemon"] = False
        try:
            self._proc.start()
        finally:
            cur._config["daemon"] = was
        child.close()
        self.loaded = False

    def submit(self, texts: list[str], threads: int) -> None:
        if self._inflight is not None:
            raise RuntimeError("a chunk is already in flight")
        self._start()
        self._seq += 1
        self._inflight = self._seq
        self._conn.send(("encode", self._seq, list(texts), int(threads)))

    def poll(self, timeout: float = 0.0):
        """The embeddings of the submitted chunk, or None if not ready."""
        if self._inflight is None:
            return None
        deadline_left = timeout
        while True:
            if not self._conn.poll(deadline_left):
                if not self._proc.is_alive():
                    self._inflight = None
                    raise RuntimeError("CPU indexer process died")
                return None
            try:
                msg = self._conn.recv()
            except EOFError:
                self._inflight = None
                raise RuntimeError("CPU indexer process died") from None
            deadline_left = 0.0
            if msg[0] == "hello":
                self.info = msg[1:]
                continue
            kind, seq, payload = msg
            if seq != self._inflight:
                continue                       # a cancelled chunk's late answer
            self._inflight = None
            if kind == "error":
                raise RuntimeError(f"CPU indexer: {payload}")
            self.loaded = True
            return payload

    def cancel(self) -> None:
        """Give up on the chunk in flight (its answer is ignored)."""
        self._inflight = None

    def unload(self) -> None:
        """Free the encoder's memory; the process stays, the next chunk reloads."""
        if self._proc is not None and self._proc.is_alive() \
                and self._inflight is None:
            self._conn.send(("unload",))
            self.loaded = False

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            self._conn.send(("stop",))
        except (OSError, ValueError):
            pass
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        self._inflight, self.loaded = None, False
