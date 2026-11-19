# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Indexing on the GPU: a process of its own that shares the route encoder's
weights with the worker.

The worker that holds the route encoder (``route_worker``) spawns this process
and hands it the encoder's parameters over CUDA IPC — the same memory, not a
copy, exactly as the backward child gets the model's weights. The process runs
the encoder in its own CUDA context on indexing chunks the engine's
``IndexingRunner`` sends over a unix socket.

Why a process and not a thread in the worker: a fault here (a CUDA error, an
out-of-memory in a large chunk) must not take the inference worker with it;
its activations come out of its own allocator, not out of vLLM's budget; and
it does not share the worker's interpreter lock. The price is one more CUDA
context on the card (a few hundred MiB outside ``gpu_memory_utilization``).

**When it may run** — the scheduler decides, the process obeys:

- ``IndexingGrant`` is one byte of shared memory the ENGINE writes: 1 while no
  inference request is present (after a quiet period), 0 from the moment one
  arrives (``IndexingPlacement``). The engine writes it directly — an idle
  engine issues no steps to carry a message.
- The process checks it before every encoder layer. A cleared grant aborts the
  chunk there (under a millisecond away) and answers ``preempted``; the runner
  gives the same samples to the CPU.

So GPU indexing never shares the card with inference: it uses the GPU only
while the engine has no inference request at all.
"""

import hashlib
import os
import struct
import threading
from multiprocessing.connection import Client, Listener

from deltaserve import dprint

GRANT_ENV = "DSERVE_INDEX_GRANT_SHM"


class IndexingGrant:
    """May indexing use the GPU? One byte in POSIX shared memory: the engine
    writes, the indexing process reads."""

    def __init__(self, shm, owner: bool) -> None:
        self._shm, self._owner = shm, owner
        self._pack = struct.Struct("<B")

    @classmethod
    def create(cls) -> "IndexingGrant":
        from multiprocessing import shared_memory

        name = f"dserve_index_grant_{os.getpid()}"
        try:        # left by a crashed engine whose pid came round again
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=8)
        grant = cls(shm, owner=True)
        grant.set(False)
        return grant

    @classmethod
    def attach(cls, name: str) -> "IndexingGrant":
        from multiprocessing import shared_memory

        return cls(shared_memory.SharedMemory(name=name, create=False), owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def set(self, granted: bool) -> None:
        self._pack.pack_into(self._shm.buf, 0, 1 if granted else 0)

    def is_set(self) -> bool:
        return bool(self._pack.unpack_from(self._shm.buf, 0)[0])

    def close(self) -> None:
        try:
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except Exception:  # noqa: BLE001
            pass


def socket_path(index_dir: str) -> str:
    """Where the indexing process of this index directory listens. Unix socket
    paths are short (108 bytes), so it lives under /tmp, keyed by the directory."""
    key = hashlib.sha256(os.path.abspath(index_dir).encode()).hexdigest()[:16]
    return os.path.join("/tmp", f"deltaserve-indexer-{os.getuid()}-{key}.sock")


def _authkey(path: str) -> bytes:
    # Not a secret: the socket file's permissions (0600) are the access control.
    return hashlib.sha256(b"deltaserve-indexer" + path.encode()).digest()


class Preempted(Exception):
    """The grant was cleared while a chunk was being encoded."""


# --------------------------------------------------------------- the process
def _build_encoder(hf_config: dict, state: dict, device: str):
    """The encoder on ``device`` with ``state``'s tensors AS its parameters
    (the worker's memory, shared)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(**hf_config)
    model = AutoModel.from_config(config)
    for name, param in model.named_parameters():
        param.data = state[name]
        param.requires_grad_(False)
    return model.to(device).eval()      # parameters are there already: buffers move


def _encode(model, tok, grant, prefix, max_tokens, device, texts):
    import torch

    with torch.inference_mode():
        enc = tok([prefix + t for t in texts], padding=True, truncation=True,
                  max_length=max_tokens, return_tensors="pt").to(device)
        if grant is None or not grant.is_set():
            raise Preempted()
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return pooled.cpu().numpy()


def _serve(conn, model, tok, grant, prefix, max_tokens, device, stats) -> None:
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            op = msg[0]
            if op == "ping":
                conn.send(("pong", dict(stats)))
            elif op == "encode":
                try:
                    emb = _encode(model, tok, grant, prefix, max_tokens, device,
                                  list(msg[1]))
                    stats["chunks"] += 1
                    reply = ("ok", emb)
                except Preempted:
                    stats["preempted"] += 1
                    reply = ("preempted",)
                except Exception as e:  # noqa: BLE001 — reported to the engine
                    stats["errors"] += 1
                    reply = ("error", f"{type(e).__name__}: {e}")
                try:
                    conn.send(reply)
                except OSError:
                    return                 # the engine hung up meanwhile
            else:
                conn.send(("error", f"unknown request {op!r}"))
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _index_process_main(pipe, hf_config, encoder_name, prefix, max_tokens,
                        device, path) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from deltaserve.backward_process import arm_parent_death_signal

        arm_parent_death_signal("indexer")
    except Exception:  # noqa: BLE001 — never block the process on this
        pass
    try:
        state = pipe.recv()                      # CUDA-IPC handles
        model = _build_encoder(hf_config, state, device)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder_name)
        name = os.environ.get(GRANT_ENV)
        grant = IndexingGrant.attach(name) if name else None

        def check(module, args):
            # No grant from the engine = never on the GPU.
            if grant is None or not grant.is_set():
                raise Preempted()

        for layer in model.encoder.layer:
            layer.register_forward_pre_hook(check)
        try:
            os.unlink(path)
        except OSError:
            pass
        listener = Listener(path, family="AF_UNIX", authkey=_authkey(path))
        os.chmod(path, 0o600)
    except Exception as e:  # noqa: BLE001
        pipe.send({"error": f"{type(e).__name__}: {e}"})
        return
    pipe.send({"ready": True, "pid": os.getpid(),
               "grant": grant is not None})
    stats = {"chunks": 0, "preempted": 0, "errors": 0}
    import signal

    # The worker stops this process with SIGTERM: leave quietly (the listener
    # removes its own socket file).
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    try:
        while True:
            conn = listener.accept()
            threading.Thread(target=_serve, daemon=True, args=(
                conn, model, tok, grant, prefix, max_tokens, device, stats)).start()
    except (OSError, KeyboardInterrupt):
        pass


class IndexEncoderProcess:
    """Worker side: spawn the indexing process and share the encoder with it."""

    def __init__(self, rc, device: str, path: str) -> None:
        self.rc, self.device, self.path = rc, device, path
        self._proc = self._pipe = None

    def start(self, model) -> dict:
        import torch.multiprocessing as mp

        from deltaserve.backward_process import _allow_children

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        rc = self.rc
        self._proc = ctx.Process(
            target=_index_process_main, name="deltaserve-indexer", daemon=True,
            args=(child, model.config.to_dict(), rc.encoder,
                  rc.encoder_query_prefix, int(rc.encoder_max_tokens),
                  self.device, self.path))
        with _allow_children():
            self._proc.start()
        # The parameters themselves: CUDA IPC, the child maps the same memory.
        parent.send({n: p.data for n, p in model.named_parameters()})
        self._pipe = parent
        if not parent.poll(120.0):
            raise RuntimeError("the indexing process did not come up")
        ack = parent.recv()
        if "error" in ack:
            raise RuntimeError(f"indexing process: {ack['error']}")
        return ack

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.kill()
        try:
            os.unlink(self.path)
        except OSError:
            pass


# ------------------------------------------------------------ engine side
class SharedEncoderExecutor:
    """The runner's GPU executor: chunks go to the indexing process. ``poll``
    raises ``Preempted`` when inference took the GPU back mid-chunk (the
    runner redoes the samples on the CPU), any other failure as RuntimeError.
    ``cancel`` stops waiting for the chunk in flight; the process drops it at
    its next layer because the grant is cleared by then."""

    device, loaded = "gpu", True

    def __init__(self, path: str, timeout_s: float = 60.0) -> None:
        self.path, self.timeout_s = path, timeout_s
        self._conn = None
        self._lock = threading.Lock()
        self._seq = 0
        self._job = None            # [seq, Event, embeddings, error]

    @property
    def ready(self) -> bool:
        """The process is listening (it comes up after the worker's model load)."""
        return os.path.exists(self.path)

    def _call(self, msg):
        with self._lock:
            try:
                if self._conn is None:
                    self._conn = Client(self.path, family="AF_UNIX",
                                        authkey=_authkey(self.path))
                self._conn.send(msg)
                if not self._conn.poll(self.timeout_s):
                    raise TimeoutError("the indexing process did not answer")
                return self._conn.recv()
            except Exception:
                conn, self._conn = self._conn, None
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
                raise

    def encode(self, texts: list):
        reply = self._call(("encode", list(texts)))
        if reply[0] == "ok":
            return reply[1]
        if reply[0] == "preempted":
            raise Preempted()
        raise RuntimeError(reply[1] if len(reply) > 1 else "indexing process error")

    def ping(self):
        try:
            reply = self._call(("ping",))
            return reply[1] if reply[0] == "pong" else None
        except Exception:  # noqa: BLE001
            return None

    def submit(self, texts: list, threads: int = 0) -> None:
        self._seq += 1
        job = [self._seq, threading.Event(), None, None]
        self._job = job

        def run():
            try:
                job[2] = self.encode(texts)
            except Exception as e:  # noqa: BLE001 — surfaced by poll()
                job[3] = e
            job[1].set()

        threading.Thread(target=run, daemon=True,
                         name="deltaserve-gpu-chunk").start()

    def poll(self, timeout: float = 0.0):
        job = self._job
        if job is None or not job[1].wait(timeout):
            return None
        self._job = None
        if job[3] is not None:
            raise job[3]
        return job[2]

    def cancel(self) -> None:
        self._job = None

    def unload(self) -> None:
        """Nothing to free: the weights are the worker's."""

    def stop(self) -> None:
        self._job = None
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


def log_started(ack: dict, device: str) -> None:
    dprint(f"[indexing] GPU indexing process pid={ack.get('pid')} on {device}: "
           f"shares the route encoder's weights"
           + ("" if ack.get("grant") else " (no grant from the engine: it "
              "will never be allowed to encode)"))
