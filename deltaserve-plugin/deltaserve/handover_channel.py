# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A direct line from the engine to every rank's backward child, for the
adapter handover at the end of a finetuning job.

The backward child is the worker's child: its command pipe ends in the worker,
so the engine could only reach it through a worker call — synchronous on the
engine loop, queued behind the step in flight, and sharing a strictly
question-and-answer pipe with the backward trigger and its ack. For the
handover none of that is wanted: saving a trained adapter and switching to the
next job's are not time-critical, and nothing about them needs the worker.

So each backward child also listens on a unix socket of its own
(``<DSERVE_HANDOVER_DIR>/rank<N>.sock``; the engine creates the directory
before the workers exist and its name travels in their environment), served by
one thread next to the child's command loop, and an ENGINE BACKGROUND THREAD
talks to it:

    ("save", shard_path)             snapshot the trained masters to host
                                     memory and write this rank's shard
    ("switch", adapter_dir, hparams) take the next job's adapter (the child
                                     reads the file itself); with
                                     ``hparams["dry_run"]`` only check it
    ("ping",)

The engine loop takes no part: it starts the background thread and later sees
that it has finished.

Only valid while the child is not training — no backward in flight. The
callers guarantee it the way they did for the worker calls: a job's adapter is
saved after its training stage is done, and the switch waits until nothing of
the previous job is pending. The existing command pipe is not touched.
"""

import hashlib
import os
import threading
from multiprocessing.connection import Client, Listener

from deltaserve import dprint

ENV = "DSERVE_HANDOVER_DIR"


def socket_path(directory: str, rank: int) -> str:
    return os.path.join(directory, f"rank{int(rank)}.sock")


def _authkey(path: str) -> bytes:
    # Not a secret: the directory's permissions (0700) are the access control.
    return hashlib.sha256(b"deltaserve-handover" + path.encode()).digest()


def make_directory() -> str:
    """Engine: the directory the children's sockets live in (short: unix
    socket paths are limited to 108 bytes)."""
    import tempfile

    return tempfile.mkdtemp(prefix=f"dserve-handover-{os.getpid()}-", dir="/tmp")


# ------------------------------------------------------------ child side
def write_shard(state: dict, path: str) -> int:
    """``state`` as an fp32 safetensors file; it appears under its name only
    when complete."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()},
              tmp)
    os.replace(tmp, path)
    return len(state)


def handle(svc, msg):
    """One request against a backward service. Returns the reply tuple;
    never raises."""
    try:
        op = msg[0]
        if op == "ping":
            return ("ok", {"pid": os.getpid()})
        if op == "save":
            return ("ok", {"tensors": write_shard(svc.export_lora_state(), msg[1])})
        if op == "switch":
            import torch
            from safetensors.torch import load_file

            from deltaserve.ft_meta import read_lora_scaling

            directory, hparams = msg[1], dict(msg[2] or {})
            device = (f"cuda:{svc.device_index}" if torch.cuda.is_available()
                      else "cpu")
            raw = load_file(os.path.join(directory, "adapter_model.safetensors"),
                            device=device)
            ft = {k: v.float() for k, v in raw.items()}
            hparams["lora_scaling"] = read_lora_scaling(directory)
            return ("ok", svc.switch_adapter(ft, hparams))
        return ("error", f"unknown request {op!r}")
    except Exception as e:  # noqa: BLE001 — reported to the engine
        return ("error", f"{type(e).__name__}: {e}")


def _serve(svc, conn) -> None:
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            try:
                conn.send(handle(svc, msg))
            except OSError:
                return
    finally:
        try:
            conn.close()
        except OSError:
            pass


def start_listener(svc, rank: int | None = None):
    """Backward child: listen for the engine, if the engine offers a
    directory. Returns the socket path, or None (no direct line: the handover
    goes through the worker, as before)."""
    directory = os.environ.get(ENV)
    if not directory or not os.path.isdir(directory):
        return None
    path = socket_path(directory, svc.device_index if rank is None else rank)
    try:
        try:
            os.unlink(path)
        except OSError:
            pass
        listener = Listener(path, family="AF_UNIX", authkey=_authkey(path))
        os.chmod(path, 0o600)
    except Exception as e:  # noqa: BLE001 — never keep the child from starting
        dprint(f"[backward] no handover line ({type(e).__name__}: {e})")
        return None

    def accept():
        while True:
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001 — the child is going away
                return
            threading.Thread(target=_serve, args=(svc, conn), daemon=True,
                             name="deltaserve-handover").start()

    threading.Thread(target=accept, daemon=True,
                     name="deltaserve-handover-accept").start()
    return path


# ----------------------------------------------------------- engine side
class HandoverClient:
    """The engine's end: one connection per rank's backward child. Used from a
    background thread; calls are serialized."""

    def __init__(self, directory: str, ranks: int, timeout_s: float = 120.0) -> None:
        self.directory, self.ranks = directory, int(ranks)
        self.timeout_s = float(timeout_s)
        self._conns: dict = {}
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Every rank's child is listening."""
        return all(os.path.exists(socket_path(self.directory, r))
                   for r in range(self.ranks))

    def _conn(self, rank: int):
        conn = self._conns.get(rank)
        if conn is None:
            path = socket_path(self.directory, rank)
            conn = self._conns[rank] = Client(path, family="AF_UNIX",
                                              authkey=_authkey(path))
        return conn

    def _drop(self, rank: int) -> None:
        conn = self._conns.pop(rank, None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def call_all(self, make_msg) -> list:
        """``make_msg(rank) -> request`` to every rank at once; the replies'
        payloads in rank order. Raises RuntimeError with the first rank's
        error (every rank is still read, so no reply is left on a line)."""
        with self._lock:
            sent = []
            try:
                for rank in range(self.ranks):
                    self._conn(rank).send(make_msg(rank))
                    sent.append(rank)
            except Exception as e:
                for rank in range(self.ranks):
                    self._drop(rank)
                raise RuntimeError(f"backward child of rank {len(sent)} is not "
                                   f"reachable ({type(e).__name__}: {e})") from e
            replies, error = [], None
            for rank in sent:
                try:
                    conn = self._conn(rank)
                    if not conn.poll(self.timeout_s):
                        raise TimeoutError("no answer")
                    reply = conn.recv()
                except Exception as e:  # noqa: BLE001
                    self._drop(rank)
                    reply = ("error", f"rank {rank}: {type(e).__name__}: {e}")
                if reply[0] != "ok" and error is None:
                    error = reply[1] if len(reply) > 1 else "failed"
                replies.append(reply[1] if len(reply) > 1 else None)
            if error is not None:
                raise RuntimeError(error)
            return replies

    def close(self) -> None:
        with self._lock:
            for rank in list(self._conns):
                self._drop(rank)
