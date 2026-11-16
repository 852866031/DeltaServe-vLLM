# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The router service: one process that holds the sentence encoder and answers
"which adapter for this text?" for every API server.

- ``Router``: encoder + centroid index file + fallback threshold → a
  ``RouteDecision``. The index file is re-read when it changes (an adapter was
  indexed, disabled, ...), so the service follows the pool without a restart.
- ``serve``: the request loop on a unix socket. One encoder shared by every
  API server process; requests that arrive while it is busy are encoded
  together as one batch (``RouteBatcher``).
- ``RouterClient``: what an API server uses. It never raises: a service that is
  down, slow or broken yields a fallback decision (the request is served by the
  base model) and the client reconnects on the next request.

Run it by hand with::

    python -m deltaserve.adapter_routing.service --config <serving yaml> [--device cpu]

torch / transformers are imported only when the encoder first encodes.
"""

import hashlib
import os
import threading
import time
from multiprocessing.connection import Client, Listener

from deltaserve import dprint
from deltaserve.adapter_routing.index import CentroidIndex
from deltaserve.route_middleware import (  # noqa: F401 — re-exported
    BELOW_THRESHOLD,
    FALLBACK,
    NO_ADAPTERS,
    NO_TEXT,
    ROUTED,
    TIMEOUT,
    UNAVAILABLE,
    RouteDecision,
)

def socket_path(index_dir: str) -> str:
    """Where the service of this index directory listens. Unix socket paths are
    short (108 bytes), so it lives under /tmp, keyed by the directory."""
    key = hashlib.sha256(os.path.abspath(index_dir).encode()).hexdigest()[:16]
    return os.path.join("/tmp", f"deltaserve-router-{os.getuid()}-{key}.sock")


def _authkey(path: str) -> bytes:
    # Not a secret: the socket file's permissions (0600) are the access control.
    return hashlib.sha256(b"deltaserve-router" + path.encode()).digest()


class Router:
    def __init__(self, encoder, index_file: str, threshold: float = 0.0,
                 reload_s: float = 1.0) -> None:
        self.encoder, self.index_file = encoder, index_file
        self.threshold, self.reload_s = float(threshold), float(reload_s)
        self.index = CentroidIndex()
        self._mtime = None
        self._checked = 0.0
        self.requests = 0

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._checked < self.reload_s:
            return
        self._checked = now
        try:
            mtime = os.stat(self.index_file).st_mtime_ns
        except OSError:
            if self._mtime is not None:
                self.index, self._mtime = CentroidIndex(), None
            return
        if mtime != self._mtime:
            self.index, self._mtime = CentroidIndex.load(self.index_file), mtime
            dprint(f"[router] index loaded: {len(self.index)} adapters")

    def route_many(self, texts: list) -> list[RouteDecision]:
        """One decision per text, with ONE encoder call for all of them."""
        self.requests += len(texts)
        self.refresh()
        out: list = [None] * len(texts)
        todo = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                out[i] = RouteDecision(None, None, NO_TEXT)
            elif not len(self.index):
                out[i] = RouteDecision(None, None, NO_ADAPTERS)
            else:
                todo.append(i)
        if todo:
            embs = self.encoder.encode([texts[i] for i in todo])
            for i, emb in zip(todo, embs):
                name, score = self.index.route(emb, self.threshold)
                out[i] = RouteDecision(name, score, "" if name else BELOW_THRESHOLD)
        return out

    def route(self, text: str | None) -> RouteDecision:
        return self.route_many([text])[0]


class RouteBatcher:
    """Turns concurrent route requests into encoder batches.

    A single thread owns the encoder. When it is free it takes EVERYTHING that
    is waiting (up to ``max_batch``) and encodes it in one call. So a request
    arriving at an idle service is encoded at once, alone — no added latency —
    and under load the requests that queued up while the encoder was busy share
    one batch, which costs little more than a single request on a GPU.
    ``wait_s`` > 0 additionally holds a batch open that long to let it fill."""

    def __init__(self, router: Router, max_batch: int = 16,
                 wait_s: float = 0.0) -> None:
        import queue

        self.router = router
        self.max_batch, self.wait_s = max(1, int(max_batch)), float(wait_s)
        self._q: "queue.Queue" = queue.Queue()
        self._empty = queue.Empty
        self.batches = 0
        self.largest = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="router-batcher")
        self._thread.start()

    def submit(self, text) -> RouteDecision:
        """Blocks until this text's decision is ready; raises what the encoder
        raised."""
        item = [text, threading.Event(), None, None]
        self._q.put(item)
        item[1].wait()
        if item[3] is not None:
            raise item[3]
        return item[2]

    def _loop(self) -> None:
        while True:
            first = self._q.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait_s
            while len(batch) < self.max_batch:
                try:
                    left = deadline - time.monotonic()
                    nxt = (self._q.get(timeout=left) if left > 0
                           else self._q.get_nowait())
                except self._empty:
                    break
                if nxt is None:
                    self._stop = True
                    break
                batch.append(nxt)
            try:
                decisions = self.router.route_many([it[0] for it in batch])
                for it, d in zip(batch, decisions):
                    it[2] = d
            except Exception as e:  # noqa: BLE001 — handed to every waiter
                for it in batch:
                    it[3] = e
            self.batches += 1
            self.largest = max(self.largest, len(batch))
            for it in batch:
                it[1].set()
            if self._stop:
                return

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5.0)


def _handle(conn, router: Router, batcher: RouteBatcher) -> None:
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError, TypeError):   # closed, here or by serve()
                return
            try:
                if msg[0] == "route":
                    d = batcher.submit(msg[1])
                    reply = ("ok", d.adapter, d.score, d.reason)
                elif msg[0] == "ping":
                    reply = ("ok", len(router.index), router.requests,
                             batcher.batches, batcher.largest)
                else:
                    reply = ("error", f"unknown request {msg[0]!r}")
            except Exception as e:  # noqa: BLE001 — reported to the client
                reply = ("error", f"{type(e).__name__}: {e}")
            try:
                conn.send(reply)
            except OSError:
                return
    finally:
        conn.close()


def serve(router: Router, path: str, stop: threading.Event | None = None,
          ready: threading.Event | None = None, max_batch: int = 16,
          batch_wait_s: float = 0.0) -> None:
    """Answer route requests on the unix socket ``path`` until ``stop`` is set
    (forever when ``stop`` is None)."""
    try:
        os.unlink(path)
    except OSError:
        pass
    old = os.umask(0o177)                 # socket file readable by its owner only
    try:
        listener = Listener(path, family="AF_UNIX", authkey=_authkey(path))
    finally:
        os.umask(old)
    batcher = RouteBatcher(router, max_batch, batch_wait_s)
    stopping = threading.Event()
    conns = []

    def acceptor() -> None:
        while not stopping.is_set():
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001 — bad client or closing: keep going
                continue
            if stopping.is_set():
                conn.close()
                return
            conns.append(conn)
            threading.Thread(target=_handle, args=(conn, router, batcher),
                             daemon=True).start()

    thread = threading.Thread(target=acceptor, daemon=True, name="router-accept")
    thread.start()
    if ready is not None:
        ready.set()
    try:
        if stop is None:
            thread.join()
        else:
            stop.wait()
    finally:
        stopping.set()
        try:                                # wake the blocked accept()
            Client(path, family="AF_UNIX", authkey=_authkey(path)).close()
        except Exception:  # noqa: BLE001
            pass
        thread.join(timeout=2.0)
        listener.close()
        batcher.stop()
        for conn in conns:                  # hang up on every client
            try:
                conn.close()
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass


class RouterClient:
    """An API server's handle on the router service. Thread-safe; never raises."""

    def __init__(self, path: str, timeout_s: float = 0.2) -> None:
        self.path, self.timeout_s = path, float(timeout_s)
        self._conn = None
        self._lock = threading.Lock()
        self.failures = 0

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def _call(self, msg: tuple):
        if self._conn is None:
            self._conn = Client(self.path, family="AF_UNIX",
                                authkey=_authkey(self.path))
        self._conn.send(msg)
        if not self._conn.poll(self.timeout_s):
            # A late answer would be read as the next request's: drop the link.
            self._close()
            raise TimeoutError
        return self._conn.recv()

    def route(self, text: str | None) -> RouteDecision:
        with self._lock:
            try:
                reply = self._call(("route", text))
            except TimeoutError:
                self.failures += 1
                return RouteDecision(None, None, TIMEOUT)
            except Exception:  # noqa: BLE001 — service down / socket gone
                self._close()
                self.failures += 1
                return RouteDecision(None, None, UNAVAILABLE)
        if reply[0] != "ok":
            self.failures += 1
            return RouteDecision(None, None, UNAVAILABLE)
        return RouteDecision(reply[1], reply[2], reply[3])

    def ping(self) -> tuple | None:
        with self._lock:
            try:
                reply = self._call(("ping",))
            except Exception:  # noqa: BLE001
                self._close()
                return None
        return tuple(reply[1:]) if reply[0] == "ok" else None

    def close(self) -> None:
        with self._lock:
            self._close()


def router_placement(cfg) -> tuple[str, str | None]:
    """Where the encoder goes: ``(torch device, CUDA_VISIBLE_DEVICES to set)``.

    ``router_gpu: N`` names a PHYSICAL GPU (the nvidia-smi index): the service
    process is restricted to that one card, whether or not the model uses it.
    Otherwise ``router_device`` is used as is (``cpu`` hides every GPU from
    the service; ``cuda:N`` counts within the GPUs the engine can see)."""
    if cfg.router_gpu is not None:
        return "cuda:0", str(int(cfg.router_gpu))
    if cfg.router_device == "cpu":
        return "cpu", ""
    return cfg.router_device, None


def apply_router_placement(cfg) -> str:
    """Apply ``router_placement`` to THIS process (call before any CUDA use);
    returns the torch device."""
    device, visible = router_placement(cfg)
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return device


def serve_config(cfg, device: str | None = None) -> None:
    """Run the router service of an ``AdapterRoutingConfig`` in this process."""
    if device is None:
        device = apply_router_placement(cfg)
    router, path = build_router(cfg, device)
    router.encoder.load()
    router.refresh(force=True)
    where = device if cfg.router_gpu is None or device == "cpu" \
        else f"GPU {cfg.router_gpu}"
    dprint(f"[router] serving on {path} | encoder={cfg.encoder} on {where} | "
           f"{len(router.index)} adapters | threshold="
           f"{cfg.fallback_score_threshold} | batch<={cfg.route_batch_max}")
    serve(router, path, max_batch=cfg.route_batch_max,
          batch_wait_s=cfg.route_batch_wait_ms / 1e3)


def build_router(cfg, device: str = "cpu") -> tuple[Router, str]:
    """(Router, socket path) for an ``AdapterRoutingConfig``."""
    from deltaserve.adapter_routing.encoder import SentenceEncoder
    from deltaserve.adapter_routing.pool import ROUTABLE_FILE
    from deltaserve.adapter_routing.stages import index_dir

    directory = index_dir(cfg)
    encoder = SentenceEncoder(cfg.encoder, cfg.encoder_query_prefix,
                              cfg.encoder_max_tokens, device)
    router = Router(encoder, os.path.join(directory, ROUTABLE_FILE),
                    cfg.fallback_score_threshold)
    return router, socket_path(directory)


def main() -> None:
    import argparse

    from deltaserve.config_loader import adapter_routing_config, load_yaml_config

    ap = argparse.ArgumentParser(description="DeltaServe adapter-routing service")
    ap.add_argument("--config", required=True, help="serving YAML")
    ap.add_argument("--device", default=None,
                    help="override the YAML's placement (cpu / cuda:N)")
    args = ap.parse_args()
    cfg = adapter_routing_config(load_yaml_config(args.config, verbose=False))
    if not cfg.enable_adapter_routing:
        raise SystemExit("adapter_routing.enable_adapter_routing is off")
    serve_config(cfg, args.device)


if __name__ == "__main__":
    main()
