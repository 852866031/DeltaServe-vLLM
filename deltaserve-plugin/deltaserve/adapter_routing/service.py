# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The router service: one process that holds the sentence encoder and answers
"which adapter for this text?" for every API server.

- ``Router``: encoder + centroid index file + fallback threshold → a
  ``RouteDecision``. The index file is re-read when it changes (an adapter was
  indexed, disabled, ...), so the service follows the pool without a restart.
- ``serve``: the request loop on a unix socket. One encoder, one request at a
  time — every API server process shares it instead of loading its own copy.
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

    def route(self, text: str | None) -> RouteDecision:
        self.requests += 1
        if not text or not text.strip():
            return RouteDecision(None, None, NO_TEXT)
        self.refresh()
        if not len(self.index):
            return RouteDecision(None, None, NO_ADAPTERS)
        emb = self.encoder.encode([text])[0]
        name, score = self.index.route(emb, self.threshold)
        return RouteDecision(name, score, "" if name else BELOW_THRESHOLD)


def _handle(conn, router: Router, lock: threading.Lock) -> None:
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError, TypeError):   # closed, here or by serve()
                return
            try:
                with lock:                      # one encoder, one request at a time
                    if msg[0] == "route":
                        d = router.route(msg[1])
                        reply = ("ok", d.adapter, d.score, d.reason)
                    elif msg[0] == "ping":
                        reply = ("ok", len(router.index), router.requests)
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
          ready: threading.Event | None = None) -> None:
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
    lock = threading.Lock()
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
            threading.Thread(target=_handle, args=(conn, router, lock),
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
    ap.add_argument("--device", default="cpu", help="encoder device (cpu / cuda:N)")
    args = ap.parse_args()
    cfg = adapter_routing_config(load_yaml_config(args.config, verbose=False))
    if not cfg.enable_adapter_routing:
        raise SystemExit("adapter_routing.enable_adapter_routing is off")
    router, path = build_router(cfg, args.device)
    router.encoder.load()
    router.refresh(force=True)
    dprint(f"[router] serving on {path} | encoder={cfg.encoder} on {args.device} "
           f"| {len(router.index)} adapters | threshold="
           f"{cfg.fallback_score_threshold}")
    serve(router, path)


if __name__ == "__main__":
    main()
