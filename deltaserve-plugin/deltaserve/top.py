# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""dserve-top: a live web dashboard for a running DeltaServe server.

    dserve-top                                   # server at 127.0.0.1:8000
    dserve-top --server http://127.0.0.1:8001 --port 8095

Run it on the GPU machine, next to the server. It

1. waits for the server if it is not up yet (start dserve-top first if you
   like), then asks it to record telemetry (POST /telemetry/start) unless it
   already does (telemetry.metrics_dir in its config) — into ``--dir``,
   default ``./dserve-top/<time>``; again after a server restart; on exit it
   stops a recording it started;
2. polls the live feed (GET /telemetry/data) ~4×/s into its own buffers — one
   poller however many browser tabs are open — and refreshes the per-GPU
   memory snapshot every 30 s;
3. serves the dashboard page on ``--port`` (127.0.0.1 only) and prints how to
   open it through an SSH tunnel.

Standard library only (no vLLM import): starts instantly, safe to run anywhere.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("top_page.html")
MAX_STEPS = 20000
MAX_SERIES = 3600


class DServe:
    """HTTP client for the dserve server's telemetry routes."""

    def __init__(self, base: str, timeout: float = 5.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}


class Feed:
    """The poller's merged view of the server's live feed."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.steps: list[dict] = []
        self.series: list[dict] = []
        self.latest: dict = {}
        self.connected = False
        self.error: str | None = None
        self.last_ok = 0.0
        self.server_t_start = None
        self.models: list[dict] = []        # GET /v1/models (base + adapters)

    @property
    def last_seq(self) -> int:
        return self.steps[-1]["s"] if self.steps else 0

    @property
    def last_t(self) -> float:
        return self.series[-1]["t"] if self.series else 0.0

    def merge(self, d: dict) -> None:
        with self.lock:
            if self.server_t_start is not None and d.get("t_start") != self.server_t_start:
                self.steps, self.series = [], []          # server / recording restarted
            self.server_t_start = d.get("t_start")
            new = d.get("steps") or []
            if new:
                first = new[0]["s"]
                while self.steps and self.steps[-1]["s"] >= first:
                    self.steps.pop()                      # the re-sent open decode run
                self.steps.extend(new)
                if len(self.steps) > 2 * MAX_STEPS:
                    del self.steps[:len(self.steps) - MAX_STEPS]
            ser = d.get("series") or []
            if ser:
                self.series.extend(r for r in ser if r["t"] > self.last_t)
                if len(self.series) > 2 * MAX_SERIES:
                    del self.series[:len(self.series) - MAX_SERIES]
            self.latest = {k: v for k, v in d.items() if k not in ("steps", "series")}
            self.connected, self.error, self.last_ok = True, None, time.time()

    def state(self, steps_since: int, series_since: float, limit: int = 3000) -> dict:
        with self.lock:
            st = [e for e in self.steps[-limit:] if e["s"] >= steps_since]
            se = [r for r in self.series if r["t"] > series_since]
            return {**self.latest, "steps": st, "series": se, "models": self.models,
                    "connected": self.connected, "error": self.error,
                    "last_ok": self.last_ok}


class Link:
    """dserve-top's connection to the server: waits for it, starts the
    recording when the server is not recording (again after a restart), and
    remembers whether it started it (to stop it on exit)."""

    def __init__(self, srv: DServe, a, info: dict) -> None:
        self.srv, self.a, self.info = srv, a, info
        self.live = False
        self.started = False
        self._said = None

    def _say(self, msg: str) -> None:
        if msg != self._said:                       # print each state change once
            self._said = msg
            print(f"dserve-top: {msg}", flush=True)

    def connect(self, feed: "Feed") -> bool:
        base = self.srv.base
        try:
            code, d = self.srv.call("GET", "/telemetry/data?steps_since="
                                    "9223372036854775807&series_since=1e18")
        except Exception:  # noqa: BLE001 — not up yet
            feed.connected, feed.error = False, f"waiting for the DeltaServe server at {base} …"
            self._say(f"waiting for the DeltaServe server at {base} …")
            return False
        if code == 404:
            feed.connected = False
            feed.error = (f"{base} is up but has no /telemetry routes — finetuning off "
                          "(finetune_param.enable_finetuning) or an old plugin")
            self._say(feed.error)
            return False
        if code != 200:
            feed.connected, feed.error = False, f"server answered HTTP {code}; retrying"
            self._say(feed.error)
            return False
        if d.get("active"):
            self.started = self.started and bool(d.get("runtime"))
            self.info.update(recording_dir=d.get("dir"), started_by_top=self.started)
            self._say(f"connected; the server records telemetry → {d.get('dir')}")
            self.live = True
            return True
        rec_dir = os.path.abspath(self.a.dir or os.path.join(
            "dserve-top", time.strftime("%Y%m%d-%H%M%S")))
        code, r = self.srv.call("POST", "/telemetry/start",
                                {"dir": rec_dir, "interval_s": self.a.interval})
        if code != 200 or not r.get("active"):
            feed.connected, feed.error = False, f"could not start recording: {r}"
            self._say(feed.error)
            return False
        self.started = bool(r.get("started"))
        self.info.update(recording_dir=r.get("dir", rec_dir), started_by_top=self.started)
        self._say(f"connected; started telemetry recording → {r.get('dir', rec_dir)}")
        self.live = True
        return True


def poll_loop(link: Link, feed: Feed, stop: threading.Event, every: float) -> None:
    srv = link.srv
    last_mem = 0.0
    while not stop.is_set():
        if not link.live:
            if not link.connect(feed):
                stop.wait(2.0)
                continue
            last_mem = 0.0
        try:
            code, d = srv.call(
                "GET", f"/telemetry/data?steps_since={feed.last_seq}"
                       f"&series_since={feed.last_t}")
            if code == 200 and d.get("active"):
                feed.merge(d)
            else:                                   # stopped / restarted: reconnect
                link.live = False
                continue
            if time.time() - last_mem > 10:
                # adapters come and go (on-demand loads, LRU): refresh the
                # per-GPU memory / adapter snapshot and the API's model list
                last_mem = time.time()
                srv.call("POST", "/telemetry/memory", {})
                c, m = srv.call("GET", "/v1/models")
                if c == 200:
                    with feed.lock:
                        feed.models = [{"id": x.get("id"), "parent": x.get("parent"),
                                        "root": x.get("root")}
                                       for x in m.get("data", [])]
        except Exception:  # noqa: BLE001 — server gone: wait for it again
            link.live = False
            feed.connected = False
            feed.error = f"lost the server at {srv.base}; waiting for it …"
            link._say(feed.error)
            stop.wait(1.0)
            continue
        stop.wait(every)


def make_handler(feed: Feed, info: dict):
    page = PAGE.read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/api/state":
                q = urllib.parse.parse_qs(u.query)
                body = feed.state(int(q.get("steps_since", ["0"])[0]),
                                  float(q.get("series_since", ["0"])[0]))
                body["top"] = info
                return self._send(200, json.dumps(body).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

    return H


def ssh_guide(port: int) -> str:
    user = getpass.getuser()
    host = socket.gethostname()
    target, extra = f"{user}@{host}", ""
    conn = os.environ.get("SSH_CONNECTION", "").split()
    if len(conn) == 4:                      # client_ip client_port server_ip server_port
        target = f"{user}@{conn[2]}"
        if conn[3] != "22":
            extra = f" -p {conn[3]}"
    local = port
    alt = port + 10000 if port + 10000 <= 65535 else port - 10000
    lines = [
        "",
        "  dserve-top — open the dashboard from your laptop over SSH",
        f"  (the page is served on this machine, 127.0.0.1:{port} only)",
        "",
        "  1. On your LAPTOP (a new local terminal, not this SSH session), run:",
        "",
        f"       ssh -N -L {local}:localhost:{port}{extra} {target}",
        "",
        "     -N opens only the tunnel (no shell); leave it running. If you normally",
        "     go through a jump host, add  -J <jump-host>  (or reuse your ~/.ssh/config",
        f"     alias:  ssh -N -L {local}:localhost:{port} <alias>).",
        "",
        f"  2. In your laptop's browser open:   http://localhost:{local}",
        "",
        f"  Port {local} already used on your laptop? Pick another local port, e.g.",
        f"     ssh -N -L {alt}:localhost:{port}{extra} {target}   →  http://localhost:{alt}",
        "  VS Code Remote-SSH: Ports panel → Forward a Port → "
        f"{port} → open the Local Address.",
        f"  On this machine's own desktop: just open http://127.0.0.1:{port}",
        "",
    ]
    return "\n".join(lines)


def bind(bind_host: str, port: int, handler) -> ThreadingHTTPServer:
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer((bind_host, p), handler)
        except OSError:
            continue
    raise SystemExit(f"dserve-top: no free port in {port}..{port + 19}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dserve-top", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000",
                    help="the DeltaServe server (default http://127.0.0.1:8000)")
    ap.add_argument("--port", type=int, default=8090,
                    help="dashboard port on this machine (default 8090, next free)")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="address to serve on (default 127.0.0.1: reach it via SSH)")
    ap.add_argument("--dir", default=None,
                    help="where the server records when dserve-top starts the "
                         "recording (default ./dserve-top/<time>)")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="summary window seconds for a recording started here")
    ap.add_argument("--keep-recording", action="store_true",
                    help="leave a recording dserve-top started running on exit")
    ap.add_argument("--poll", type=float, default=0.25, help="seconds between polls")
    ap.add_argument("--no-wait", action="store_true",
                    help="exit if the server is not reachable now (default: serve "
                         "the page and wait for it)")
    a = ap.parse_args(argv)

    srv = DServe(a.server)
    if a.no_wait:
        try:
            srv.call("GET", "/health")
        except Exception as e:  # noqa: BLE001
            print(f"dserve-top: cannot reach the server at {a.server} ({e}).")
            return 1
    feed = Feed()
    stop = threading.Event()
    info = {"server": a.server, "recording_dir": None, "started_by_top": False}
    link = Link(srv, a, info)
    httpd = bind(a.bind, a.port, make_handler(feed, info))
    port = httpd.server_address[1]
    print(ssh_guide(port), flush=True)
    print(f"dserve-top: dashboard served on http://{a.bind}:{port} · watching the "
          f"server at {a.server} (the page works before it is up)")
    print("dserve-top: Ctrl-C to quit (stops a recording dserve-top started"
          + (", unless --keep-recording)" if not a.keep_recording else " — kept)"),
          flush=True)
    th = threading.Thread(target=poll_loop, args=(link, feed, stop, a.poll), daemon=True)
    th.start()

    def shutdown(*_):
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.server_close()
        if link.started and not a.keep_recording:
            try:
                srv.call("POST", "/telemetry/stop", {})
                print(f"\ndserve-top: recording stopped (kept in {info['recording_dir']})")
            except Exception as e:  # noqa: BLE001
                print(f"\ndserve-top: could not stop the recording: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
