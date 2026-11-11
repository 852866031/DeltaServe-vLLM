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
   memory / adapter snapshot and the model list every 10 s;
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
EXPORT_MAX_MB = 50.0
MAX_STEPS = 1_000_000      # step entries kept (≈ 55 h at 5 / s; bounds RAM under heavy load)
GZ_B64_RATIO = 0.12 * 4 / 3  # measured: gzip ≈ 0.12 × the JSON, base64 × 4/3


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



def _bsearch(items: list[dict], t: float, key: str = "t") -> int:
    """First index with items[i][key] > t (items sorted by key)."""
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid][key] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _compact_window(w: dict) -> dict:
    """The parts of the server's window the page reads (not memory: that is
    kept separately, only when it changes)."""
    inf, ft, tot = w.get("inference") or {}, w.get("finetune") or {}, w.get("totals") or {}
    return {"inference": {k: inf.get(k) for k in
                          ("ttft_slo", "max_tbt_slo", "tbt_slo_attainment")},
            "finetune": {k: ft.get(k) for k in
                         ("backward_gpu_ms_mean", "backward_wall_ms_mean")},
            "totals": {k: tot.get(k) for k in ("first_tokens", "ttft_slo_met", "requests")}}


class Feed:
    """Everything dserve-top has seen from the server, for the live page and
    for scrubbing back (``history_s`` of it): the step stream, the
    per-second series, status snapshots every 2 s, and memory / adapter /
    model-list snapshots when they change. Kept after the server goes away;
    reset when a new server run (another ``t_start``) connects."""

    SNAP_EVERY = 2.0

    def __init__(self, history_s: float = 12 * 3600) -> None:
        self.lock = threading.Lock()
        self.history_s = history_s
        self._reset()
        self.connected = False
        self.error: str | None = None
        self.last_ok = 0.0

    def _reset(self) -> None:
        self.steps: list[dict] = []
        self.series: list[dict] = []
        self.snaps: list[dict] = []          # {t, ft, win}
        self.mems: list[dict] = []           # {t, memory}
        self.models_h: list[dict] = []       # {t, models}
        self.latest: dict = {}
        self.server_t_start = None
        self._mem_key = None
        self._models_key = None
        self.raw_bytes = 0                   # appended JSON bytes (export estimate)

    @property
    def models(self) -> list[dict]:
        return self.models_h[-1]["models"] if self.models_h else []

    @models.setter
    def models(self, value: list[dict]) -> None:
        key = json.dumps(value, sort_keys=True)
        if key != self._models_key:
            self._models_key = key
            self.models_h.append({"t": time.time(), "models": value})

    @property
    def last_seq(self) -> int:
        return self.steps[-1]["s"] if self.steps else 0

    @property
    def last_t(self) -> float:
        return self.series[-1]["t"] if self.series else 0.0

    def merge(self, d: dict) -> None:
        with self.lock:
            if self.server_t_start is not None and d.get("t_start") != self.server_t_start:
                self._reset()                     # a new server run
            self.server_t_start = d.get("t_start")
            now = float(d.get("now") or time.time())
            new = d.get("steps") or []
            if new:
                first = new[0]["s"]
                while self.steps and self.steps[-1]["s"] >= first:
                    self.steps.pop()              # the re-sent open decode run
                self.steps.extend(new)
                self.raw_bytes += 105 * len(new)
            ser = [r for r in (d.get("series") or []) if r["t"] > self.last_t]
            if ser:
                self.series.extend(ser)
                self.raw_bytes += 230 * len(ser)
            win = d.get("window") or {}
            if not self.snaps or now - self.snaps[-1]["t"] >= self.SNAP_EVERY:
                self.snaps.append({"t": now, "ft": d.get("finetuning") or {},
                                   "win": _compact_window(win)})
                self.raw_bytes += 600
            mem = win.get("memory") or {}
            key = json.dumps({k: (v.get("gib"), v.get("lora")) if isinstance(v, dict)
                              and "gib" in v else v for k, v in mem.items()},
                             sort_keys=True, default=str)
            if mem and key != self._mem_key:
                self._mem_key = key
                self.mems.append({"t": now, "memory": mem})
                self.raw_bytes += len(key)
            self.latest = {k: v for k, v in d.items() if k not in ("steps", "series")}
            self.connected, self.error, self.last_ok = True, None, time.time()
            self._prune(now)

    def _prune(self, now: float) -> None:
        cut = now - self.history_s
        for name in ("steps", "series", "snaps"):
            arr = getattr(self, name)
            if arr and arr[0]["t"] < cut - 600:            # amortized
                del arr[:_bsearch(arr, cut)]
        if len(self.steps) > MAX_STEPS + MAX_STEPS // 10:
            del self.steps[:len(self.steps) - MAX_STEPS]
        for arr in (self.mems, self.models_h):             # keep the one in force
            i = _bsearch(arr, cut)
            if i > 1:
                del arr[:i - 1]

    def state(self, steps_since: int, series_since: float, snaps_since: float,
              mems_since: float, models_since: float) -> dict:
        with self.lock:
            i = 0
            if self.steps and steps_since > self.steps[0]["s"]:
                i = max(0, min(len(self.steps), steps_since - self.steps[0]["s"]))
                while i > 0 and self.steps[i - 1]["s"] >= steps_since:
                    i -= 1
            return {**self.latest,
                    "steps": self.steps[i:],
                    "series": self.series[_bsearch(self.series, series_since):],
                    "snaps": self.snaps[_bsearch(self.snaps, snaps_since):],
                    "mems": self.mems[_bsearch(self.mems, mems_since):],
                    "models_h": self.models_h[_bsearch(self.models_h, models_since):],
                    "models": self.models,                 # the current list
                    "connected": self.connected, "error": self.error,
                    "last_ok": self.last_ok,
                    "export_estimate_mb": round((self.raw_bytes * GZ_B64_RATIO + 40000) / 2**20, 2),
                    "history_s": self.history_s}

    def export(self, info: dict, max_mb: float = EXPORT_MAX_MB) -> tuple[bytes, dict]:
        """A single self-contained HTML: the page + this history, gzip'd and
        base64-embedded (the page replays it with the same scrubber). Over
        ``max_mb`` the oldest part is dropped."""
        import base64
        import gzip

        with self.lock:
            data = {"format": 1, "exported_unix_time": time.time(), "top": dict(info),
                    "meta": self.latest.get("meta"), "t_start": self.latest.get("t_start"),
                    "dir": self.latest.get("dir"), "steps": list(self.steps),
                    "series": list(self.series), "snaps": list(self.snaps),
                    "mems": list(self.mems), "models_h": list(self.models_h)}
        page = PAGE.read_text()
        limit = int(max_mb * 2**20)

        def build(d):
            blob = base64.b64encode(gzip.compress(
                json.dumps(d, separators=(",", ":")).encode(), 6)).decode()
            return page.replace("</body>", '<script id="ds-embedded" type="application/'
                                f'gzip-base64">{blob}</script>\n</body>', 1).encode()

        html = build(data)
        trimmed_from = None
        if len(html) > limit and data["series"]:
            t0, t1 = data["series"][0]["t"], data["series"][-1]["t"]
            lo, hi = t0, t1                                   # keep [cut, t1]
            for _ in range(12):
                cut = (lo + hi) / 2
                d = {**data, **{k: [x for x in data[k] if x["t"] >= cut]
                                for k in ("steps", "series", "snaps")},
                     **{k: data[k][max(0, _bsearch(data[k], cut) - 1):]
                        for k in ("mems", "models_h")}}
                h = build(d)
                if len(h) > limit:
                    lo = cut
                else:
                    hi, html, trimmed_from = cut, h, cut
            if trimmed_from is None:                          # nothing fits: newest only
                trimmed_from = t1
                html = build({**data, **{k: data[k][-1:] for k in
                                         ("steps", "series", "snaps", "mems", "models_h")}})
        meta = {"bytes": len(html), "trimmed_from": trimmed_from,
                "span_s": (data["series"][-1]["t"] - data["series"][0]["t"])
                if data["series"] else 0}
        return html, meta


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


def make_handler(feed: Feed, info: dict, srv: DServe | None = None):
    page = PAGE.read_bytes()
    seen: set = set()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet
            pass

        def _send(self, code, body: bytes, ctype: str, headers: dict | None = None,
                  compress: bool = True):
            if (compress and len(body) > 4096
                    and "gzip" in (self.headers.get("Accept-Encoding") or "")):
                import gzip
                body = gzip.compress(body, 5)
                headers = {**(headers or {}), "Content-Encoding": "gzip"}
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path in ("/", "/index.html"):
                peer = self.client_address[0]
                if peer not in seen:
                    seen.add(peer)
                    print(f"dserve-top: a browser opened the dashboard (via {peer})",
                          flush=True)
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/healthz":
                ok = feed.connected
                msg = ("dserve-top OK · server " + ("connected" if ok else
                       f"not connected ({feed.error})") + "\n")
                return self._send(200, msg.encode(), "text/plain; charset=utf-8")
            if u.path == "/api/state":
                q = urllib.parse.parse_qs(u.query)
                g = lambda k, d="0": q.get(k, [d])[0]  # noqa: E731
                body = feed.state(int(g("steps_since")), float(g("series_since")),
                                  float(g("snaps_since")), float(g("mems_since")),
                                  float(g("models_since")))
                body["top"] = info
                return self._send(200, json.dumps(body).encode(), "application/json")
            if u.path == "/api/export":
                q = urllib.parse.parse_qs(u.query)
                mb = min(EXPORT_MAX_MB, float(q.get("max_mb", [EXPORT_MAX_MB])[0]))
                html, meta = feed.export(info, mb)
                model = str((feed.latest.get("meta") or {}).get("served_model_name") or "run")
                name = f"dserve-top-{model}-{time.strftime('%Y%m%d-%H%M%S')}.html"
                print(f"dserve-top: exported {len(html) / 2**20:.1f} MB "
                      f"({meta['span_s'] / 60:.0f} min{' , oldest part trimmed' if meta['trimmed_from'] else ''})",
                      flush=True)
                return self._send(200, html, "text/html; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{name}"'},
                                  compress=False)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/stop":
                # The page's ■ STOP button. The custom header keeps a cross-site form
                # on some other web page from posting here (it cannot set headers
                # without a CORS preflight, which this server never answers).
                if self.headers.get("X-DServe-Stop") != "1" or srv is None:
                    return self._send(403, b'{"error":"forbidden"}', "application/json")
                try:
                    code, r = srv.call("POST", "/shutdown", {})
                except Exception as e:  # noqa: BLE001
                    code, r = 502, {"error": f"server unreachable: {e}"}
                print(f"dserve-top: stop requested from the page (via {self.client_address[0]})"
                      f" → {code} {r.get('status') or r.get('error') or ''}", flush=True)
                return self._send(code, json.dumps(r).encode(), "application/json")
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
        f"       ssh -N -o ExitOnForwardFailure=yes -L {local}:localhost:{port}{extra} {target}",
        "",
        "     It prints nothing while the tunnel is up (-N: tunnel only, no shell); leave",
        "     it running. It exits with an error if the local port is taken. Add -v to",
        "     watch it connect. If you normally",
        "     go through a jump host, add  -J <jump-host>  (or reuse your ~/.ssh/config",
        f"     alias:  ssh -N -L {local}:localhost:{port} <alias>).",
        "",
        f"  2. In your laptop's browser open:   http://localhost:{local}",
        "     (dserve-top prints 'a browser opened the dashboard' here when it works.)",
        f"     Check from the laptop without a browser:  curl -s http://localhost:{local}/healthz",
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
    ap.add_argument("--history-hours", type=float, default=12.0,
                    help="how far back the page can scrub (kept in memory; default 12)")
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
    feed = Feed(history_s=a.history_hours * 3600)
    stop = threading.Event()
    info = {"server": a.server, "recording_dir": None, "started_by_top": False}
    link = Link(srv, a, info)
    httpd = bind(a.bind, a.port, make_handler(feed, info, srv))
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
