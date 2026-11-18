#!/usr/bin/env python
"""Live end-to-end run of adapter routing on one GPU (Qwen3-0.6B, TP=1).

Launches a real ``vllm serve`` with finetuning + adapter routing on, and a
finetuning job with stages [init, training, indexing, ready] on a small
translation corpus. Then, over HTTP:

  * before the job is ready, ``model: "auto"`` is answered by the base model
    (fallback header);
  * the job runs through its stages and reaches 'ready' (GET /ft_jobs);
  * the trained adapter is on disk and differs from the one it started from;
  * ``model: "auto"`` with a translation prompt is now routed to the job's
    adapter (response headers + the response's ``model`` field);
  * a request naming the base model is untouched.

    CUDA_VISIBLE_DEVICES=1 python tests/test_adapter_routing_live.py
    ... --api-servers 2      # several API server processes
    ... --router-device cuda:0      # or --router-gpu <physical index>

Needs a GPU; the server log goes to <out dir>/server.log (printed on failure).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# --family picks the model (set in main, before anything reads these):
#   qwen3-0.6b  one small GPU's worth, TP=1 by default (the default)
#   qwen3-14b   the adapter-routing production preset; run it with --tp 2
_FAMILIES = {
    "qwen3-0.6b": "serving_config_finetuning_qwen3_0.6b.yaml",
    "qwen3-14b": "serving_config_finetuning_qwen3_14b_tp2_adapter_routing.yaml",
}
_BASE = "qwen3-0.6b"
_PRESET = _REPO / "configs" / _FAMILIES[_BASE]

_passed = 0
_failed = 0
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  {_GREEN}[PASS]{_RESET} {name}", flush=True)
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}", flush=True)
    return bool(cond)


SENTENCES = [
    "Guten Morgen, wie geht es dir?", "Das Wetter ist heute schön.",
    "Ich möchte einen Kaffee bestellen.", "Wo ist der Bahnhof?",
    "Wir fahren morgen nach Berlin.", "Das Buch liegt auf dem Tisch.",
    "Kannst du mir bitte helfen?", "Ich habe meinen Schlüssel verloren.",
    "Der Zug kommt um acht Uhr an.", "Sie arbeitet in einem Krankenhaus.",
]


def http(port, path, body=None, timeout=120, method=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method=method or ("GET" if body is None else "POST"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, {k.lower(): v for k, v in r.headers.items()}, \
                (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = raw.decode(errors="replace")
        return e.code, {k.lower(): v for k, v in e.headers.items()}, payload
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return None, {}, str(e)


def stop_tree(server: subprocess.Popen) -> None:
    """Stop the launcher AND everything under it. The launcher starts
    ``vllm serve`` in its own session, so killing the launcher's process group
    would leave the server (and its GPU memory) behind."""
    import psutil

    try:
        root = psutil.Process(server.pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        procs = []
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=30)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def write_config(out: Path, args) -> Path:
    import yaml

    cfg = yaml.safe_load(_PRESET.read_text())
    ft = cfg["finetune"]
    src_adapter = (_PRESET.parent / ft.pop("finetuning_lora_path")).resolve()
    ft.pop("data_path")
    ft.pop("num_epochs")
    ft["start_on_launch"] = True
    cfg["slo"]["profile_on_launch"] = False
    corpus = out / "translate.txt"
    corpus.write_text("".join(
        f"Translate to English: {s} (example {i})\n"
        for i in range(args.repeats) for s in SENTENCES))
    cfg["ft_job"] = {
        "stages": ["init", "training", "indexing", "ready"],
        "auto_routing": True,
        "adapter_path": str(src_adapter),
        "data_path": str(corpus),
        "num_epochs": args.epochs,
    }
    cfg["adapter_routing"] = {
        "enable_adapter_routing": True,
        "index_path": str(out / "index"),
        "router_device": args.router_device,
        "router_gpu": args.router_gpu,
        "route_keep_percent": 100,      # one adapter: this run tests the pipeline
        "route_timeout_ms": 5000,
        "indexing_cpus": args.indexing_cpus,
    }
    cfg["adapters"]["lora_path_0"] = str(
        (_PRESET.parent / cfg["adapters"]["lora_path_0"]).resolve())
    cfg["lora"]["max_loras"] = max(4, int(cfg["lora"].get("max_loras") or 0))
    cfg["server"]["api_server_count"] = args.api_servers
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, src_adapter


def _vary(templates, items):
    return [t.format(x) for x in items for t in templates]


TOPICS = {
    "ext-translate": _vary(
        ["Translate to English: {}", "Translate this German sentence into English: {}",
         "German to English: {}", "What does this mean in English? {}"], SENTENCES),
    "ext-sql": _vary(
        ["Write a SQL query that {}.", "Give me the SQL to {}.",
         "SQL: {}", "How do I {} in SQL?"],
        ["list the ten newest orders", "count users per country",
         "join customers and invoices", "find duplicate email addresses",
         "return the average order value per month", "delete inactive sessions",
         "select products that were never sold", "rank employees by salary",
         "update the stock after a sale", "group page views by day"]),
    "ext-math": _vary(
        ["Solve for x: {}", "Find x if {}", "Solve the equation {}",
         "What is x when {}?"],
        ["3x + 5 = 20", "2x - 7 = 11", "x / 4 + 1 = 6", "5x = 45", "9 - x = 2",
         "4x + 4 = 28", "x squared = 49", "7x - 3 = 4x + 9", "10 / x = 2",
         "6x + 1 = 37"]),
    "ext-poem": _vary(
        ["Write a short poem about {}.", "Compose a poem on {}.",
         "A four-line poem about {}, please.", "Write some verses about {}."],
        ["the sea at night", "autumn leaves", "a quiet mountain", "city rain",
         "the first snow", "an old lighthouse", "spring flowers", "a lost letter",
         "the morning sun", "a sleeping cat"]),
}


def external_main(args, out: Path) -> None:
    """Finetuning OFF: existing adapters are indexed and routed, no training."""
    import yaml

    cfg = yaml.safe_load(_PRESET.read_text())
    toy = (_PRESET.parent / cfg["adapters"]["lora_path_0"]).resolve()
    toy_ft = (_PRESET.parent / cfg["finetune"]["finetuning_lora_path"]).resolve()
    cfg["finetune"] = {"enable_finetuning": False}
    for sec in ("throttles", "slo", "debug"):
        cfg.pop(sec, None)
    data = {}
    for name, lines in TOPICS.items():
        data[name] = out / f"{name}.txt"
        data[name].write_text("".join(f"{t}\n" for t in lines))
    cfg["adapter_routing"] = {
        "enable_adapter_routing": True, "index_path": str(out / "index"),
        "router_device": args.router_device, "router_gpu": args.router_gpu,
        "route_timeout_ms": 5000, "route_decision_log": True,
        # 40 templated lines per adapter: genuine requests sit a few confidence
        # units below every training sample, unrelated ones tens of units.
        "route_confidence_slack": 4.0,
        "indexing_cpus": args.indexing_cpus,
        "adapters": [
            {"name": "ext-math", "path": str(toy),
             "data_path": str(data["ext-math"])},
            {"name": "ext-translate", "path": str(toy),
             "data_path": str(data["ext-translate"])},
            {"name": "ext-sql", "path": str(toy_ft),
             "data_path": str(data["ext-sql"])}],
    }
    cfg["adapters"]["lora_path_0"] = str(toy)
    cfg["lora"]["max_loras"] = max(4, int(cfg["lora"].get("max_loras") or 0))
    cfg["server"]["api_server_count"] = args.api_servers
    config = out / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_path = out / "server.log"
    print(f"=== existing adapters, no training, live (GPU "
          f"{os.environ['CUDA_VISIBLE_DEVICES']}, TP={args.tp}, "
          f"{args.api_servers} API server(s)) ===\n    out dir: {out}")
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false",
               VLLM_USE_V2_MODEL_RUNNER="0")   # v2 + LoRA + CUDA graphs fails upstream
    env.setdefault("HF_HOME", "/home/jiaxuan/.cache/huggingface")
    log = open(log_path, "w")
    server = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--family", _BASE, "--config", str(config), "--tp", str(args.tp),
         "--port", str(args.port)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    port, ok = args.port, False

    def routable(names, timeout=300):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            status, _, st = http(port, "/adapters")
            if status == 200:
                idx = {a["name"] for a in st["adapters"] if a["indexed"]}
                if set(names) <= idx:
                    return st, time.monotonic() - t0
            time.sleep(0.5)
        return None, timeout

    def auto(prompt, want):
        last = None
        for _ in range(20):      # the router re-reads its index within ~1 s
            status, headers, body = http(port, "/v1/completions", {
                "model": "auto", "prompt": prompt, "max_tokens": 4,
                "temperature": 0})
            last = (status, headers.get("x-deltaserve-adapter"),
                    headers.get("x-deltaserve-routing-score"),
                    body.get("model") if isinstance(body, dict) else body,
                    headers.get("x-deltaserve-routing-confidence"),
                    headers.get("x-deltaserve-routing-candidates"))
            if status == 200 and last[1] == want:
                break
            time.sleep(0.5)
        print(f"    auto {prompt[:42]!r:46} → {last}")
        return last[0] == 200 and last[1] == want and last[3] == want

    try:
        t0 = time.monotonic()
        while True:
            status, _, _ = http(port, "/health", timeout=5)
            if status == 200:
                break
            if server.poll() is not None or time.monotonic() - t0 > args.timeout:
                check("server became healthy", False)
                return
            time.sleep(2)
        check(f"server healthy after {time.monotonic() - t0:.0f} s "
              f"(finetuning off)", True)
        status, _, jobs = http(port, "/ft_jobs")
        check("no finetuning routes on this server (GET /ft_jobs → 404)",
              status == 404)

        st, took = routable(["ext-translate", "ext-sql", "ext-math"])
        if not check(f"the listed adapters are indexed and in the pool "
                     f"({took:.1f} s after the server was healthy)", st is not None):
            print(f"    /adapters: {http(port, '/adapters')[2]}")
            return
        by = {a["name"]: a for a in st["adapters"]}
        print(f"    pool: { {n: (a['tier'], a['lora_int_id']) for n, a in by.items()} }")
        check("registered without loading: both still on the disk tier",
              all(a["tier"] == "disk" for a in by.values()))
        fin = {j["name"]: j for j in st["jobs"]["finished"]}
        check("their jobs ran [init, indexing, admission, ready] with no error; the pool "
              "uses the adapters where they are (no copy)",
              all(list(fin[n]["stages"]) == ["init", "indexing", "admission", "ready"]
                  and fin[n]["error"] is None
                  and fin[n]["saved_path"] == by[n]["path"] for n in by))

        check("a translation request is routed to ext-translate",
              auto("Translate to English: Wir gehen heute ins Kino.", "ext-translate"))
        check("a SQL request is routed to ext-sql",
              auto("Write a SQL query that lists customers without orders.",
                   "ext-sql"))

        status, _, reply = http(port, "/adapters", {
            "name": "ext-poem", "path": str(toy),
            "data_path": str(data["ext-poem"])})
        check("POST /adapters accepts a third adapter while serving (202)",
              status == 202 and reply["job"]["name"] == "ext-poem")
        st, took = routable(["ext-poem"])
        check(f"it becomes routable without a restart ({took:.1f} s)", st is not None)
        check("a poem request is now routed to ext-poem",
              auto("Write a short poem about a river in winter.", "ext-poem"))
        check("the first two still route correctly",
              auto("Translate to English: Das ist mein Bruder.", "ext-translate")
              and auto("Write a SQL query that sums sales by region.", "ext-sql"))

        _, headers, body = http(port, "/v1/completions", {
            "model": "auto", "prompt": "Translate to English: Es regnet heute.",
            "max_tokens": 2, "temperature": 0})
        check("with four adapters the response carries a pool-relative confidence",
              "x-deltaserve-routing-confidence" in headers)
        outs = []
        for q in ("What is the capital of Australia?",
                  "Tell me about the history of the Roman empire.",
                  "Is it going to rain tomorrow?", "Who painted the Mona Lisa?"):
            _, h, b = http(port, "/v1/completions", {
                "model": "auto", "prompt": q, "max_tokens": 2, "temperature": 0})
            outs.append((h.get("x-deltaserve-routing"),
                         h.get("x-deltaserve-routing-reason"),
                         h.get("x-deltaserve-routing-confidence"),
                         h.get("x-deltaserve-adapter"), b.get("model")))
            print(f"    auto {q[:42]!r:46} → {outs[-1]}")
        check("unrelated questions now fall back to the base model as "
              f"low-confidence ({sum(o[1] == 'low-confidence' for o in outs)}/4)",
              sum(o[1] == "low-confidence" and o[4] == _BASE
                  for o in outs) >= 3)
        dlog = out / "index" / "decisions.jsonl"
        check("decisions are logged for calibration",
              dlog.is_file() and len(dlog.read_text().splitlines()) >= 8)

        _, _, st = http(port, "/adapters")
        used = {a["name"]: a["hits"] for a in st["adapters"]}
        print(f"    pool: active {st['active']} retired {st['retired']} "
              f"version {st['version']} hits {used}")
        check("requests are counted per adapter with finetuning OFF",
              used.get("ext-translate", 0) >= 2 and used.get("ext-sql", 0) >= 1)
        fin = {j["name"]: j for j in st["jobs"]["finished"]}
        rep = fin["ext-poem"]["admission"]
        print(f"    admission of ext-poem: { {k: rep[k] for k in ('level', 'self_recall', 'nearest', 'nearest_cosine', 'stolen_max')} }")
        check("the adapter added while serving has an admission report",
              rep["level"] == "active" and rep["self_recall"] is not None)
        status, _, r = http(port, "/adapters/ext-math/retire", {})
        ok_retire = status == 200 and r["adapter"]["level"] == "retired"
        time.sleep(1.5)
        s2, h2, b2 = http(port, "/v1/completions", {
            "model": "auto", "prompt": "Solve for x: 8x - 4 = 20", "max_tokens": 2,
            "temperature": 0})
        s3, h3, b3 = http(port, "/v1/completions", {
            "model": "ext-math", "prompt": "Solve for x: 8x - 4 = 20",
            "max_tokens": 2, "temperature": 0})
        print(f"    after retiring ext-math: auto → {h2.get('x-deltaserve-adapter')} "
              f"({h2.get('x-deltaserve-routing-reason')}), by name → {s3} "
              f"{b3.get('model') if isinstance(b3, dict) else b3}")
        check("POST /adapters/ext-math/retire: no longer auto-selected, still "
              "servable by name",
              ok_retire and h2.get("x-deltaserve-adapter") != "ext-math"
              and s3 == 200 and b3.get("model") == "ext-math")
        status, _, r = http(port, "/adapters/ext-math/reactivate", {})
        time.sleep(1.5)
        check("POST /adapters/ext-math/reactivate: back through admission and "
              "selectable again",
              status == 200 and r["admission"]["level"] == "active"
              and auto("Solve for x: 8x - 4 = 20", "ext-math"))
        status, _, r = http(port, "/adapters/ext-sql/pin", {})
        check("POST /adapters/ext-sql/pin", status == 200 and r["adapter"]["pinned"])

        status, _, reply = http(port, "/adapters", {"name": "bad",
                                                    "path": str(out / "nope")})
        time.sleep(1.5)
        _, _, st = http(port, "/adapters")
        failed = [j for j in st["jobs"]["finished"] if j["name"] == "bad"]
        check("a bad adapter path fails its own job with a clear error; the "
              "server keeps serving",
              bool(failed) and failed[0]["stage"] == "failed"
              and "no adapter weights" in failed[0]["error"]
              and auto("Translate to English: Guten Abend.", "ext-translate"))
        # ---- indexing borrows the GPU encoder when inference is quiet ----
        def big(name, n):
            p = out / f"{name}.txt"
            items = [f"item number {i} of the {name} collection, described at "
                     f"some length so the encoder has work to do" for i in range(n)]
            p.write_text("".join(f"Describe {x}.\n" for x in items))
            return str(p)

        def job_detail(name, timeout=240):
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                _, _, st = http(port, "/adapters")
                for j in st["jobs"]["finished"]:
                    if j["name"] == name:
                        return j, time.monotonic() - t0
                time.sleep(0.25)
            return None, timeout

        if args.router_gpu is not None or args.router_device != "cpu":
            time.sleep(4.0)                       # inference quiet
            http(port, "/adapters", {"name": "ext-quiet", "path": str(toy),
                                     "data_path": big("quiet", 1200)})
            j, took = job_detail("ext-quiet")
            d1 = (j or {}).get("detail", {}).get("indexing", {})
            print(f"    quiet:  indexed 1200 samples in {took:.1f} s  {d1}")
            check("no inference → indexing ran on the GPU encoder",
                  j is not None and j["error"] is None
                  and d1.get("chunks", {}).get("gpu", 0) > 0
                  and d1["chunks"]["gpu"] >= 5 * max(1, d1["chunks"].get("cpu", 0)))

            stop_load = threading.Event()
            sent = []

            def load():
                while not stop_load.is_set():
                    s9, _, _ = http(port, "/v1/completions", {
                        "model": _BASE, "prompt": "Hello there",
                        "max_tokens": 2, "temperature": 0}, timeout=60)
                    sent.append(s9)
                    time.sleep(0.2)

            th = threading.Thread(target=load, daemon=True)
            th.start()
            time.sleep(1.0)
            http(port, "/adapters", {"name": "ext-busy", "path": str(toy),
                                     "data_path": big("busy", 300)})
            j, took = job_detail("ext-busy")
            stop_load.set()
            th.join(5)
            d2 = (j or {}).get("detail", {}).get("indexing", {})
            print(f"    busy:   indexed 300 samples in {took:.1f} s  {d2}  "
                  f"({len(sent)} inference requests meanwhile, "
                  f"{sum(x == 200 for x in sent)} ok)")
            check("inference active → indexing stayed on the CPU",
                  j is not None and j["error"] is None
                  and d2.get("chunks", {}).get("cpu", 0) > 0
                  and d2["chunks"].get("gpu", 0) <= d2["chunks"]["cpu"] // 4
                  and sent and all(x == 200 for x in sent))

        # ---- /v1/responses, and an adapter removed while cached ----
        s8, h8, _ = http(port, "/v1/responses", {
            "model": "auto", "input": "Translate to English: Guten Morgen.",
            "max_output_tokens": 4})
        print(f"    /v1/responses auto → status {s8}, "
              f"adapter {h8.get('x-deltaserve-adapter')}")
        check("an auto request on /v1/responses is routed as well",
              h8.get("x-deltaserve-routing") == "routed"
              and h8.get("x-deltaserve-adapter") == "ext-translate")
        byname = [http(port, "/v1/completions", {
            "model": "ext-poem", "prompt": "A poem:", "max_tokens": 2,
            "temperature": 0})[0] for _ in range(6)]
        s7, _, r7 = http(port, "/adapters/ext-poem", None, method="DELETE")
        time.sleep(1.5)
        gone = [http(port, "/v1/completions", {
            "model": "ext-poem", "prompt": "A poem:", "max_tokens": 2,
            "temperature": 0})[0] for _ in range(6)]
        print(f"    ext-poem by name: before removal {byname}, after {gone}")
        check("DELETE /adapters/ext-poem: every API server stops serving it by "
              "name although it had it cached",
              s7 == 200 and all(x == 200 for x in byname)
              and all(x == 404 for x in gone))

        status, headers, body = http(port, "/v1/completions", {
            "model": _BASE, "prompt": "Hello", "max_tokens": 2,
            "temperature": 0})
        check("a request naming the base model is untouched",
              status == 200 and body.get("model") == _BASE
              and not any(k.startswith("x-deltaserve") for k in headers))
        ok = _failed == 0
    finally:
        stop_tree(server)
        log.close()
        if not ok:
            print(f"\n--- last lines of {log_path} ---")
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]))


def second_job(args, out, port, first, first_saved, src_adapter) -> None:
    """A second finetuning job, submitted while serving, runs after the first."""
    from safetensors.torch import load_file

    print("  -- a second job, submitted over HTTP --")
    data = out / "sql.txt"
    data.write_text("".join(f"{t}\n" for t in TOPICS["ext-sql"]))
    status, _, r = http(port, "/ft_jobs", {
        "name": "job-sql", "adapter_path": str(src_adapter), "data_path": str(data),
        "stages": ["init", "training", "indexing", "admission", "ready"],
        "auto_routing": True, "num_epochs": 2, "learning_rate": 2e-5})
    check("POST /ft_jobs accepts it (202)",
          status == 202 and r["job"]["name"] == "job-sql")
    status, _, r = http(port, "/ft_jobs", {
        "name": "job-bad", "data_path": str(data),
        # an adapter of the OTHER model size
        "adapter_path": str(_REPO / "adapters" / (
            "qwen3-0.6b-toy-lora-ft" if _BASE == "qwen3-14b"
            else "qwen3-14b-toy-lora-ft"))})
    check("a job whose adapter has another shape is accepted into the queue",
          status == 202)
    status, _, r = http(port, "/ft_jobs", {"adapter_path": "/nope",
                                           "data_path": str(data)})
    check("a job with no adapter weights is refused at submission (400)",
          status == 400 and "adapter" in r.get("error", ""))

    seen, t0, served, final = [], time.monotonic(), 0, None
    while time.monotonic() - t0 < args.timeout:
        status, _, jobs = http(port, "/ft_jobs")
        cur = jobs.get("current")
        label = None if cur is None else f"{cur['name']}:{cur['stage']}"
        if label and (not seen or seen[-1] != label):
            seen.append(label)
            print(f"    [{time.monotonic() - t0:5.1f}s] {label}", flush=True)
        s1, _, b1 = http(port, "/v1/completions", {
            "model": _BASE, "prompt": "Hello", "max_tokens": 1,
            "temperature": 0}, timeout=60)
        served += s1 == 200
        if cur is None and not jobs.get("queued"):
            final = {j["name"]: j for j in jobs["finished"]}
            break
        time.sleep(0.5)
    check(f"inference kept being served across the job switch ({served} requests)",
          served >= 3 and final is not None)
    if final is None:
        return
    j2, bad = final.get("job-sql"), final.get("job-bad")
    print(f"    job-sql: { {k: j2[k] for k in ('stage', 'error', 'saved_path')} }")
    print(f"    job-bad: { {k: bad[k] for k in ('stage', 'error')} }")
    check("the second job ran every stage to done",
          j2 is not None and j2["error"] is None
          and all(v == "done" for v in j2["stages"].values()))
    check("the mis-shaped job failed at init with the reason; the server and the "
          "queue went on",
          bad is not None and bad["stage"] == "failed" and "init" in bad["error"]
          and ("shape" in bad["error"] or "LoRA tensors" in bad["error"]
               or "refused" in bad["error"]))
    a2 = load_file(os.path.join(j2["saved_path"], "adapter_model.safetensors"))
    a1 = load_file(os.path.join(first_saved, "adapter_model.safetensors"))
    a0 = load_file(str(src_adapter / "adapter_model.safetensors"))
    d20 = max(float((a2[k].float() - a0[k].float()).abs().max()) for k in a0)
    d21 = max(float((a2[k].float() - a1[k].float()).abs().max()) for k in a0)
    d10 = max(float((a1[k].float() - a0[k].float()).abs().max()) for k in a0)
    print(f"    max |job2 - initial| {d20:.3e}  |job2 - job1| {d21:.3e}  "
          f"|job1 - initial| {d10:.3e}")
    check("the second adapter was trained from the initial weights (it differs "
          "from them and from the first job's result)", d20 > 0 and d21 > 0)
    check("the first job's saved adapter was not touched by the second",
          first_saved != j2["saved_path"] and d10 > 0)
    _, _, st = http(port, "/adapters")
    names = sorted(a["name"] for a in st["adapters"] if a["level"] == "active")
    check(f"both trained adapters are active in the pool {names}",
          names == sorted([first, "job-sql"]))
    for prompt, want in (("Translate to English: Wir gehen heute ins Kino.", first),
                         ("Write a SQL query that lists customers without orders.",
                          "job-sql")):
        got = None
        for _ in range(20):
            s3, h3, b3 = http(port, "/v1/completions", {
                "model": "auto", "prompt": prompt, "max_tokens": 4,
                "temperature": 0})
            got = h3.get("x-deltaserve-adapter")
            if got == want:
                break
            time.sleep(0.5)
        check(f"auto request → {want} ({prompt[:32]!r}…)",
              got == want and s3 == 200 and b3.get("model") == want)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--api-servers", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--family", choices=sorted(_FAMILIES), default="qwen3-0.6b")
    ap.add_argument("--router-device", default="auto")
    ap.add_argument("--router-gpu", type=int, default=None,
                    help="physical GPU index for the router's encoder")
    ap.add_argument("--indexing-cpus", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=4, help="corpus = 10 x this")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--second-job", action="store_true",
                    help="then submit a second finetuning job over HTTP")
    ap.add_argument("--external", action="store_true",
                    help="finetuning off: index and route existing adapters")
    args = ap.parse_args()
    global _BASE, _PRESET
    _BASE, _PRESET = args.family, _REPO / "configs" / _FAMILIES[args.family]
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES to the GPU this may use")

    out = Path(args.out or tempfile.mkdtemp(prefix="ds-routing-live-"))
    out.mkdir(parents=True, exist_ok=True)
    if args.external:
        external_main(args, out)
        return
    config, src_adapter = write_config(out, args)
    log_path = out / "server.log"
    where = (f"GPU {args.router_gpu}" if args.router_gpu is not None
             else args.router_device)
    print(f"=== adapter routing, live (GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
          f"TP={args.tp}, {args.api_servers} API server(s), router on {where}) ===")
    print(f"    out dir: {out}")

    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               TOKENIZERS_PARALLELISM="false")
    env.setdefault("HF_HOME", "/home/jiaxuan/.cache/huggingface")
    log = open(log_path, "w")
    server = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--family", _BASE, "--config", str(config), "--tp", str(args.tp),
         "--port", str(args.port)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    port = args.port
    ok = False
    try:
        t0 = time.monotonic()
        early = None
        while True:
            status, _, _ = http(port, "/health", timeout=5)
            if status == 200:
                break
            if server.poll() is not None or time.monotonic() - t0 > args.timeout:
                check("server became healthy", False)
                return
            time.sleep(2)
        check(f"server healthy after {time.monotonic() - t0:.0f} s", True)

        status, headers, body = http(port, "/v1/completions", {
            "model": "auto", "prompt": "Translate to English: Guten Tag.",
            "max_tokens": 4, "temperature": 0})
        early = (status, headers.get("x-deltaserve-routing"),
                 headers.get("x-deltaserve-routing-reason"),
                 (body or {}).get("model") if isinstance(body, dict) else body)
        print(f"    auto request right after startup: {early}")
        check("an auto request is answered (200) whatever the job's state",
              status == 200)

        status, _, jobs = http(port, "/ft_jobs")
        check("GET /ft_jobs works", status == 200 and isinstance(jobs, dict))
        job = (jobs or {}).get("current") or ((jobs or {}).get("finished") or [None])[0]
        name = job["name"] if job else None
        print(f"    job: {name} stage={job and job['stage']}")
        check("the job has a generated dserve-routing- name and the four stages",
              bool(name) and name.startswith("dserve-routing-")
              and list(job["stages"]) == ["init", "training", "indexing", "ready"])

        seen, last, final = set(), None, None
        t1 = time.monotonic()
        while time.monotonic() - t1 < args.timeout:
            status, _, jobs = http(port, "/ft_jobs")
            if status != 200:
                break
            cur = jobs.get("current")
            if cur is None:
                final = (jobs.get("finished") or [None])[-1]
                break
            if cur["stage"] != last:
                last = cur["stage"]
                seen.add(last)
                print(f"    [{time.monotonic() - t1:5.1f}s] stage: {last} "
                      f"{cur.get('detail')}", flush=True)
            # keep the engine stepping: stages advance on scheduler steps
            http(port, "/v1/completions", {
                "model": _BASE, "prompt": "Hello", "max_tokens": 1,
                "temperature": 0}, timeout=60)
            time.sleep(1.0)
        print(f"    stages seen while running: {sorted(seen)}")
        print(f"    final: {final and {k: final[k] for k in ('stage', 'stages', 'error', 'saved_path')}}")
        if not check("the job finished without error, every stage done",
                     final is not None and final["error"] is None
                     and all(v == "done" for v in final["stages"].values())):
            return
        check("training and indexing were seen running",
              any("training" in s for s in seen)
              or final["stages"]["training"] == "done")

        saved = final["saved_path"]
        check("trained adapter saved on disk",
              bool(saved) and os.path.isfile(os.path.join(saved, "adapter_model.safetensors"))
              and os.path.isfile(os.path.join(saved, "adapter_config.json")))
        if saved:
            from safetensors.torch import load_file

            a = load_file(os.path.join(saved, "adapter_model.safetensors"))
            b = load_file(str(src_adapter / "adapter_model.safetensors"))
            diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in b)
            print(f"    max |trained - initial| over all factors: {diff:.3e}")
            check("same tensors and shapes as the source adapter",
                  set(a) == set(b) and all(a[k].shape == b[k].shape for k in b))
            check("the saved weights are trained (differ from the initial ones)",
                  diff > 0)

        idx = Path(out / "index")
        check("pool registry and routable index written",
              (idx / "pool.json").is_file() and any(idx.glob("routable-*.npz")))

        routed = None
        for _ in range(20):      # the router re-reads its index within ~1 s
            status, headers, body = http(port, "/v1/completions", {
                "model": "auto",
                "prompt": "Translate to English: Wir gehen heute ins Kino.",
                "max_tokens": 8, "temperature": 0})
            routed = (status, headers, body)
            if headers.get("x-deltaserve-routing") == "routed":
                break
            time.sleep(0.5)
        status, headers, body = routed
        print(f"    routed request: status={status} "
              f"{ {k: v for k, v in headers.items() if k.startswith('x-deltaserve')} } "
              f"model={body.get('model') if isinstance(body, dict) else body}")
        check("auto + translation prompt → 200", status == 200)
        check("headers: routed to the job's adapter, with a score",
              headers.get("x-deltaserve-routing") == "routed"
              and headers.get("x-deltaserve-adapter") == name
              and "x-deltaserve-routing-score" in headers)
        check("the response's model field is the adapter",
              isinstance(body, dict) and body.get("model") == name)

        status, headers, body = http(port, "/v1/completions", {
            "model": "auto", "prompt": "What is the capital of Australia?",
            "max_tokens": 4, "temperature": 0})
        print(f"    unrelated auto request: "
              f"{ {k: v for k, v in headers.items() if k.startswith('x-deltaserve')} }")
        check("an unrelated auto request is also answered", status == 200)

        status, headers, body = http(port, "/v1/completions", {
            "model": _BASE, "prompt": "Hello", "max_tokens": 2,
            "temperature": 0})
        check("a request naming the base model: 200, no routing headers",
              status == 200
              and not any(k.startswith("x-deltaserve") for k in headers)
              and body.get("model") == _BASE)

        status, headers, body = http(port, "/v1/completions", {
            "model": name, "prompt": "Translate to English: Danke.",
            "max_tokens": 4, "temperature": 0})
        check("the adapter is also servable by its name, without routing headers",
              status == 200 and body.get("model") == name
              and not any(k.startswith("x-deltaserve") for k in headers))
        for _ in range(6):       # several requests: with >1 API server each
            s2, h2, b2 = http(port, "/v1/completions", {   # resolves on its own
                "model": "auto", "prompt": "Translate to English: Bis morgen.",
                "max_tokens": 2, "temperature": 0})
            if s2 != 200 or (b2 or {}).get("model") != name:
                check(f"repeated routed requests keep working ({s2}, {b2})", False)
                break
        else:
            check("repeated routed requests keep working (all API servers)", True)
        if args.second_job:
            second_job(args, out, port, name, saved, src_adapter)
        ok = _failed == 0
    finally:
        stop_tree(server)
        log.close()
        if not ok:
            print(f"\n--- last lines of {log_path} ---")
            lines = log_path.read_text(errors="replace").splitlines()
            print("\n".join(lines[-60:]))


if __name__ == "__main__":
    main()
    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)
