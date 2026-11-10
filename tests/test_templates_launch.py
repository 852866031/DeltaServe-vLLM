#!/usr/bin/env python
"""Every configs/template_*.yaml launches, serves and trains as shipped.

For each template: a real server through eval-tp/launch_deltaserve.py with the
template's settings unchanged except that its output paths (adapter saves,
telemetry) are redirected to a temp dir. Checks: healthy; GET /finetuning_status
shows backward cycles; a completion is answered; POST /save_adapter with no path
writes under the template's adapter_save_path; the telemetry backward log has
rows.

    python tests/test_templates_launch.py                     # every template
    python tests/test_templates_launch.py qwen3_0.6b llama3   # name filters

Needs the template's model weights + adapters and its GPUs (TP=2 templates use
two). ~3-6 min per template.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
_PORT = 8017
_TTY = sys.stdout.isatty()
_G, _R, _X = ("\033[92m", "\033[91m", "\033[0m") if _TTY else ("", "", "")
_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  {_G + '[PASS]' if cond else _R + '[FAIL]'}{_X} {name}  {detail}", flush=True)
    _passed += bool(cond)
    _failed += not cond


def http(method, path, body=None, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{_PORT}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception as e:  # noqa: BLE001
        return None, {"error": str(e)}


def redirected(template: Path, tmp: Path) -> Path:
    """The template with relative paths made absolute and its outputs moved
    into ``tmp`` (nothing else changed)."""
    raw = yaml.safe_load(template.read_text())
    for sec in raw.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                if "path" in k and isinstance(v, str) and not os.path.isabs(v):
                    sec[k] = str((template.parent / v).resolve())
    raw["finetune_param"]["adapter_save_path"] = str(tmp / "checkpoints")
    for k in list((raw.get("telemetry") or {})):
        if k.endswith("_path") or k.endswith("_dir"):
            raw["telemetry"][k] = str(tmp / "logs" / Path(raw["telemetry"][k]).name)
    raw["server"]["port"] = _PORT
    out = tmp / template.name
    out.write_text(yaml.safe_dump(raw))
    return out


def run_template(template: Path) -> None:
    raw = yaml.safe_load(template.read_text())
    tp = int(raw["parallel"]["tensor_parallel_size"])
    print(f"=== {template.relative_to(_REPO)} (TP={tp}) ===", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="ds-tmpl-"))
    cfg = redirected(template, tmp)
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    log = open(tmp / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--config", str(cfg), "--port", str(_PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    ok_all = False
    try:
        deadline = time.monotonic() + 1200
        while http("GET", "/health", timeout=3)[0] != 200:
            if proc.poll() is not None or time.monotonic() > deadline:
                check(f"server healthy (log: {tmp / 'server.log'})", False)
                return
            time.sleep(3)
        check("server healthy", True)
        st, deadline = {}, time.monotonic() + 600
        while time.monotonic() < deadline:
            code, st = http("GET", "/finetuning_status", timeout=10)
            if code == 200 and st.get("backwards_done", 0) >= 2:
                break
            time.sleep(2)
        check("trains as shipped (≥2 backward cycles)",
              st.get("backwards_done", 0) >= 2 and st.get("disabled_reason") is None,
              json.dumps({k: st.get(k) for k in ("backwards_done", "tokens_trained",
                                                 "last_loss", "disabled_reason")}))
        _, models = http("GET", "/v1/models")
        code, out = http("POST", "/v1/completions", {
            "model": models["data"][0]["id"], "prompt": "The capital of France is",
            "max_tokens": 8, "temperature": 0})
        check("answers a completion", code == 200 and bool(out["choices"][0]["text"]),
              repr(out.get("choices", [{}])[0].get("text")))
        code, res = http("POST", "/save_adapter", {})
        check("POST /save_adapter → adapter_save_path/manual-*",
              code == 200 and res.get("saved")
              and res["path"].startswith(str(tmp / "checkpoints" / "manual-"))
              and (Path(res["path"]) / "adapter_model.safetensors").exists()
              and res.get("tp_size") == tp, res.get("path", res.get("error", "")))
        logs = list((tmp / "logs").glob("*.csv")) if (tmp / "logs").exists() else []
        rows = sum(max(0, len(p.read_text().splitlines()) - 1) for p in logs)
        check("telemetry backward log written", rows >= 1,
              f"{[p.name for p in logs]} {rows} rows")
        ok_all = True
    finally:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(90)
        except Exception:  # noqa: BLE001
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
        if ok_all and not _failed:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"  (kept {tmp})")


def main():
    filters = sys.argv[1:]
    templates = sorted((_REPO / "configs").glob("template_*.yaml"))
    if filters:
        templates = [t for t in templates if any(f in t.name for f in filters)]
    for t in templates:
        run_template(t)
    color = _R if _failed else _G
    print(f"\n{color}{_passed} passed, {_failed} failed{_X}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
