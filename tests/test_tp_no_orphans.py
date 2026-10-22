#!/usr/bin/env python
"""Phase 7 / M1 gate: a crashed EngineCore leaks no GPU process.

Launches a real TP=2 co-serving server (Qwen3-0.6B, both GPUs), waits for it
to be healthy, SIGKILLs the EngineCore, and checks that every TP worker and
every backward child dies with it (parent-death signal) within 20 s.

    python tests/test_tp_no_orphans.py        # needs both GPUs free (~2 min)
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

_REPO = Path(__file__).resolve().parent.parent
_PORT = 8011
_TTY = sys.stdout.isatty()
_GREEN = "\033[92m" if _TTY else ""
_RED = "\033[91m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""
_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    mark = f"{_GREEN}[PASS]{_RESET}" if cond else f"{_RED}[FAIL]{_RESET}"
    print(f"  {mark} {name}")
    if cond:
        _passed += 1
    else:
        _failed += 1


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _cmd(p: psutil.Process) -> str:
    try:
        return " ".join(p.cmdline()) + " " + p.name()
    except psutil.Error:
        return ""


def _kind(p: psutil.Process) -> str | None:
    name = _cmd(p)
    if "EngineCore" in name:
        return "engine"
    if "Worker_TP" in name or "VllmWorker" in name:
        return "worker"
    return None


def _backward_children(worker: psutil.Process) -> list[psutil.Process]:
    """The worker's spawn-context children, minus Inductor's compile pool."""
    return [c for c in worker.children()
            if "multiprocessing.spawn" in _cmd(c) and "compile_worker" not in _cmd(c)]


def main() -> None:
    print("=== M1: no orphans after an EngineCore crash (TP=2) ===")
    log = open("/tmp/test_tp_no_orphans.log", "w")
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
    env.setdefault("HF_HOME", "/mnt/storage/huggingface")
    launcher = subprocess.Popen(
        [sys.executable, str(_REPO / "eval-tp/launch_deltaserve.py"),
         "--family", "qwen3-0.6b", "--tp", "2", "--port", str(_PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 600
        while not _healthy():
            if launcher.poll() is not None or time.monotonic() > deadline:
                check("server became healthy (see /tmp/test_tp_no_orphans.log)", False)
                return
            time.sleep(2)
        check("server healthy", True)
        tree = psutil.Process(launcher.pid).children(recursive=True)
        kinds: dict[str, list[psutil.Process]] = {}
        for p in tree:
            k = _kind(p)
            if k:
                kinds.setdefault(k, []).append(p)
        engines = kinds.get("engine", [])
        workers = kinds.get("worker", [])
        # Backward children are the workers' children.
        children = [c for w in workers for c in _backward_children(w)]
        check("found the EngineCore", len(engines) == 1)
        check("found 2 TP workers", len(workers) == 2)
        check("found a backward child per worker", len(children) == 2)
        if len(engines) != 1:
            return
        engines[0].send_signal(signal.SIGKILL)
        gone, alive = psutil.wait_procs(workers + children, timeout=20)
        check(f"all workers + backward children exit within 20 s "
              f"(alive: {[p.pid for p in alive]})", not alive)
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        launcher.wait()
        log.close()


if __name__ == "__main__":
    main()
    color = _RED if _failed else _GREEN
    print(f"\n{color}{_passed} passed, {_failed} failed{_RESET}")
    sys.exit(1 if _failed else 0)
