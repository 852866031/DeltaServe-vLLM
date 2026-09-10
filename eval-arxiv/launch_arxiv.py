#!/usr/bin/env python
"""Bring up the Qwen3-14B TP=2 server on the arxiv (16k-context) YAML.

Thin wrapper over eval-tp/launch_deltaserve.build_server_cmd so the launch is
identical to the TP eval harness. Two modes:

    python eval-arxiv/launch_arxiv.py                      # inference only (no backward child)
    python eval-arxiv/launch_arxiv.py --co [--start-finetuning]   # co-serving

Inference-only forces the v1 model runner (VLLM_USE_V2_MODEL_RUNNER=0) so the
prefill numbers are measured on the same runner co-serving uses (Qwen3 would
otherwise default to the v2 runner). ``--max-num-batched-tokens`` overrides the
prefill chunk (vLLM default 2048). Logs / bwd log / step trace land in
eval-arxiv/output/. Ctrl-C shuts the whole process group down.
"""
import argparse, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "eval-tp"))
from launch_deltaserve import (build_server_cmd, post_start_finetuning,  # noqa: E402
                               terminate, wait_for_health)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", default=str(_ROOT / "configs" / "serving_config_finetuning_qwen3_14b_tp2_arxiv.yaml"))
ap.add_argument("--family", default="qwen3-14b")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--tp", type=int, default=None)
ap.add_argument("--co", action="store_true", help="co-serving (finetune sections passed; backward children spawned)")
ap.add_argument("--start-finetuning", action="store_true", help="POST /start_finetuning once healthy (co only)")
ap.add_argument("--max-num-batched-tokens", type=int, default=None)
ap.add_argument("--tag", default="", help="suffix for the output files")
ap.add_argument("--startup-timeout", type=float, default=900.0)
args = ap.parse_args()

out = _HERE / "output"; out.mkdir(exist_ok=True)
mode = "co" if args.co else "inf"
tag = f"_{args.tag}" if args.tag else ""
spec = build_server_cmd(args.config, args.port, None, args.tp, family=args.family, co=args.co,
                        bwd_log_path=str(out / f"bwd_log_{mode}{tag}.csv") if args.co else None,
                        step_trace_path=str(out / f"step_trace_{mode}{tag}.csv") if args.co else None)
cmd = list(spec.cmd)
if args.max_num_batched_tokens:
    cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
env = dict(os.environ)
env.setdefault("HF_HOME", "/mnt/storage/huggingface"); env.setdefault("HF_HUB_OFFLINE", "1")
env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0"); env["PYTHONUNBUFFERED"] = "1"
if not args.co:
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
log = out / f"server_{mode}{tag}.log"
print(f"[arxiv] mode={mode} tp={spec.tp} log={log}\n[arxiv] cmd = {' '.join(cmd)}", flush=True)
with open(log, "w") as lf:
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(), start_new_session=True,
                            stdout=lf, stderr=subprocess.STDOUT)
server = f"http://127.0.0.1:{args.port}"
try:
    if not wait_for_health(server, args.startup_timeout, proc):
        terminate(proc); sys.exit(1)
    if args.co and args.start_finetuning:
        post_start_finetuning(server)
    print("[arxiv] server is up. Ctrl-C to shut down.", flush=True)
    while proc.poll() is None:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\n[arxiv] Ctrl-C — shutting down…", flush=True)
finally:
    terminate(proc)
    # the server does not always exit on SIGINT to its group: SIGTERM the tree
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
