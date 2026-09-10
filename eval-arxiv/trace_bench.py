#!/usr/bin/env python
"""Open-loop arxiv replay: Poisson arrivals at --rps, --n requests, prompts of
~--len tokens (± --jitter) cut from distinct articles (reused round-robin only if
the pool runs out), max_tokens = the article's abstract length in tokens (Ramya's
`output_tokens`), capped by --max-out. Prompts are tokenized/cached up front so the
first request goes out the moment the run starts. Writes one row per request:
arrival offset, TTFT, E2E, output tokens, mean / max inter-token gap.

    python eval-arxiv/trace_bench.py --data eval-arxiv/data/long_articles.jsonl \
        --rps 0.4 --n 20 --len 8192 --out eval-arxiv/output/trace_co.csv
"""
import argparse, asyncio, csv, json, random, statistics as st, time
import aiohttp
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen3-14b")
ap.add_argument("--tok", default="Qwen/Qwen3-14B-Base")
ap.add_argument("--data", required=True)
ap.add_argument("--rps", type=float, default=0.4)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--len", type=int, default=8192)
ap.add_argument("--jitter", type=int, default=512, help="prompt length drawn uniformly in [len-jitter, len+jitter]")
ap.add_argument("--max-out", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--warmup", action="store_true", help="one untimed 1k request before the trace")
ap.add_argument("--out", required=True)
args = ap.parse_args()

rng = random.Random(args.seed)
tok = AutoTokenizer.from_pretrained(args.tok)
docs = [json.loads(l) for l in open(args.data)]
# arrival offsets (Poisson) + prompt lengths, drawn first so they are identical across modes
gaps = [rng.expovariate(args.rps) for _ in range(args.n)]
arrivals = [sum(gaps[:i]) for i in range(args.n)]           # first arrival at t=0
lens = [rng.randint(args.len - args.jitter, args.len + args.jitter) for _ in range(args.n)]

# pick distinct articles long enough; reuse round-robin if fewer than n qualify
pool = []
for d in docs:
    ids = tok(d["article"], add_special_tokens=False)["input_ids"]
    if len(ids) >= args.len + args.jitter + 64:
        pool.append((ids, d["abstract"]))
    if len(pool) >= args.n:
        break
if not pool:
    raise SystemExit("no article long enough")
if len(pool) < args.n:
    print(f"[trace] only {len(pool)} articles >= {args.len + args.jitter} tokens — reusing round-robin", flush=True)
reqs = []
for i in range(args.n):
    ids, abstract = pool[i % len(pool)]
    prompt = tok.decode(ids[:lens[i]])
    n_out = min(args.max_out, max(16, len(tok(abstract, add_special_tokens=False)["input_ids"])))
    reqs.append((prompt, n_out))
print(f"[trace] {args.n} requests, rps={args.rps}, last arrival at {arrivals[-1]:.1f}s, "
      f"prompt tokens {min(lens)}-{max(lens)}, output tokens {min(r[1] for r in reqs)}-{max(r[1] for r in reqs)} "
      f"(median {st.median(r[1] for r in reqs):.0f}), distinct articles {min(len(pool), args.n)}", flush=True)

async def one(session, i, prompt, n_out, t_start, t_arr):
    await asyncio.sleep(max(0.0, t_start + t_arr - time.perf_counter()))
    body = {"model": args.model, "prompt": prompt, "max_tokens": n_out, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; ptoks = None; stamps = []
    async with session.post(f"{args.server}/v1/completions", json=body) as r:
        assert r.status == 200, (r.status, await r.text())
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            if j.get("choices") and j["choices"][0].get("text"):
                now = time.perf_counter(); stamps.append(now)
                if ttft is None: ttft = now - t0
            if j.get("usage"):
                ptoks = j["usage"]["prompt_tokens"]
    e2e = time.perf_counter() - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    row = {"req": i, "arrival_s": round(t_arr, 3), "prompt_tokens": ptoks, "max_tokens": n_out,
           "out_chunks": len(stamps), "ttft_s": round(ttft, 4) if ttft else None, "e2e_s": round(e2e, 4),
           "tbt_mean_ms": round(1e3 * st.mean(gaps), 2) if gaps else None,
           "tbt_max_ms": round(1e3 * max(gaps), 2) if gaps else None}
    print("[trace]", row, flush=True)
    return row

async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        if args.warmup:
            await one(s, -1, reqs[0][0][:4000], 8, time.perf_counter(), 0.0)
            await asyncio.sleep(2.0)
        t_start = time.perf_counter()
        t_wall = time.time()
        rows = await asyncio.gather(*[one(s, i, p, n, t_start, arrivals[i]) for i, (p, n) in enumerate(reqs)])
    for r in rows: r["t_wall_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_wall + r["arrival_s"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump({"t0_wall": t_wall, "t0_wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_wall)),
                   "rps": args.rps, "n": args.n, "seed": args.seed, "len": args.len, "jitter": args.jitter}, f)
    tt = [r["ttft_s"] for r in rows]; ee = [r["e2e_s"] for r in rows]; mx = [r["tbt_max_ms"] for r in rows if r["tbt_max_ms"]]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    print(f"\n[trace] TTFT p50/p90/max = {st.median(tt):.2f}/{q(tt,.9):.2f}/{max(tt):.2f} s | "
          f"E2E p50/max = {st.median(ee):.2f}/{max(ee):.2f} s | max TBT p50/max = {st.median(mx):.0f}/{max(mx):.0f} ms "
          f"| trace {arrivals[-1]:.1f}s, wall {max(r['arrival_s']+r['e2e_s'] for r in rows):.1f}s", flush=True)
asyncio.run(main())
