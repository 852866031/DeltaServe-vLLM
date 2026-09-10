"""Prefill-time probe: send real arxiv articles truncated to N tokens, stream the
completion, and report TTFT (= prefill time when the server is otherwise idle).
Distinct article per request so vLLM prefix caching never hits."""
import argparse, asyncio, csv, json, statistics as st, time
import aiohttp
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen3-14b")
ap.add_argument("--tok", default="Qwen/Qwen3-14B-Base")
ap.add_argument("--data", required=True)
ap.add_argument("--lengths", default="2048,4096,8192,12288,16000")
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--conc", type=int, default=4, help="concurrent 8k requests in phase 2 (0=skip)")
ap.add_argument("--conc-len", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=8)
ap.add_argument("--gap", type=float, default=0.0, help="idle seconds between sequential requests")
ap.add_argument("--tag", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tok)
docs = [json.loads(l) for l in open(args.data)]
ids = []
for d in docs:
    ids.append(tok(d["article"], add_special_tokens=False)["input_ids"])
ntok = [len(x) for x in ids]
print(f"[bench] {len(docs)} articles; tokens p50={st.median(ntok):.0f} max={max(ntok)}", flush=True)

used = set()
def take(L):
    """Return a prompt string with ~L tokens from an unused article of >= L tokens."""
    for i in sorted(range(len(ids)), key=lambda i: ntok[i]):
        if i in used or ntok[i] < L + 64:
            continue
        used.add(i)
        return tok.decode(ids[i][:L])
    raise SystemExit(f"no unused article with >= {L} tokens")

async def one(session, prompt, tag):
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; ptoks = None; nchunks = 0
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
                nchunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if j.get("usage"):
                ptoks = j["usage"]["prompt_tokens"]
    e2e = time.perf_counter() - t0
    return {"tag": tag, "prompt_tokens": ptoks, "ttft_s": round(ttft, 4) if ttft else None,
            "e2e_s": round(e2e, 4), "gen_chunks": nchunks}

async def main():
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        # warm-up (not recorded)
        await one(s, take(1024), "warm")
        for L in [int(x) for x in args.lengths.split(",")]:
            for r in range(args.reps):
                if args.gap > 0:
                    await asyncio.sleep(args.gap)
                row = await one(s, take(L), f"seq_L{L}{args.tag}")
                row["mode"] = "sequential"; row["target_len"] = L
                rows.append(row); print("[bench]", row, flush=True)
        if args.conc > 0:
            prompts = [take(args.conc_len) for _ in range(args.conc)]
            t0 = time.perf_counter()
            res = await asyncio.gather(*[one(s, p, f"conc{args.conc}_L{args.conc_len}") for p in prompts])
            wall = time.perf_counter() - t0
            for row in res:
                row["mode"] = f"concurrent{args.conc}"; row["target_len"] = args.conc_len
                rows.append(row); print("[bench]", row, flush=True)
            print(f"[bench] concurrent batch wall = {wall:.3f}s", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n[bench] summary (sequential, TTFT s):")
    for L in sorted({r["target_len"] for r in rows if r["mode"] == "sequential"}):
        v = [r["ttft_s"] for r in rows if r["mode"] == "sequential" and r["target_len"] == L]
        pt = [r["prompt_tokens"] for r in rows if r["mode"] == "sequential" and r["target_len"] == L]
        print(f"  L={L:6d} prompt_tokens~{st.median(pt):.0f}  ttft min/median/max = {min(v):.3f}/{st.median(v):.3f}/{max(v):.3f}")
asyncio.run(main())
