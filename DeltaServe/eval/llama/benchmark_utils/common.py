import socket, requests, json, time, aiohttp, asyncio
import pandas as pd
import random, string
from pathlib import Path
import numpy as np


def internet_available(timeout=2):
    try:
        socket.gethostbyname("huggingface.co")
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except:
        return False


if internet_available():
    base_model = "huggyllama/llama-7b"
    DEFAULT_LORA = "tloen/alpaca-lora-7b"
else:
    base_model = "/projects/I20240005/jchen/hf_cache/models--huggyllama--llama-7b/snapshots/llama-7b"
    DEFAULT_LORA = "/projects/I20240005/jchen/hf_cache/hub/models--tloen--alpaca-lora-7b/snapshots/12103d6baae1b320aa60631b38acb6ea094a0539"


def generate_random_sentence(length: int) -> str:
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 5)))
        for _ in range(max(1, length))
    ]
    return " ".join(words).capitalize() + "."


def make_payload(prompt, max_new):
    return {
        "model_dir": base_model,
        "lora_dir": DEFAULT_LORA,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": int(max_new)
        }
    }


def load_schedule(csv_path):
    df = pd.read_csv(csv_path)
    schedule = []
    for idx, row in enumerate(df.itertuples(index=False)):
        prompt = generate_random_sentence(int(row.prompt_length))
        schedule.append((idx, float(row.timestamp_s), make_payload(prompt, row.max_new_tokens)))
    return schedule


async def try_health(session, server):
    try:
        async with session.get(f"{server}/health", timeout=1) as resp:
            return resp.status == 200
    except:
        return False


async def try_probe(session, server):
    try:
        async with session.post(
            f"{server}/generate",
            json=make_payload("ping", 4),
            timeout=2
        ) as resp:
            return resp.status == 200
    except:
        return False


async def wait_for_server(server, max_wait_s=120):
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server) or await try_probe(session, server):
                print("[ready]", server)
                return
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"{server} timeout")
            await asyncio.sleep(0.5)


async def send_one(session, server, idx, payload, t0):
    t_send = time.monotonic()
    t_rel = t_send - t0
    try:
        async with session.post(f"{server}/generate", json=payload) as resp:
            body = await resp.read()
            latency = time.monotonic() - t_send
            try:
                data = json.loads(body)
                return (idx, t_rel, latency, "ok",
                        data.get("ttft"), data.get("avg_tbt"), data.get("worst_tbt"))
            except:
                return (idx, t_rel, latency, "ok", None, None, None)
    except:
        return (idx, t_rel, time.monotonic() - t_send, "err", None, None, None)


async def start_finetuning(session, server):
    try:
        async with session.post(f"{server}/start_finetuning", timeout=1) as resp:
            return resp.status == 200
    except:
        return False


def summarize(results):
    ok = [x[2] for x in results if x[3] == "ok"]
    print("== Latency ==")
    print("mean:", np.mean(ok))
    print("p50/p90:", np.percentile(ok, 50), np.percentile(ok, 90))
    print("min/max:", min(ok), max(ok))


def write_latency_csv(results, out_stem):
    path = f"{out_stem}"
    with open(path, "w") as f:
        f.write("idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s\n")
        for r in results:
            f.write(",".join(str(x) for x in r) + "\n")
    print("[auto] Wrote CSV:", path)