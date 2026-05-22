import os, sys, signal, subprocess, asyncio
from tqdm import tqdm
import aiohttp

from .common import send_one


def launch_instance(port, gpu, enable_finetuning, current_dir, instance_index):
    """
    Launch one server instance.
    NO NCCL PORT manipulation. No env overrides.
    """

    env = os.environ.copy()

    cmd = [
        sys.executable,
        f"{current_dir}/launch_server.py",
        "--port", str(port),
        "--rank_id", str(gpu),
    ]
    if enable_finetuning:
        cmd.append("--enable-finetuning")

    print("[launch]", " ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)


def kill_instance(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except:
        pass


async def warmup_instances(servers, schedule):
    warm = [(i, t, p) for i, t, p in schedule if t < 3.0]
    if not warm:
        return
    n = len(servers)

    async with aiohttp.ClientSession() as session:
        t0 = asyncio.get_event_loop().time()
        tasks = []
        for idx, _, payload in warm:
            tasks.append(
                asyncio.create_task(
                    send_one(session, servers[idx % n], idx, payload, t0)
                )
            )
        await asyncio.gather(*tasks)


async def run_rr_schedule(servers, schedule):
    n = len(servers)
    t0 = asyncio.get_event_loop().time()
    results = []

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(schedule))

        async def fire(i, t_off, payload):
            delay = (t0 + t_off) - asyncio.get_event_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)

            server = servers[i % n]
            r = await send_one(session, server, i, payload, t0)
            pbar.update(1)
            return r

        tasks = [asyncio.create_task(fire(i, t, p)) for i, t, p in schedule]
        results = await asyncio.gather(*tasks)
        pbar.close()

    return results