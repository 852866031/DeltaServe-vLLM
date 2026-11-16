# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU governor for indexing: how many threads may the CPU indexer use now?

Indexing on the CPU must not disturb the vLLM processes or anything else on the
host. Before each chunk the governor looks at the host and answers with a
thread count (0 = pause):

- memory: pause while the host's available memory is below what the encoder
  needs plus a reserve;
- CPU: count the cores (of the allowed set) that were idle over the last
  interval, keep ``reserve_cores`` of them free, use at most ``max_threads``.

The indexer process additionally runs under ``SCHED_IDLE`` (it only gets CPU
time nothing else wants) and can be pinned to a core set — see
``apply_idle_priority`` / ``pin_to_cpus``. Priority does not protect against
memory-bandwidth or cache contention; the thread cap is what bounds that.

Linux /proc readers with injectable probes; no torch / vLLM import.
"""

import os
from dataclasses import dataclass


def parse_cpu_list(spec: str | None) -> set[int] | None:
    """'8-15,24' → {8..15, 24}; None / '' → None (no restriction)."""
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def read_mem_available_bytes() -> int:
    """Memory this process could still get: host MemAvailable, capped by the
    cgroup-v2 limit when there is one."""
    avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
                break
    if avail is None:
        raise OSError("MemAvailable not found in /proc/meminfo")
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                avail = min(avail, int(limit) - int(f.read().strip()))
    except (OSError, ValueError):
        pass
    return max(0, avail)


def read_cpu_times() -> dict[int, tuple[int, int]]:
    """Per CPU: (total jiffies, idle jiffies incl. iowait) from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = [int(x) for x in p[1:]]
                out[int(p[0][3:])] = (sum(v), v[3] + v[4])
    return out


@dataclass
class Decision:
    threads: int
    reason: str

    @property
    def run(self) -> bool:
        return self.threads > 0


class CpuGovernor:
    def __init__(self, max_threads: int = 8, reserve_cores: int = 2,
                 min_free_memory_bytes: int = 4 << 30,
                 need_memory_bytes: int = 3 << 30,
                 cpus: set[int] | None = None, idle_fraction: float = 0.8,
                 mem_probe=read_mem_available_bytes,
                 cpu_probe=read_cpu_times) -> None:
        self.max_threads = int(max_threads)
        self.reserve_cores = int(reserve_cores)
        self.min_free_memory_bytes = int(min_free_memory_bytes)
        # What loading the encoder costs; no longer counted once it is loaded.
        self.need_memory_bytes = int(need_memory_bytes)
        self.cpus = cpus
        self.idle_fraction = float(idle_fraction)
        self._mem_probe, self._cpu_probe = mem_probe, cpu_probe
        self._last = None

    def idle_cores(self) -> int | None:
        """Allowed cores that were at least ``idle_fraction`` idle since the
        previous call; None on the first call (no interval yet)."""
        now = self._cpu_probe()
        last, self._last = self._last, now
        if last is None:
            return None
        n = 0
        for cpu, (total, idle) in now.items():
            if self.cpus is not None and cpu not in self.cpus:
                continue
            if cpu not in last:
                continue
            dt = total - last[cpu][0]
            if dt > 0 and (idle - last[cpu][1]) / dt >= self.idle_fraction:
                n += 1
        return n

    def decide(self, loaded: bool = False) -> Decision:
        """Threads for the next chunk. ``loaded``: the encoder is already in
        memory (then only the reserve has to stay free)."""
        need = self.min_free_memory_bytes + (0 if loaded else self.need_memory_bytes)
        avail = self._mem_probe()
        if avail < need:
            return Decision(0, f"low memory: {avail >> 20} MiB available, "
                               f"need {need >> 20} MiB")
        idle = self.idle_cores()
        if idle is None:
            return Decision(0, "measuring CPU idle time")
        threads = min(self.max_threads, idle - self.reserve_cores)
        if threads < 1:
            return Decision(0, f"no idle cores: {idle} idle, "
                               f"{self.reserve_cores} reserved")
        return Decision(threads, f"{idle} idle cores")


def apply_idle_priority() -> str:
    """Put the calling process in the lowest scheduling class. Returns which
    mechanism took effect."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        return "SCHED_IDLE"
    except (AttributeError, OSError):
        pass
    try:
        os.nice(19)
        return "nice 19"
    except OSError:
        return "none"


def pin_to_cpus(cpus: set[int] | None) -> set[int]:
    """Restrict the calling process to ``cpus`` (intersected with what it may
    use); returns the resulting set."""
    allowed = os.sched_getaffinity(0)
    if cpus:
        target = allowed & set(cpus)
        if target:
            os.sched_setaffinity(0, target)
            return target
    return allowed
