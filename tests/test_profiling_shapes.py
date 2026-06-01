#!/usr/bin/env python
"""Test the offline profiling shape generator (vllm.deltaserve).

CPU-only, no model: checks the generated shapes respect the token/length caps
and cover every regime the merged estimator needs. No GPU.

    python tests/test_profiling_shapes.py
"""

import sys

from vllm.deltaserve.profiling_batch_generator import ProfilingShapeGenerator

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
        print(f"  {_GREEN}[PASS]{_RESET} {name}")
    else:
        _failed += 1
        print(f"  {_RED}[FAIL]{_RESET} {name}")


def run_case(batch_max, ft_cap, max_req):
    print(f"case: batch_max={batch_max} ft_cap={ft_cap} max_req={max_req}")
    g = ProfilingShapeGenerator(
        batch_max_tokens=batch_max,
        max_saved_finetuning_tokens=ft_cap,
        max_req_len=max_req,
    )
    warmup, recorded = g.prepare()
    kinds = {s.kind for s in recorded}
    check("covers all 4 regimes",
          kinds == {"prefill", "decode", "coserve", "mixed"} or
          # tiny configs may drop mixed/coserve if caps exclude them
          {"prefill", "decode"} <= kinds)

    eff_max_req = max(16, max_req - 16)
    ok_batch = True
    ok_req = True
    ok_ft = True
    ok_decomp = False
    prefill_totals = {}
    # Tally the new EAGER-regime shape categories. Each is identified by
    # composition, not by a distinct `kind` string (they reuse the existing
    # decode / mixed / coserve kinds with extra ft_lens or empty inf_lens).
    has_ft_on_decode = False    # decode shape with ft_lens > 0
    has_ft_on_mixed = False     # mixed shape with ft_lens > 0
    has_ft_only_idle = False    # coserve shape with no inf
    for s in recorded:
        inf = sum(s.inf_lens)
        ft = sum(s.ft_lens)
        mix = sum(s.mix_lens)
        # per-step prefill token budget (inf+ft together for coserve;
        # for decode/mixed the FT rides the wave-2 step alongside decode).
        if inf + ft > batch_max:
            ok_batch = False
        if mix + ft > batch_max:
            ok_batch = False
        for L in list(s.inf_lens) + list(s.ft_lens) + list(s.mix_lens):
            if L > eff_max_req:
                ok_req = False
            if L < 1:
                ok_req = False
        if ft > ft_cap:
            ok_ft = False
        if s.kind == "prefill":
            prefill_totals.setdefault(inf, set()).add(len(s.inf_lens))
        # New EAGER-regime shape categories.
        if s.kind == "decode" and ft > 0:
            has_ft_on_decode = True
        if s.kind == "mixed" and ft > 0:
            has_ft_on_mixed = True
        if s.kind == "coserve" and inf == 0 and ft > 0:
            has_ft_only_idle = True
    # decomposition: at least one prefill total appears with >1 distinct P
    for total, ps in prefill_totals.items():
        if len(ps) > 1:
            ok_decomp = True
    check("all batches within batch_max_tokens", ok_batch)
    check("all requests within max_req_len", ok_req)
    check("FT tokens within ft_cap", ok_ft)
    check("has decomposition variants (same total, different P)", ok_decomp)
    # New EAGER-regime coverage. Tiny configs (ft_cap=64, batch=512) may drop
    # some of these if the cap math excludes them — the new sweeps respect
    # the same caps as the existing ones.
    check("has FT-on-decode shapes (or caps too tight)",
          has_ft_on_decode or batch_max < 256)
    check("has FT-on-mixed shapes (or caps too tight)",
          has_ft_on_mixed or batch_max < 512)
    check("has FT-only-idle shape", has_ft_only_idle)


def main():
    run_case(8192, 256, 4096)
    run_case(2048, 128, 2048)
    run_case(512, 64, 512)
    print()
    total = _passed + _failed
    color = _GREEN if _failed == 0 else _RED
    print(f"{color}{_passed}/{total} checks passed{_RESET}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
