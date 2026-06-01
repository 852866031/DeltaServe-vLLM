# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline profiling shape generator for the merged SLO estimator (Phase 4).

Produces *shape specs* (plain lists of ints — no vLLM objects, so this is
unit-testable) that the launch-time driver (`EngineCore.profile_execution_model`)
turns into real synthetic requests and runs through the live scheduler. The
realized per-step features + GPU-timed durations are recorded by the
`FinetuneScheduler` and fitted into:

    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c

To identify all six coefficients the sweep needs diversity along every axis:

  * prefill token sweep + **decomposition variants** (same total tokens, varied
    sample count) → break the S↔T_in collinearity, identify α vs β.
  * decode B_d × K grid (vary request count and KV length independently) → δ, ε.
  * co-serve inf×ft grid → γ (activation-saving overhead).
  * mixed prefill+decode → the single constant c and the fused-step coupling
    (a fused step's decode tokens ride the prefill GEMM, so c/δ/ε fitted on
    mixed data differ from the pure-decode corners).

Caps respected: every prefill/co-serve batch ≤ ``batch_max_tokens``; FT tokens
≤ ``ft_cap``; per-request length ≤ ``max_req_len``.

Port of DeltaServe's ``router/profiling_batch_generator.py``, adapted to vLLM's
unified prefill+decode batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProfileShape:
    """One profiling unit the driver injects + drives.

    kind:
      "prefill" — inject inf reqs, run one step (their joint prefill).
      "coserve" — inject inf + FT reqs, run one step (eager co-serve).
      "decode"  — inject inf reqs, prefill them, then drive `n_decode` decode
                  steps (each recorded — pure decode features).
      "mixed"   — prefill `inf_lens` (max_tokens lets them decode), then inject
                  `mix_lens` as a second wave so the next step decodes the first
                  wave while prefilling the second (a fused mixed step).
    """

    kind: str
    inf_lens: list[int] = field(default_factory=list)
    ft_lens: list[int] = field(default_factory=list)
    mix_lens: list[int] = field(default_factory=list)
    n_decode: int = 0


class ProfilingShapeGenerator:
    def __init__(
        self,
        *,
        batch_max_tokens: int,
        max_saved_finetuning_tokens: int,
        max_req_len: int,
    ) -> None:
        self.batch_max_tokens = int(batch_max_tokens)
        self.ft_cap = max(1, int(max_saved_finetuning_tokens))
        # Reserve a small margin under the per-request hard cap.
        self.max_req_len = max(16, int(max_req_len) - 16)
        # Keep prefill sweep points safely within the per-step token budget.
        self.inf_cap = max(64, min(self.batch_max_tokens, 4096))

    # ---------------- sweeps ----------------
    def _prefill_token_targets(self) -> list[int]:
        base = [64, 128, 256, 512, 1024, 2048, 4096]
        targets = {t for t in base if t <= self.inf_cap}
        for frac in (0.5, 0.75, 1.0):
            targets.add(max(64, int(self.inf_cap * frac)))
        return sorted(targets)

    def _partition(self, total: int, n_parts: int) -> list[int]:
        n_parts = max(1, min(n_parts, total))
        # bump n_parts so no piece exceeds max_req_len
        n_parts = max(n_parts, math.ceil(total / self.max_req_len))
        base, rem = divmod(total, n_parts)
        return [base + 1] * rem + [base] * (n_parts - rem)

    def _prefill_shapes(self) -> list[ProfileShape]:
        out: list[ProfileShape] = []
        # token sweep, default ~128 tok/req
        for total in self._prefill_token_targets():
            n = max(1, min(8, total // 128))
            out.append(ProfileShape("prefill", inf_lens=self._partition(total, n)))
        # decomposition variants at two pivots: vary sample count at fixed total
        for total in (min(512, self.inf_cap), min(2048, self.inf_cap)):
            if total < 256:
                continue
            for n in (1, 4, max(4, total // 64)):
                out.append(ProfileShape("prefill",
                                        inf_lens=self._partition(total, n)))
        return out

    def _decode_shapes(self) -> list[ProfileShape]:
        out: list[ProfileShape] = []
        # B_d grid at fixed-ish per-seq KV; K grid at fixed B_d.
        for b in (1, 2, 4, 8, 16, 32):
            per = min(self.max_req_len, max(16, self.inf_cap // max(b, 1)))
            if b * per > self.batch_max_tokens:
                per = max(16, self.batch_max_tokens // b)
            out.append(ProfileShape("decode", inf_lens=[per] * b, n_decode=2))
        for per in (32, 128, 512, 1024):
            per = min(per, self.max_req_len)
            if 4 * per <= self.batch_max_tokens:
                out.append(ProfileShape("decode", inf_lens=[per] * 4, n_decode=2))
        return out

    def _coserve_shapes(self) -> list[ProfileShape]:
        out: list[ProfileShape] = []
        ft_levels = sorted({max(8, self.ft_cap // 4),
                            max(16, self.ft_cap // 2),
                            self.ft_cap})
        inf_levels = [128, 512, 1024]
        for ft in ft_levels:
            for inf in inf_levels:
                if inf + ft > self.batch_max_tokens:
                    continue
                out.append(ProfileShape(
                    "coserve",
                    inf_lens=self._partition(inf, max(1, inf // 256)),
                    ft_lens=self._partition(ft, max(1, ft // 64))))
        # FT-dominant edge: tiny inf, full FT cap
        if 32 + self.ft_cap <= self.batch_max_tokens:
            out.append(ProfileShape("coserve", inf_lens=[32],
                                    ft_lens=self._partition(self.ft_cap,
                                                            max(1, self.ft_cap // 64))))
        return out

    def _mixed_shapes(self) -> list[ProfileShape]:
        out: list[ProfileShape] = []
        for inf, mix in ((256, 128), (512, 256), (1024, 256)):
            if inf > self.inf_cap or mix > self.inf_cap:
                continue
            out.append(ProfileShape(
                "mixed",
                inf_lens=self._partition(inf, max(1, inf // 256)),
                mix_lens=self._partition(mix, max(1, mix // 128)),
                n_decode=1))
        return out

    # ─── EAGER regime coverage (three-regime estimator redesign) ──────
    # The three composition-based regimes are:
    #   INF_PREFILL (T_ft=0, T_in>0) — covered by _prefill_shapes + _mixed_shapes
    #   DECODE_ONLY (T_in=T_ft=0)    — covered by _decode_shapes
    #   EAGER       (T_ft>0)         — covered by _coserve_shapes ((T_in>0, B_d=0))
    #                                  + the three sweeps below (B_d>0 cases
    #                                  + pure FT-only)
    # Required to seed the EAGER regime across all shapes the live unified-
    # phase scheduler may encounter.

    def _ft_on_decode_shapes(self) -> list[ProfileShape]:
        """EAGER regime, ``(T_in=0, B_d>0, K>0, T_ft>0)`` — FT admitted onto a
        decode-only step (unified-phase scheduler's new admission case).
        Uses the existing ``decode`` kind extended with ``ft_lens``: the
        driver prefills the inf reqs (wave 1), then injects FT alongside the
        first decode step."""
        out: list[ProfileShape] = []
        # Single-sample FT injection — the typical unified-phase admission
        # pattern (admit one FT sample per decode-only step). Reserve room
        # for FT in the batch budget when sizing per-decode KV.
        ft_size = max(8, min(self.ft_cap, 64))
        budget_for_decode = self.batch_max_tokens - ft_size
        if budget_for_decode < 16:
            return []
        for b in (1, 2, 4, 8):
            # Size per-decode KV to fit alongside ft_size in the batch.
            per = min(self.max_req_len, max(16, budget_for_decode // b))
            if b * per > budget_for_decode:
                per = max(16, budget_for_decode // b)
            # Final cap check (should always hold after the sizing above).
            if b * per + ft_size > self.batch_max_tokens:
                continue
            out.append(ProfileShape(
                "decode",
                inf_lens=[per] * b,
                ft_lens=[ft_size],
                n_decode=2))
        return out

    def _ft_on_mixed_shapes(self) -> list[ProfileShape]:
        """EAGER regime, ``(T_in>0, B_d>0, K>0, T_ft>0)`` — FT admitted onto
        a mixed prefill+decode step (the typical default-scheduler admission
        case under steady-state). Uses the existing ``mixed`` kind extended
        with ``ft_lens``: the driver injects FT alongside wave-2 prefill."""
        out: list[ProfileShape] = []
        for inf, mix, ft in ((256, 128, 64), (512, 256, 128), (1024, 256, 64)):
            if inf > self.inf_cap or mix > self.inf_cap or ft > self.ft_cap:
                continue
            if mix + ft > self.batch_max_tokens:
                continue
            out.append(ProfileShape(
                "mixed",
                inf_lens=self._partition(inf, max(1, inf // 256)),
                mix_lens=self._partition(mix, max(1, mix // 128)),
                ft_lens=self._partition(ft, max(1, ft // 64)),
                n_decode=1))
        return out

    def _ft_only_idle_shapes(self) -> list[ProfileShape]:
        """EAGER regime, ``(T_in=0, B_d=0, K=0, T_ft>0)`` — pure FT-fill on
        idle inference (the buffer-fill case during quiet inference windows).
        Uses the existing ``coserve`` kind with empty ``inf_lens`` — the
        driver's coserve branch already iterates inf_lens (no-op when empty)
        and then ft_lens. Single representative shape; cheap to record."""
        if self.ft_cap <= 0:
            return []
        return [ProfileShape(
            "coserve",
            inf_lens=[],
            ft_lens=self._partition(self.ft_cap, max(1, self.ft_cap // 64)))]

    def prepare(self) -> tuple[list[ProfileShape], list[ProfileShape]]:
        """Return (warmup_shapes, recorded_shapes).

        Warmup is a small subset run once with recording OFF (primes cuBLAS /
        FlashInfer JIT / autotune). Recorded shapes are the full sweep; the
        driver repeats them `num_repeats` times.
        """
        recorded = (self._prefill_shapes()
                    + self._decode_shapes()
                    + self._coserve_shapes()
                    + self._mixed_shapes()
                    + self._ft_on_decode_shapes()
                    + self._ft_on_mixed_shapes()
                    + self._ft_only_idle_shapes())
        # Warmup: one representative of each kind so JIT/autotune covers the
        # eager-with-FT-decode path too.
        warmup = [
            ProfileShape("prefill", inf_lens=[256, 256]),
            ProfileShape("decode", inf_lens=[128, 128], n_decode=1),
        ]
        if 128 + min(self.ft_cap, 64) <= self.batch_max_tokens:
            warmup.append(ProfileShape("coserve", inf_lens=[128],
                                       ft_lens=[min(self.ft_cap, 64)]))
        # FT-on-decode warmup (primes the new shape's kernels).
        ft_warm = min(self.ft_cap, 32)
        if 2 * 64 + ft_warm <= self.batch_max_tokens:
            warmup.append(ProfileShape(
                "decode", inf_lens=[64, 64],
                ft_lens=[ft_warm], n_decode=1))
        return warmup, recorded
