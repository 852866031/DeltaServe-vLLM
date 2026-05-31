# Questions for the DeltaServe-vLLM Implementer

I'm taking over the next round of paper writing. All questions below
are scoped to what you can answer from the **original DeltaServe
code** and the **DeltaServe-vLLM code** — you don't need to look at
the paper. Pretend I haven't seen either codebase and need you to
explain what's there.

Context you don't need to give me: the SGLang port is acknowledged as
existing and is not discussed in this revision; the S-LoRA port is
assumed unchanged from the original DeltaServe.

Most questions should be answerable in 1–3 sentences. I've flagged the
few that benefit from a code pointer or a diagram.

---

## 1. One-paragraph description of DeltaServe-vLLM

1. In your own words, what is DeltaServe-vLLM? How would you describe
   to a new engineer (a) what it does, (b) what it sits on top of, and
   (c) what an operator gets for free that they wouldn't get with
   vanilla vLLM?

2. Do you think of what you built as a vLLM-specific feature, or as a
   reusable layer that happens to be wired into vLLM first? If the
   latter, what code in the DeltaServe-vLLM tree is host-agnostic
   versus vLLM-specific? A rough split (directories or modules) is
   fine.

---

## 2. What survives from the original DeltaServe code

For each mechanism below, mark **kept / modified / dropped**. For
"modified", 2–3 sentences on what changed in DeltaServe-vLLM and why.

| Mechanism (original DeltaServe) | kept / modified / dropped | If modified, what changed |
|---|---|---|
| Mixed prefill batch fusing inference + LoRA forward | | |
| Closed-form prefill latency model | | |
| Closed-form decode latency model | | |
| Two-mode (graph vs eager) coefficient sets in the estimator | | |
| Offline profiling pass that seeds the estimator | | |
| Online least-squares refit of estimator coefficients | | |
| Greedy router that selects requests by SLO budget | | |
| BatchConstructor that gates on TTFT and TPOT | | |
| Decoupled backward subprocess | | |
| Cooperative per-layer pause / `_maybe_pause()` contract | | |
| MPS partition for concurrent backward | | |
| Unified paged pool extended with FT activation pages | | |
| GQA-packed KV view | | |
| Backward CUDA graph capture at fixed `max_saved_finetuning_tokens` | | |

---

## 3. New / changed mechanisms in DeltaServe-vLLM

3. **Fine-tuning admission**: walk me through the code path from a
   fine-tuning request arriving to its tokens being attached to a
   prefill batch. Name the relevant files / classes / methods. What
   inputs does the admission decision read, and what does it produce?

4. **Scheduling**: is DeltaServe-vLLM a new top-level scheduler, or a
   set of hooks into vLLM's existing scheduler? If hooks, which vLLM
   decision points do you intercept (sequence selection, batch
   assembly, eviction, swap-out, anything else)?

5. **Anything entirely new that wasn't in the original DeltaServe.**
   List anything you built for the vLLM port that has no analog in
   the original code. Chunked-prefill awareness, priority queues per
   adapter, interaction with vLLM's PagedAttention block manager,
   anything else. Three or four bullets is fine.

---

## 4. vLLM-specific integration details

6. **vLLM version / release tag** you're working against (e.g.,
   `v0.6.x`, `v1`, a fork SHA). The integration surface changes a lot
   between versions, so I want to be precise.

7. **Where the hook sits.** A 5–10-line sketch is enough:

    ```
    [client] --> vLLM API server
              --> vLLM Scheduler   <-- DeltaServe hook here? what does it do?
              --> ModelExecutor    <-- DeltaServe hook here? what does it do?
              --> PagedAttention BlockManager  <-- DeltaServe hook? what?
    ```

   Just annotate where you tap in and what you intercept.

8. **CUDA graphs in vLLM.** vLLM captures CUDA graphs for decode (and
   piecewise for prefill in newer versions). When the batch contains
   fine-tuning tokens, does DeltaServe-vLLM force eager execution, use
   a captured graph, or fall through to vLLM's normal capture logic?
   Does the original two-mode (graph vs eager) estimator still apply
   here, or does vLLM's graph behavior change what the estimator sees?

---

## 5. Memory model in DeltaServe-vLLM

9. **KV cache.** Does DeltaServe-vLLM use vLLM's native PagedAttention
   block manager for KV storage, or does it bypass / replace it?

10. **FT activations.** Where are LoRA fine-tuning activations stored
    in DeltaServe-vLLM? Same block manager (extended with new page
    types), a separate allocator next to it, or somewhere else?

11. **Adapter weights.** How are LoRA adapter weights managed —
    vLLM's existing LoRA support, the original DeltaServe path, or a
    new path?

12. **The "unified pool" framing.** In the original DeltaServe, KV +
    adapter + FT activations live in a single paged pool. Is that
    still true in DeltaServe-vLLM, or do KV and FT-activations live in
    separate pools because of how vLLM is structured?

13. **Activation budget knob.** Is there still a single
    `max_saved_finetuning_tokens`-style knob driving backward graph
    capture, or has the knob shape changed in the vLLM port?

---

## 6. Backward execution

14. Is the backward pass still a separate GPU subprocess under MPS in
    DeltaServe-vLLM?

15. Does the per-layer cooperative pause still rely on the same
    shared-status contract from the original code, or has the
    signaling path changed (CUDA events, streams, shared memory)?

16. Is backward CUDA graph capture still done at a fixed shape, and is
    that shape configured the same way as in the original?

---

## 7. Experiments you have run

17. What configurations have you actually run end-to-end with
    DeltaServe-vLLM? Hardware (GPU model, count), base model
    (Llama-3-8B?), workload (Company X trace? `timeline_loose.csv`?
    `timeline_tight.csv`?), what was on the other side of the
    comparison (vanilla vLLM, LLMStation, FlexLLM, anything else).

18. For each configuration in (17), what are your headline
    measurements right now? P99 TTFT, P99 TPOT, fine-tuning tokens
    per second / total fine-tuning tokens over the window. Rough
    numbers are fine; I just need to know what's "real" vs "planned".

19. Anything in `DeltaServe/eval_plan.md` that you'd flag as
    obsolete or replaced now that the vLLM port exists?

20. Cold-start behavior. The original DeltaServe profiles offline
    before serving. Does the vLLM port still do that, or does it
    profile online from cold? If it still does the offline pass, how
    long does it take on your test rig?

---

## 8. Pointers I can read myself

21. Is there a `README.md`, `CLAUDE.md`, design doc, or in-tree notes
    file inside the DeltaServe-vLLM tree that I should read before
    pinging you again? A directory + filename is enough.

22. Are there any failing edge cases or known limitations you'd flag
    pre-emptively? If you know that the vLLM port doesn't yet handle
    speculative decoding, or chunked prefill, or multi-LoRA at scale,
    say so now and I'll keep those out of the contribution claims.

---

*Once these are answered, I have what I need to redraft the relevant
parts of the paper. I won't need you in the loop for the writing
itself.*
