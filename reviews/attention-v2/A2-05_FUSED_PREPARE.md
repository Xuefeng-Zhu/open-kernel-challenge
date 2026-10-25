# A2-05 — one fused prologue, only B1 sliding verify

Parent selected `mixed_fa3_decode_small.py`, source28991e7/SHA5ef5c9e9,
job66b44d7586184708, score1.0770658900×. Candidate
`mixed_fa3_fused_prepare.py` preserves all parent helpers and all other11scored
paths; one additional dispatch calls a new fused prologue before the unchanged
FA3+output norm/gate pipeline. This is not a new attention algorithm or precision
sweep, and no competitor whole source is executed.

Problem: the parent's current verify1/sliding profile has separate QKnorm/RoPE,
three FP8 conversion calls and cache writeback. It also has FA3 schedule/fwd/
combine and output gate, which remain. The separate instrumented profile is
recorded in A2-05-cpu-preflight.json (20.8µs summed activity,19.4µs GPUbusy);
neither is the official23.389947µs adjusted median. PDLoverlap means the profile
rows cannot simply be subtracted to forecast fused walltime.

Hypothesis: fusing only the prologue cuts reference-normalized target latency
at least15%, while all11controls remain unchanged. A20%scenario is also recorded,
with six-load arithmetic freezing other parent times. No predicted run result.
Reject if native correctness fails or targetcut<15%; inspect>5%control shifts
and aggregateCIs separately. No metadata tuning/shape sweep on failure.

## Exact numerical and ownership contract

- FP32 per-head RMSNorm, then BF16 rounding BEFORE interleaved RoPE, then BF16
  rounding after RoPE, then FP8 E4M3 query. Query factor stays solely in FA3's
  softmax scale. No FP16/TF32/FP4 lowering.
- Positions are loaded from supplied GPU positions; cos/sin loads reuse the
  unchanged config-only2^21float32table. No input-derived cache or host sync.
- NewK follows the same BF16 norm/RoPE order. Scale division is BF16-rounded
  before FP8, matching PyTorch BF16/scalar-FP32 promotion. V stays raw BF16
  before scale/FP8. K/V use direct FP8 casts, without an extra saturation operation. The earlier
  preflight incorrectly attributed an explicit clamp to the official reference:
  `_quant` actually uses a direct cast. The added clamp was removed before any
  native admission. CPU Torch 2.14.1 saturates the chosen overflow examples; that
  environment-specific observation does not establish CUDA conversion behavior.
- Qrows and K/Vrows have separate masks. Cacheloc0 is LIVE, unlike vendor
  FusedQKNorm's optional fused KV-store path; use0<=loc<capacity, not loc!=0.
  Every required Q element and newKV entry is written once, no old row written.
- Q is invocation-local FP8 storage; global table is readonly config data. The
  producer/FA3 kernel boundary uses ordinary stream/graph dependencies, with
  no atomic tickets in this new path. Fallback E2 concurrency limits persist.

RMSNorm/RoPE component is adapted from pinned official SGLang source, Apache2;
see THIRD_PARTY_NOTICES.md and standalone embedded license. Source has no harness
imports, file/network access, custom timing, persistent output or cached inputs.

## Local gates and exact source assembly

```
.venv/bin/python reviews/attention-v2/build_fused_prepare.py --parent mixed_fa3_decode_small.py --expected-sha 5ef5c9e93b56ca382aea363d0022d558a8163f5df0ef8ae5673ba1a854bbf97b --output mixed_fa3_fused_prepare.py
.venv/bin/python reviews/attention-v2/test_mixed_fa3_fused_prepare.py
.venv/bin/python reviews/attention-v2/test_fused_prepare_component.py
```

Assembly refuses an existing output, so use a fresh reproduction directory or
compare its bytes instead of overwriting a preserved source. Shape-only head
count was made explicitly constexpr before first native admission. Exact final
source/component/builder hashes are in A2-05-cpu-preflight.json.

Three complete-file+five component tests pass on CPU. They prove AST preservation,
host arguments, mathematical oracle/cache cases, symbolic ownership, CPU FP8 overflow behavior,
page64/nonunit/live0/untouchedrows and same-addressA→B→A length/content changes.
Launch stub models math; it does NOT execute the Triton kernel. Initial suite
had a test-only ImportError for a projection-branch validator; replaced with
stdlib scope analysis, no kernel bug or GPU failure claimed. New JITbody has no
free numeric globals. CPUtableextent8192; realcandidatebound2^21unchanged.

Native compilation, resource usage, CUDAgraph correctness and performance are
UNVERIFIED before the official receipt. Exact native Torch/Triton versions are
not exposed in receipts; retain official evaluator/image pin and GPU/driver,
separately from local Python/Torch versions. Explicit target nativeGPUABA,
page64/nonunit/untouched-row/concurrent stress remains absent unless provided.

After source commit, one check from the remaining6:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-05 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-05 --cycle2
```

All priorjobs terminal, no concurrent measurement; reserve final allowance.
No Modal/paid/newcredentials/terms/ranked entry. Reviewed private auto-push only;
never retry an uncertain upload or alter comments solely to evade deduplication.


## Resume audit before first admission

Preparation commit451b475 was preserved. Its original explicit clamp and claim of
reference saturation were corrected locally before any upload. Direct FP8 casts
now match the parent; all other candidate math, dimensions and dispatch stay fixed.
The CPU oracle now asserts finite reference tensors before claiming correctness.
No GPU correctness or timing was inferred from CPU overflow behavior.

The same task resumed after the model-transition request. The configured model is
not exposed by available tools. Authentication and terminal prior jobs were read
again; quota14attention/17projection, authorized budgets6/5 at the resume snapshot.
The new source SHA and test evidence below are authoritative for first admission.
