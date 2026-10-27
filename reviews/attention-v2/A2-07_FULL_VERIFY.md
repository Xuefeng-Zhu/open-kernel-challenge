# A2-07 — full single-request verify, without RoPE

Parent A2-06 source58cf5cf/job05b52e7834064b14,1.1381117300x. Only full B1/L8
switches from separate preprocessing to the identical proven `_fa3_prepare_fused`
body specialized ROPE=False. New host adapter passes an unused qkv pointer instead
of allocating a RoPE table. All other helpers and11timedpaths are AST-preserved.
No change to FP8 precision, BF16 normalization rounding, softmax scaling, window
(-1,-1), causal semantics, scheduler metadata, cache values or final norm/gate.

Evidence: the fullverify1 profile still has five preprocessing calls (norm,
three casts, cache write). Prior A2-05 profile shows5.9us in those instrumented
calls, but summed kernels/PDL waits are not walltime. Whole target currently
35.401us adjusted. The full path has more attention work than sliding, so the
predeclared goal is>=10% normalized target latency reduction, not the B1sliding
15% threshold. Frozen-parent10/15/20% scenarios are saved separately in the CPU
report; none is an achieved result. Inspect all controls>5%regression and CIs.

Correctness gate includes the existing B1/L8 fullctx1,048,576 prerequisite,
page64, plus the80000context scored target. This exercises the newly specialized
ROPE=False kernel at page64 natively if it passes. Nonunit-scale/DP/explicitABA
GPU stress remains absent for the new path; CPU models cover those separately.
Same precision as previous fullFA3 parent; no new FP8 change here. Native compiler
and resource metadata remain unverified until receipt and available exposure.

Seven CPU/static methods pass: byte-exact reproducible source assembly, AST
helper/dispatch invariants, full hostargs, direct FP8/cache oracle, finite outputs,
page1/64, scales1/64, live0, TP8/DP and same-address changed-length A-B-A. CPU
scheduler sentinel verifies only forwarding, not valid native scheduler content.

```
.venv/bin/python -B reviews/attention-v2/test_full_fused_verify_dispatch.py
.venv/bin/python -B reviews/attention-v2/test_full_fused_prepare_component.py
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-07 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-07 --cycle2
.venv/bin/python reviews/attention-v2/assess_cycle2.py A2-07
```

Standalone source is reproducible with build_full_fused_verify.py into a fresh
filename; it refuses overwrite and pins the exact parent. One authorized check,
failures count, no concurrent GPU. No Modal/ranked/paid/newcredentials. Preserve
final allowance and reviewed private automatic pushes.
