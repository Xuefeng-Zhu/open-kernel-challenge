# Failed and rejected experiment index

Short navigation only; original evidence remains authoritative. **Slower** is a measured regression on the stated scope. **Gate missed**, **compile/setup failure** and **inconclusive** are different outcomes. Prepared or active work is excluded. This is not a retry queue.

## Attention

| Experiment | Outcome / reason | Evidence |
|---|---|---|
| Original attention v2 | Compile failure; non-constexpr `LOG2E`, no native correctness or timing | [Result](reviews/attention-v2/REMOTE_CHECK.md) |
| E1 serial merge | Slower in the weighted comparison | [Result](reviews/attention-v2/REPORT.md) |
| E3 loop stages | Inconclusive; tiny score movement did not prove overlap | [Result](reviews/attention-v2/E3_PIPELINE.md) |
| A09 pipeline/replay | Result retrieval failed; missing findings are not a numerical failure | [Record](reviews/attention-v2/E3_PIPELINE.md) |
| A2-10 decode serial merge | Slower sliding targets despite correctness pass | [Result](reviews/attention-v2/A2-10-RESULT.md) |
| A2-12 query hoist | Target slower; extra preparation launch counted | [Result](reviews/attention-v2/A2-12-RESULT.md) |
| A2-14 deferred denominator | Full decode B32 slower | [Result](reviews/attention-v2/A2-14-RESULT.md) |
| Alpha-one branch | Resource/buffering gate rejected; not a measured slowdown | [Compiler result](reviews/attention-v2/alpha-one-rescale-probe/RESULTS.md) |
| Common-PV alpha-one | Correct, benefit gate missed | [Result](reviews/attention-v2/alpha-one-common-pv-probe/RESULTS.md) |
| Gluon nominal serial control | Codegen contrast invalid; both arms already prefetched K | [Compiler result](reviews/attention-v2/gluon-overlap-probe/RESULTS.md) |
| Gluon V publication | Faster than its prototype control, still about 3.82× FA3 latency | [Result](reviews/attention-v2/gluon-vstage-probe/RESULTS.md) |
| M64N8 integrated producer | Official correctness passed; score 1.09955× versus then-control 1.23363× | [Result](reviews/attention-v2/m64n8-producer-probe/official-check/RESULTS.md) |
| Sliding compact KV | Benefit screen missed; historical driver change confounds causal comparison | [Result](reviews/attention-v2/sliding-compact-kv-probe/RESULTS.md) |
| User v3/v3b, original archive | Both compile failures; no benchmark results | [Errors](reviews/attention-v2/user-v3-probe/RESULTS.md) |
| User v3/v3b, revised archive | Both correct but slower than the then-selected source | [Results](reviews/attention-v2/user-v3-revised-probe/RESULTS.md) |
| v298 alternate merge mode | Correct; both L8 targets slower in paired comparison | Result (local evidence; `reviews/attention-v2/vjeux-v298-probe/merge-regime-probe/RESULTS.md`) |
| v298 L1 FP8 QK helper | Native instruction gate failed; no candidate numerical/timing invocation | Compiler result (local evidence; `reviews/attention-v2/vjeux-v298-probe/fp8qk-decode-probe/RESULTS.md`) |
| v298 static page banks | Correct, benefit gate missed; waits redistributed without net gain | Paired test (local evidence; `reviews/attention-v2/vjeux-v298-probe/page-bank-probe/RESULTS.md`) · NCU diagnosis (local evidence; `reviews/attention-v2/vjeux-v298-probe/profile-04-page-bank-paired/RESULTS.md`) |

## Projection

The Projection branch maintains its [concise failure index](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/99db6390fdf7c55f7b448963e91c4e1264991ab5/FAILED_EXPERIMENTS.md#projection). That saved index has 21 Projection entries and links each to its original evidence; no duplicate report is created here. Later work is listed in [Projection status](https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/d55496b422e996807614a2661346d589144bc1ba/STATUS.md).

[Current baselines](README.md) · [successes and profiling](EXPERIMENTS.md)
