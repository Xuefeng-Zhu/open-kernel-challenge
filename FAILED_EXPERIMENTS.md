# Failed and rejected experiment index

Navigation only: original sources, receipts and decisions remain authoritative.
**Slower** means a measured regression on the stated scope; **gate missed** means
a predeclared acceptance target was not met, not necessarily slower or incorrect.
**Inconclusive** is not proof of failure. Compile/setup errors and offline gates
are separate from numerical correctness. Unrun/parked work and active jobs are
excluded. This index is not a retry queue or a new selection decision.

## Projection

| Experiment | Outcome and reason | Takeaway | Original evidence |
|---|---|---|---|
| R06 — pooler split-8 | **Slower:** low-M pooler regressed despite passing correctness. | Do not compose this losing pooler component into the router gains. | [Comparison](COMPARISON_R01_R06.md) |
| M26 — QKV transpose layout | **Slower:** M256 regressed; broad layout replacement rejected. | Validate the largest affected shape before adopting a layout across M. | [Result](experiments/026-modal-qkv-layout.md) |
| M28 — unpooler transpose layout | **Inconclusive overall:** near-flat aggregate, with a limited M128 gain. | One useful shape does not justify replacing every path. | [Result](experiments/028-modal-unpooler-layout.md) |
| M30 — grouped pooler GEMM | **Slower:** large-M custom kernel lost decisively to the selected library path. | Compare against the actual selected path, not an assumed weak GEMM. | [Result](experiments/030-modal-pooler-grouped.md) |
| M32 — narrower pooler tile | **Slower:** BN32 did not rescue M30's large regression. | More blocks alone do not establish useful parallelism. | [Result](experiments/032-modal-pooler-narrow.md) |
| M33 — resource diagnostic | **Setup failure, recovered by M34:** worker CLI rejected its missing mode before diagnostics. | Check launcher/worker interface compatibility before allocating a diagnostic run. | [Failure](experiments/033-resource-probe.md) |
| M36 — QKV M128 full-K | **Slower:** removing split-K reduction left a much slower main kernel. | Count full-call cost rather than assuming reducer removal wins. | [Result](experiments/036-modal-qkv-m128-singlepass.md) |
| P038 — pooler M256 TMA | **Slower:** complete candidate regressed; ordinary-copy control was unmeasured. | This rejects the changed implementation, not TMA in isolation. | [Result](experiments/038-pooler-m256-pipeline-local.md) |
| P040 — O M1 grouped GEMV | **Gate missed:** correct, but the measured target gain missed the declared 15% goal. | A higher aggregate point does not override a failed target gate. | [Result](experiments/040-o-m1-group8-local.md) |
| P043 — lossless pooler codec | **Slower:** all compressed targets regressed despite exact decoding and correctness passes. | Separate decoder cost from the raw consumer's cost; later P062/P063 supply distinct controls. | [Result](experiments/043-pooler-lossless-codec.md) |
| P045 — broad output-column BMM | **Slower / gates missed:** small-M improved, but M128 regressed and broad gates failed. | Preserve the broad rejection; P046 separately validated the narrower composition. | [Result](experiments/045-pooler-column-batch2-local.md) |
| P049 — zero-bias library route | **Slower / gates missed:** unpooler regressed and combined benefit missed its threshold. | A favorable QKV route change does not transfer automatically to unpooler. | [Result](experiments/049-zero-bias-lt-route.md) |
| P052 — weight-first WGMMA | **Slower:** codegen changed as intended, but M8 lost to actual P046. | A better instruction path versus a custom control is not a win over the selected kernel. | [Result](experiments/052-orientation-execution-screen.md) |
| P053 — reducer-only rescue | **Gate rejected by native phase evidence:** the custom M8 partial alone was slower than complete P046. | Removing its reducer cannot rescue that fixed mainloop. | [Result](experiments/053-phase-cost-attribution-local.md) |
| P056 — native cuBLASLt interface | **Compile failure, recovered by P057:** NVCC failed before GEMM execution. | Fix the compilation boundary before judging algorithms or precision. | [Failure](experiments/056-native-cublaslt-interface-local.md) |
| P057 — restricted no-split tactics | **Slower / benefit unproven:** QKV was far slower; unpooler was near-flat or worse. | The restricted tactic subset did not represent the vendor ceiling. | [Result](experiments/057-host-compiled-cublaslt-interface.md) |
| P059 — native FP32 split-K | **Gate missed / benefit inconclusive:** recovered P046-level performance and its visible kernel family. | Beating the slow no-split control is not project progress. | [Result](experiments/059-native-fp32-split-policy-local.md) |
| P062 — matched raw P043 consumer | **Slower:** raw consumer missed near-baseline gates on all three targets; the stronger storage-screen rejection did not trigger. | A mixed deficit does not disprove lossless compression generally. | [Result](experiments/062-raw-consumer-official-screen.md) |
| P063 — first offline compile attempt | **Setup failure, recovered locally:** unnecessary driver discovery blocked GPU-free compilation. | Keep explicit-target offline compilation independent of device discovery. | [Failure and repair](experiments/063-offline-compiler-comparison.md) |
| P064 — decoded operand orientation | **Offline gate rejection:** registers fell, but decoded BF16 shared staging remained. | Removing a layout view is not evidence that underlying data movement disappeared. | [Compiler result](experiments/064-decoder-operand-orientation.md) |
| P065 — explicit direct-load consumer | **Slower:** compiler gate and official correctness passed, but all three pooler targets regressed. | Reject this prototype only; its matched Gluon raw GPU timing is missing, so compression-wide conclusions are unsupported. | [Result](experiments/065-explicit-consumer-layout.md) |

## Attention

Attention evidence is linked to immutable committed snapshots so this index does
not read or modify its active working files. The active check is not classified
here. Source snapshot: `04447c1cc8f64fdb489b74e1e942a64e0fae3cbb`.

| Experiment | Outcome and reason | Takeaway | Original evidence |
|---|---|---|---|
| Original attention v2 | **Compile failure:** non-constexpr `LOG2E` prevented validation; no numerical/timing result. | CPU equation checks do not establish target compilation. | [Failure][a-v2] |
| E1 — full-attention separate serial merge | **Slower:** weighted paired comparison regressed overall, including full B1 paths. | This rejects that serial merger, not independent merging generally. | [Result][a-e1] |
| E3 — loop `num_stages` | **Inconclusive:** tiny official movement did not demonstrate a pipeline benefit. | A source annotation does not prove useful overlap or a latency gain. | [Result][a-e3] |
| A09 — pipeline/replay probe | **Result retrieval failure:** `BlobCreate` 401 left GPU/compiler findings unavailable. | Missing evidence is not a numerical failure or proof the job never ran. | [Record][a-e3] |
| A2-10 — decode serial merge | **Slower / gate missed:** correctness passed, but sliding targets regressed. | Removing shared counters did not offset this merger's full-call cost. | [Result][a-210] |
| A2-12 — query hoist | **Slower:** target latency increased instead of meeting the 15% reduction goal. | Removing repeated producer work can lose once the extra preparation launch is counted. | [Result][a-212] |
| A2-14 — deferred denominator | **Slower:** full decode B32 regressed despite passing correctness. | Reject this full-column implementation, not every denominator schedule. | [Result][a-214] |
| Alpha-one branch variant | **Compile/resource gate rejection:** registers increased and V async buffering was lost; no runtime invocation. | Static resource changes cannot be reported as measured slowdown or occupancy. | [Compiler result][a-alpha] |
| Common-PV alpha-one | **Benefit gate missed:** correct, but zero of six target pairs improved and the required gain was absent. | Frequent `alpha=1` does not guarantee whole-call savings. | [Result][a-common] |
| Gluon overlap with nominal serial control | **Static codegen gate rejection:** both arms already prefetched K; no native timing followed. | Reject the claimed causal contrast, not overlap as a general mechanism. | [Compiler result][a-overlap] |
| Gluon V-stage publication | **Slower versus selected path:** improved over its prototype control but remained about 3.82× FA3 latency. | A local scheduling gain is not a production win. | [Result][a-vstage] |
| M64N8 integrated producer | **Compile/resource gate rejection:** sliding B1 violated the absolute zero-local-memory gate; runtime untested. | The selected control had the same local-memory count, so this is not evidence of newly introduced spilling. | [Compiler result][a-n8] |
| Sliding compact KV | **Inconclusive / not adopted:** correctness passed and the benefit screen missed, but a changed driver confounded historical timing comparison. | Do not turn worse descriptive timings into a matched causal slowdown claim. | [Result][a-compact] |

[a-v2]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/REMOTE_CHECK.md
[a-e1]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/REPORT.md
[a-e3]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/E3_PIPELINE.md
[a-210]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/A2-10-RESULT.md
[a-212]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/A2-12-RESULT.md
[a-214]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/A2-14-RESULT.md
[a-alpha]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/alpha-one-rescale-probe/RESULTS.md
[a-common]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/alpha-one-common-pv-probe/RESULTS.md
[a-overlap]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/gluon-overlap-probe/RESULTS.md
[a-vstage]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/gluon-vstage-probe/RESULTS.md
[a-n8]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/m64n8-producer-probe/RESULTS.md
[a-compact]: https://github.com/Xuefeng-Zhu/open-kernel-challenge/blob/04447c1cc8f64fdb489b74e1e942a64e0fae3cbb/reviews/attention-v2/sliding-compact-kv-probe/RESULTS.md
