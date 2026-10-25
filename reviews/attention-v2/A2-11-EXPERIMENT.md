# A2-11: reduce merge splits in parallel

Parent: A2-10 `mixed_fa3_decode_merge.py`, source a0e49b48ed456536a5e72db7d5ee5bb767aa1823, job08235841c4044048, passed16+12 at1.1405486256x. Preserve A2-09 counter-local baseline1.1466598821x and A2-08 sequential-only best1.1557671624x for comparison.

Candidate: `mixed_fa3_decode_parallel_merge.py`, SHA256 c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0.

Problem: separate merge profiles take6.6/4.8/3.2us for sliding decodeB1/B16/B32 and8.0us for fullB32. These are instrumented activities, not calibrated walltime or hardware counters. The measured cost warrants one focused reduction change.

Change: one merge CTA per valid query head; load the existing FP32 partials, find the common maximum, and reduce weighted numerators/denominators across splits in parallel. Use actual SPLITS for scratch strides and padded BS only for lanes, masking missing lanes to(-inf,0,0). Keep value scale, RMSNorm, gate and BF16 output. Exact partial computation, partition, cache writes, precision, widths, warps/stages, all FA3 and L8 paths remain unchanged. This explicitly reorders FP32 sums/rescaling; it is not a bitwise-equivalence claim.

Affected scored paths: sliding decodeB1/B16/B32 and full decodeB32. Expected-value test: normalized sliding reductions at least15/10/3%; fullB32 no more than1%regression. Inspect aggregate plus controls and cross-run intervals before promotion. No assumed register or occupancy improvement.

Sensitivity model, not prediction: holding all other A2-10 times fixed, hypothetical1/2/3us savings on each of four paths gives scores1.152056/1.164041/1.176546, respectively +1.009/+2.060/+3.156%. Instrumented merge activities cannot be subtracted directly from whole-call medians to claim an achieved upper bound.

Local: `.venv/bin/python -B reviews/attention-v2/test_decode_parallel_merge.py` passes8tests. Exact AST invariants plus FP32/FP64 models cover signed cancellation, empty/padded splits1/2/5/9/11/16/66 and64Dscale/RMS/gate. Models do not reproduce GPU FMA/exp2/reduction layout and cannot validate native accuracy/performance. No candidate import or GPU execution locally.

Native plan: one committed-file official off-board check, serialized after all prior jobs terminal. Keep all12timings and six46*sliding+16*full loads; compare against both A2-10 and A2-09, and qualify A2-08's counter-sharing constraint. Nearby L8 still uses safe per-invocation counters. Dedicated concurrent-stream/current-path ABA stress remains unrun; the prepared harness is separate evidence. No Modal, paid compute, credentials, ranked entry or competitor source execution.
