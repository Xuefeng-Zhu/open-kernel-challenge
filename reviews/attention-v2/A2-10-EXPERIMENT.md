# A2-10: independent merge for decode fallback

Parent: A2-09 `mixed_fa3_local_tickets.py`, source ef5f09a8ad4fa01aac57845a9021e0a3846c578c, official job3fb683ba982d448c,1.1466598821x. That repair passed16+12 but normalized sliding B1/B16 time regressed7.40/7.68% against A2-08, supporting a focused attempt to remove initialization cost.

Candidate: `mixed_fa3_decode_merge.py`, SHA25694cfb257da13daddec1e8eeb5e7c0b9abe768f54029ecfe2bca8733f8088b93f.

Single architectural factor: only L1 E2 fallbacks write partials then use a same-stream independent merge. No completion counter or initialization on those calls. The merge copies the parent's serial split-order recurrence, final RMSNorm/gate and BF16 output. Historical partitioning, chunk capacity, last-split new-token writes, precision, widths, warps and stages stay unchanged. L8 fallbacks retain A2-09's invocation-local counter; all eight FA3 scored paths retain identical code. One-split calls retain direct output.

Affected scored paths: sliding decodeB1/B16/B32 and full decodeB32. Candidate uses partial+merge launches versus parent's initialization+fused launch; equal launch count does not imply equal latency. This is not a claim of lower registers, improved occupancy or measured overlap.

This differs from rejected E1: E1 left sliding unchanged and used old fullD32 five-split recipe; current fullD32 uses E2fixed8192chunks, usually11splits. Prior fullD32E1parity/slightloss and severe small-batch full losses are warnings, so this version preserves L8. No third-party competitor code is used.

Declared target: recover at least3% normalized latency on the affected paths; inspect each path and any >5%control regression. No automatic promotion from total point score alone. The measured A2-09 penalty motivates the test; it does not predict that merge will win. Frozen A2-08 model: uniform1us cost on four paths reduces total by1.032%,2us by2.022%; these are sensitivity calculations, not measured new-candidate results.

Local command: `.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_decode_merge.py -v`. Five tests pass: exact AST invariants for all unchanged code, exact copied recurrence, exact host edits, and120CPUfloat64 merge/empty-chunk cases. No Triton interpreter or GPU execution. CPU results do not validate BF16 tensor-core code or allocator/stream behavior.

Native plan: one official off-board check of this committed file; serialized after terminal A2-09. Standard output/new-cache/replay correctness plus all12timings and six weighted loads. Dedicated two-independent-graph/stream and current-path ABA tests remain unverified unless separately run and recorded. No Modal, new credentials, paid resource or ranked entry.
