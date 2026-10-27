# A2-12: prepare repeated query once for sliding decode1

Parent: complete A2-11 `mixed_fa3_decode_parallel_merge.py`, source7e302d2366b194c32c433f6d6f9cc7b0b59b3928, jobcfa8a8670e394f80,1.1876787561x. Preserve this selected best.
Candidate: `mixed_fa3_decode_qhoist.py`, SHA256 45003f03d471d484f5451fa35b534ced9819a886a7be8dc6c3bed339d5b45597.

Source evidence: sliding B1/L1 launches2KVheads*16splits. Each of16valid query heads therefore repeats normalization/RoPE16times. A2-11 instrumented producer8.1us is82.4%of GPUactivity; this does not establish what fraction is query preparation. Work in separate CTAs executes concurrently, so repeated work is not directly removable walltime.

Only B1,L1,window2048,rope500000 changes: one timed same-stream preparation launch writes a per-call BF16 Qbuffer(2KiB forTP8). Use the exact `_norm_rope` helper and seq_lens position; BF16input->FP32norm->FP32computed-trigRoPE->FP32qk_factor->BF16. Consumer loads that value into unchanged DOT. Keep original block_m16,4warps/stages3; no table/vendor norm/FP8Q or moved factor. Cache update/new-token ownership, partition, partials and parallelmerge remain unchanged. Other11scoredpath computations are unchanged via constexpr false branch, though compiler lowering still requires nativevalidation.

Declared target: at least15%normalized whole-call reduction, INCLUDING the extra launch. Freeze-other-shapes arithmetic gives modeled score1.2049953407x; this is not achieved performance and small aggregate changes can overlap CIs. Stop this mechanism if launch/buffer cost outweighs benefit; do not claim saving16times its latency or start arbitrary tuning.

Local command `.venv/bin/python -B reviews/attention-v2/test_decode_qhoist.py` passes7tests. They prove source invariants/guards and illustrate rounding/fresh-input sensitivity. Separate compilation can change FP32 reduction/FMA behavior at BF16 boundaries. No Triton/nativeGPU execution locally. Buffer is rewritten every invocation/graph replay and retained through consumers; no persistent input-derived state.

Native plan: one committed-source off-board check, after A2-11 is terminal. Native output/cache/prerequisite/freshgraph checks and all12timings; compare against actual A2-11 with six weighted loads. Bespoke independent-graph stress remains unrun; no Modal/paid/newcredentials/ranked entry. If rejected, keep A2-11 and preserve this negative result.
