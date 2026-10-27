# A13 exact-source H200 replay and concurrency diagnostic

**42/42 checked A→B→A replays passed across seven cases. Actual cross-graph kernel overlap was observed only for full decode B32.** The other six profiled cases showed no overlap; their concurrent-execution correctness remains unverified. No new performance score was measured.

App `ap-f1GBalg3lZ0sfNmXyjJ7bu`; accepted 2026-10-25T06:11:58.331258+00:00; stopped 2026-10-24 23:12:39-07:00; follow-up verified tasks=0 and browser live containers/sandboxes=0. No active job remains. One app/function call, no retries or replacement launch.

## Source and environment

Frozen source `mixed_fa3_decode_parallel_merge.py`, SHA256 `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0`, original source commit `7e302d2366b194c32c433f6d6f9cc7b0b59b3928`. Diagnostic harness/controller commit `5b7a61cd0556ab1d5b83c2ee0554934f8fabf2dc`. No candidate algorithm or dispatch changed.

Runtime: NVIDIA H200, capability 9.0; Python 3.12.3, Torch 2.14.1+cu130, CUDA runtime 13.0, Triton 3.8.0. Reused Modal image `im-cPTbGpNiX3RarKResPgX7h`. Exact eleven-file allowlist and official reference hashes are in A13-ownership-modal.json and A13-ownership-result.json. Source provenance and license notices accompanied the bundle; no repository, credentials or unrelated files were uploaded.

## Coverage actually executed

Each case used two independently captured graphs, separate input/cache/output tensors and separate default private graph pools. Each graph replayed A1→B→A2 at the same addresses. Both arms changed actual sequence lengths between A and B; A2 restored A exactly. Each replay passed official output/new-cache checks, finiteness checks, and untouched-row/immutable-input checks. All 14 restored-A comparisons were bitwise equal for output and new K/V entries.

| Case | B/L | Requested ctx; actual length range | Page / scale | Checked replays | Cross-graph overlap |
|---|---|---|---|---:|---|
| tiny_full_verify2_local_ticket | 2/8 | 1024; 80–3632 | 64 / 0.015625 | 6 passed | Not observed |
| sliding_decode1 | 1/1 | 4096; 3841–4142 | 1 / 1.0 | 6 passed | Not observed |
| tiny_full_verify1_retained_fa3 | 1/8 | 1024; 80–2711 | 64 / 1.0 | 6 passed | Not observed |
| sliding_decode16 | 16/1 | 4096; 3798–4399 | 1 / 1.0 | 6 passed | Not observed |
| sliding_decode32 | 32/1 | 4096; 3793–4399 | 1 / 1.0 | 6 passed | Not observed |
| full_decode32 | 32/1 | 8192; 7586–8798 | 1 / 1.0 | 6 passed | Observed |
| nearby_full_page64_nonunit | 2/1 | 2048; 161–7265 | 64 / 0.015625 | 6 passed | Not observed |

The requested context is a fixture parameter, not a maximum: official mixed/agentic distributions produce the actual ranges above. These are reduced-context stress fixtures, not the scored ~80K workload. Sliding contexts still exercise the 2048-token window. Full B32 exercises its fixed-8192 chunk dispatch; tiny full B2/L8 exercises invocation-local tickets; full B1/L8 exercises the retained fused-prologue FA3 branch.

Output tolerance is exactly the official adaptive relative-L2 rule: 2.5× the FP8-query reference error +1.5%, clamped to 3–15%, plus per-row and finite-value checks. New cache tolerance is 8%. Actual oracle messages are preserved per replay. Untouched cache rows and immutable inputs are compared bitwise. These are real CUDA/Triton executions, separate from nine passing CPU orchestration/parser tests.

## What the overlap evidence establishes

For the first A replay pair of each case, Torch/CUPTI profiling captured only graph replay activity after both restores completed and before oracle work. Seven profiled pairs (14 replays) are included in the 42 total; the remaining 28 checked replays were unprofiled. Three warmup calls per graph and capture setup are excluded from the checked replay count.

Full decode B32 recorded four kernels with graph IDs. Distinct graphs overlapped for 133.313 µs total, with maximum two concurrent graphs on the same device. The observed overlapping A1 pair passed output/cache/immutability checks afterward. This is evidence for that particular pair and fixture, not general concurrency safety.

Other six pairs recorded kernels and graph IDs but no overlap. Merely enqueuing both streams is insufficient. The receipt therefore says `correctness_passed_overlap_unverified`, despite all correctness checks passing. It does not claim a concurrent pass for the L8 ticket path or FA3 path. No overlap in this profiled sample is not proof they cannot overlap without profiling.

Raw traces were temporary and deleted; the small receipt retains computed cross-stream/cross-graph overlap totals, counts, and only the first two kernel/overlap intervals. The two retained kernel records both belong to graph32; the other graph's records were truncated. The saved aggregate was computed from all four kernel events, but these retained intervals alone cannot independently reconstruct the entire overlap calculation. Profiler activity durations are diagnostic, not performance benchmark measurements. Same-graph simultaneous replay, intentionally shared capture pools/input/cache, broader full-length stress, DP head coverage and all retained FA3 dispatches were not covered here.

## Commands and outcome

```sh
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_native_ownership_probe.py -v
.venv/bin/python reviews/attention-v2/ownership_modal.py build
.venv/bin/python reviews/attention-v2/ownership_modal.py run
```

Local preflight: nine tests passed (0.043s), plus syntax, immutable source hash, resource-envelope and bounded-transport checks. The actual remote command used the frozen bundled `native_ownership_probe.py --worker --bounded-modal --candidate <bundle>/mixed_fa3_decode_parallel_merge.py --expected-sha c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0 --output /tmp/ownership-report.json --rounds 1 --deadline 75`.

Worker exit0, wall time35.706s; app lifetime43s. Those include fixture generation, compilation, reference checks and profiling, and are not kernel latencies. The full returned JSON is compact-row encoded (`dict(zip(record_columns, row))`); transport uses bounded compressed JSON when needed, with local SHA and decompressed-size validation. No large PTX/trace payload or new BlobCreate error occurred.

## Cost, shutdown and limits

Observed billing breakdown increased by $0.09426190; the app page displayed approximately $0.05 usage. Net billed cost remains $0. Billing may settle later. The conservative incremental bound based on the entire app lifetime is $0.078901888, below the authorized $0.25. Prior conservative cumulative $1.708917914 was preserved; new conservative cumulative is $1.787819802.

Rate basis: https://modal.com/pricing (H200 $0.001261/s, two CPU cores at $0.0000131/core/s and 8GiB at $0.00000222/GiB/s). Bound=(app lifetime+5s rounded up)×rate×1.1+$0.01. The $0 spend cap and applicable credits were verified in the existing Mac browser before launch; no financial setting changed.

Cleanup detail: the app was marked stopped at 06:12:39 UTC while the first read still reported one task. A cleanup `modal app stop ap-f1GBalg3lZ0sfNmXyjJ7bu --yes --env main` returned exit1; raw error text was not retained. This was not treated as proof of shutdown. A subsequent read verified stopped/tasks0, and the browser independently showed Finished, live containers0, live sandboxes0. No retry or replacement launch was made; shutdown uncertainty is resolved.

No OKC check or ranked submission, paid resources, new/renewed credentials, or algorithm edits. The single Modal authorization is consumed. No additional launch is authorized by unused dollars. Official attention score remains 1.1876787561×, projection remains 1.1732389990×. Next concurrency work would need a separately approved diagnostic designed to obtain overlap for the remaining paths; no such work was launched.
