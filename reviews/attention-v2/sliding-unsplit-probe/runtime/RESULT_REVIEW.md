# Independent result review: reject the unsplit candidate

The installed API, observed extension identity, native correctness and combine
elimination gates passed. The complete-call performance gate failed on both
targets. This is a measured rejection of this candidate, not an API/identity
blocker, permission denial or incomplete run. No official check is justified.

The reviewed raw result SHA256 is
`46aed5e3fba5b978d3206de15106ce7bd5961fd63b1bc8e76ac887073fb4f0ab`.
It agrees with the single app's recorded transport hash, source freeze
`5e494062a91a643df674887140125b9333f2fb92` and exit0. Every allowlisted payload
hash was independently checked against the committed bytes and current files.
The candidate remains
`586074a4bbff57ff28c53070dfcdfcf881631b9aeffeb016032a7a07b32309cf`,
worker `953537670113411a0de1445864bbaa52d4b3cfecb276da5a01ca5fdf0ee6d812`
and identity helper
`46fd1727b4744ce74f4eb4514db704710451dffba527d28abefe2c8800e9e4a0`.
The worker completed in 48.876s with all native gates complete and
`official_check_admitted=false`.

## Installed identity and actual mechanism

The installed callable is
`sgl_kernel.flash_attn.flash_attn_with_kvcache`, with an explicit
keyword-capable `num_splits=0` parameter and the same callable in both arms.
The returned wrapper source explicitly forwards the parameter to
`torch.ops.sgl_kernel.fwd.default`. Its complete 16160-byte source text hashes to
`610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`;
the retained function text hashes to
`6bbd808ae807ede41ae6d93d61f405d91301d6eed05bd13828379e1e734f87bd`.
The reviewer recomputed both text hashes.

The native worker fully hashed the one observed relevant extension,
`/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so`,
1122619328 bytes, SHA256
`cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`.
The reported total 1122635488 bytes equals the wrapper plus extension size;
there were no omitted observed extension paths or unavailable hashes. This
establishes the recorded observed-file identity, not discovery of every
transitive GPU binary. The local reviewer verified the authenticated result
and hashing code; it did not independently fetch the remote binary.

The single separate activity session contains four attributed graph/arm
replays. Independent recounting confirms the same structure on B16 and B32:

| Observed stage | Control count | Candidate count |
|---|---:|---:|
| FA3 mainloop | 1 | 1 |
| FA3 combine | 1 | 0 |
| Varlen scheduler preparation | 1 | 1 |

Scheduler preparation therefore persisted. Every raw full kernel name and
interval remains retained, with no unassigned activity or truncated names,
and all four profiled replay checks passed. The `other` count 3 in each group
includes one CUDA-side `UNSPLIT_MECHANISM_...` marker annotation plus the fused
preparation and gate kernels. It must not be described as three physical
kernels. Excluding that annotation, the known physical kernel count is 5→4.

Instrumented mainloop activities increased from 13.855→25.663µs for B16 and
17.695→26.079µs for B32. These are one-replay profiler observations, separate
from benchmark samples; sums or subtractions do not predict whole-call latency.
The full symbols identify changed specializations and a varlen dynamic
persistent scheduler. No CTA grid, occupancy, residency, hardware counter or
stall cause was measured. Lost split parallelism is a plausible explanation,
not an established bottleneck or causal hardware diagnosis.

## Correctness and clean timing

Independent validation checked all exact case/version sets and record counts:

- 12 preparation groups, containing 36 exact Q/K/V byte comparisons, finite
 queries and immutable/untouched cache checks.
- Seven complete-call cases: both targets in scored, soft-boundary and valid
 officially planted edge families, plus the unchanged full-B16 verify control.
- 42 A→B→A unchanged official oracle calls, including 14 exact return-A checks,
 and 42 cross-arm cache-byte comparisons.
- Four additional passing oracle calls from two disjoint-cache independent
 graph pairs submitted on separate streams; actual hardware overlap unmeasured.
- 96 checked warmup/measured replays across the two targets and control.

All those gates passed before clean timing. Each family retained four warmup
and twelve measured two-arm samples, with same-input reverse-order pairs.
All replay addresses, immutable inputs, untouched rows and own-source expected
output/new-KV bytes passed. Attention output equality between split policies
was not required; original oracle tolerances were preserved.

The reviewer independently recomputed every paired ratio and paired arithmetic
mean saving. Positive saving means baseline minus candidate time:

| Family | Raw pooled median µs, baseline→candidate | Median paired mean saving µs | Median paired candidate slowdown | Positive reverse-order pairs |
|---|---:|---:|---:|---:|
| Sliding B16 L8 | 27.088→35.568 | −8.528 | +31.4944% | 0/6 |
| Sliding B32 L8 | 32.992→35.808 | −2.600 | +7.8550% | 0/6 |
| Unchanged full B16 L8 | 204.416→203.776 | +0.456 | −0.2237% | 5/6 |

Both targets fail the predeclared ≥2µs paired saving and 6/6 positive-pair gates.
The unchanged control passes its ≤5% slowdown limit. The paired values,
not subtraction of pooled medians across fixtures, determine admission.
One session supplies no independent-session confidence interval or official
score, but the uniformly unfavorable target pairs suffice to reject this
candidate under its predeclared rule. Fewer launches did not yield a useful
complete-call improvement.

## Closure and review scope

The separately retained reconciliation at 20:53:15.259613UTC records
`ap-SoIIp01s2fzbOD9KKGSw9W` stopped/tasks0 and all listed apps stopped/tasks0.
Resource and cost accounting are owned by the parent and remain separate from
the native result. No official check should be consumed, no retry is implied,
and the validated fusion baseline and selected A2-11 remain preserved.

This independent review used local retained files and committed source only.
It edited only this report and made no CLI/service, network, browser, GPU,
credential or relaunch action. A second local reviewer separately confirmed
the installed identity and activity counts.
