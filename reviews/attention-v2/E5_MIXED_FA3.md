# E5 — our narrow full-verify FA3 dispatch

Problem: muse_attention_decode. Parent E2 final_attention.py, SHA256
7efbb370d2828effabdc7ed717d2e7d5785f9ee0da55380fad6e591adf7c8782,
official job7edded086c8a47f2, score0.7958818006×. E2 branch is preserved.
Candidate mixed_fa3.py, SHA256 708d3f06f46bfb24a7a43963022eb51426f3a68bb2155c77dea8e9b6ae255423.

Hypothesis: full verify B16/B32 spends339/589us in our E2 full-attention path;
replace only these two of12 scored shapes with the permitted official-image
FA3 pipeline. This is an implementation/backend dispatch experiment, not a
standalone FP8 precision sweep. No third-party competitor source is copied,
imported or executed. The public dispatch idea motivated the experiment.

Exact change: E2 kernel_fn is renamed _e2_kernel_fn, with AST-verified unchanged
body and all original helpers. A new wrapper selects FA3 only when window=0,
rope_theta=0,q_len=8,batch in(16,32). Everything else remains E2, including sliding
RoPE. Full attention has no RoPE by the official contract, so no new RoPE table
or cache is needed. Config/shape-only FusedQKNorm objects are cached; no values or
outputs derived from data are cached.

The adapter follows the current official baseline: FusedQKNorm; optional K/V
scale division; FP8 E4M3 byte cache write with reserved_skip_index=-1 (row0 live);
Q converted from BF16 to FP8 as required by FA3 with FP8 cache; qk_scale_factor
folded into softmax_scale; causal full-page attention with supplied lengths,
page table, query offsets and scheduler metadata; optional K/V descales;
BF16 output then official per-head RMSNorm/sigmoid gate. Native API imports are
the exact official baseline paths. Source attribution and Apache2 text accompany
this standalone file. Current official reference/checks/baseline bytes fetched
on2026-10-25 match our preserved contract exactly.

Prediction: only full verify B16/B32 should materially change; compare all12
shapes and six weighted loads (46sliding+16full per load), with noise/regressions
explicit. No performance claim before real H200 results. Unchanged controls can
still vary across separate runs. Do not infer a gain from CPU timings.

Local gates: test_mixed_fa3.py, four tests passed. AST proves unchanged E2 functions;
exhaustive shape/config dispatch; CPU stand-ins verify native-call shapes/dtypes,
causality/scale arguments, numerical output/new cache oracle, row0/page1/page64/
scale1/64, untouched rows and same-address A/B/A with changed actual lengths.
CPU tests do not execute native FA3, FusedQKNorm, store_cache or a CUDA graph.
Native target-path page64/non-unit scales and explicit GPU ABA remain unverified
unless a later receipt provides that evidence. Published scored paths are page1,
scale1; official prerequisites may not enter the new B16/32 dispatch.

Prospective official command: .venv/bin/python reviews/attention-v2/check_cycle.py send E5
Monitor exact accepted ID with check_cycle.py poll E5; never duplicate uncertain
POST. User attention budget2remains before this candidate; reserve1for necessary
correction/final verification. UTC reset gave20daily checks but did not reset
user authorization. Projection's8remaining checks are separate. One coordinator
owns local-notes/gpu-slot.local.json; GPU work is serialized across both tracks.

No Modal launch: cumulative conservative guard stays$1.708917914/$2,netpaid$0;
previous A09 BlobCreate401 remains unresolved. No new/renewed credentials,
spend-cap change, push, ranking entry or paid resource. E2's unchanged shared
ticket path still has a simultaneous-call limitation; no concurrency pass claimed.
