# Common-K V-stage compile result

The new prospective publication-scheduling gate passes independent static
review. This is **not V-load serialization**: both arms prefetch raw K and V.
The surviving difference is conversion completion and next-slot shared
stores/publication relative to the current QK drain. Correctness, actual
device overlap and speed were not measured by this compile stage.

Freeze `048bd62de599831edbb40b118df9a6047f6f1043` binds kernel SHA256
`6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`.
Raw compile result SHA256 is
`258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`.
Thirty-one CPU checks, source/harness review and actual committed-hash preflight
preceded the two fixed warmup compiles. Both PageTable and history pointers
use the official int32 ABI. Neither kernel was invoked.

| Resource | v_after_wait | v_before_wait |
|---|---:|---:|
| Registers/thread |166|168|
| Shared bytes |32768|32768|
| Spill/local memory |0|0|
| CUDA legal maximum CTA/SM |3|3|

These are allocation/driver limits, not achieved residency. Both arms emit
eight BF16/FP32 WGMMA instructions and two commit/wait groups. Ten artifacts
retain TTGIR, LLIR, PTX, CUBIN and SASS per arm; Gluon emitted no TTIR.

The normalized PTX bodies match after removing only the first QK wait.
SASS demonstrates the actual remaining contrast:

- `v_after_wait`: QK drain at0x2fb0; next-slot shared stores at0x3440–0x48b0,
  publication fence at0x48d0.
- `v_before_wait`: next-slot stores at0x30c0–0x3820 and fence0x3860 precede
  QK drain0x3880.

Independent [mechanism review](native/NATIVE_MECHANISM_REVIEW.md) and
[artifact review](native/ARTIFACT_REVIEW.md) trace operand pointers, actual V
values, two8192-byte K slots, the separate V bank, dependent score use and PV
drain before reuse. The earlier strict-serial experiment remains failed and
unchanged; this new card allowed documented partial hoisting.

App `ap-XzBXIhHnYx5BgeBYIoyAkL` was independently stopped/tasks0 at
2026-10-26 00:10:53.666428 UTC. Child wall6.503754s, client16.666517s;
worker/child/function/startup/watchdog bounds90/100/115/20/140s. At the
00:14:23.579875 UTC cutoff, with no new other app IDs, delayed detailed
workspace credit usage increased$0.01044102, paid$0. This is not an exact
per-app invoice; the free-credit reserve was$0.40.

This supports only the separately reviewed, bounded correctness-first screen
in `runtime/`. Its adapter must retain preparation, FP8-Q widening, new KV
stores and norm/gate inside the captured/timed call and compare against the
actual unchanged FA3 working path. No production dispatch, official check,
score claim or promotion follows from static review.
