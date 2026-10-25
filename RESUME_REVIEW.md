# Resume audit — 2026-10-24 21:09 PDT (2026-10-25 04:09 UTC)

The same existing task resumed. The model configuration is unavailable through
our tools; the parent controls the requested Astra selection. No replacement
thread or credential action was performed.

## Reconciled state

Read-only existing OKC whoami succeeded, jobs were all terminal and uncertain
requests zero before A2-05 admission. The first audit parser mistakenly expected
JSON from whoami; corrected to consume its text without printing personal data.
No upload occurred in that failed parser call. Platform quota was attention14/20,
projection17/20; user budgets were6 and5, separately. Snapshot:
reviews/attention-v2/RESUME_ACCOUNT_CHECKPOINT.json.

A2-05 accepted50465a4c440f4b75, leaving attention5 and projection5. Source535d602,
SHAa7930cda0389cf55f60759ea3f9247f0562bf3412fad00c81dcbfc48520f93de.
No other GPU operation was launched. Source451b475 is retained as initial local
preparation, never uploaded; the correction has a separate source commit.

## Critical review and correction

Official reference.py:114 uses a direct FP8 cast, not an explicit clamp. Earlier
A2-05 documentation and CPU evidence overclaimed a reference clamp. Removed the
extra clamp from the component and complete candidate, corrected attribution,
and required finite reference tensors in the CPU oracle test. Local Torch2.14.1
saturates tested overflow on CPU; no inference about official CUDA conversion is
made. All8CPU/static tests passed after repair; exact source assembly reproduced.

A second independent read-only review checked pinned SGLang rounding/order,
positions and live-row0 cache writes, masks, constexpr usage, scale arguments and
stream dependencies. No remaining definite in-contract blocker found. The model
launch stubs DO NOT execute Triton. All16 prerequisite cases bypass new B1L8
fusion: native page64/nonunit/scaled/change-length/untouched-row/concurrent stress
is unverified, even if the scored target later passes.

## Experiment question

Only B1L8 sliding fuses QK norm/RoPE, casts and cache stores. Preserve all11other
scored paths. Target at least15% normalized latency cut; reject if goal missed or
correctness fails. Sum-of-profile-kernels is not wall time because PDL overlaps.
Whole score uses six46*sliding+16*full loads geometrically combined; retain all
raw/adjusted timings. Existing parent1.0770658900x; target15% scenario1.0955944x
is a model, not a result. Compiled resource metadata unavailable until exposed.

## Preserved projection evidence

P039 d5636ac8d1404abd is selected,1.173238999x (CIoverlaps old aggregate).
P040 d57d89967d1a408f achieved1.175871327x but target cut6.51% missed15%; not
promoted. TMA6799753227ed463e lost, but ordinary-copy control was not measured,
so loss cannot be attributed to TMA alone. No fresh projection request sent.

## Git and resource safety

Projection branchaf7f149 is clean. Attention original local history remains;
sanitized export maps source commits and removes historical credential identifier
from an old log. Never push original attention-fa3-local. Promoted audit/push
helpers scan history, enforce existing private URL/visibility and fast-forward,
then verify remote SHA. No secret material or local-notes is staged. No new
Modal usage; cost guard$1.708917914/$2, prior apps stopped. Historical A09 report
retrieval failed401; it is not proof of absent execution and is not retried.

## Independent projection stopping review

Read-only second review supports preserving five projection checks. P040's
target-only frozen-parent effect is about+0.16% aggregate and missed its goal.
P038 copy control could explain a failure but is not a justified optimization.
QKV single-pass55.10us versus21.89us loses despite WGMMA/zero reported spills.
Unpooler layout's isolated m128 opportunity is about0.1% aggregate. Router m16
reaching an unmeasured7.5us from9.30us would model only0.45% aggregate. These do
not establish a new mechanism. Exact-target hardware evidence remains absent;
M34 metrics belong to other kernels. No projection request sent during resume.

Attention extension arithmetic is saved in FUSION_EXTENSION_MODELS.json. These
are conditional frozen-parent scenarios, not measured speedups or commitments.
