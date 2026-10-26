# Independent corrected V-stage compile-harness review

**PASS for this local compile-only harness; no concrete blocker found.** This
does not admit native execution, numerical correctness, timing or official
quota. The parent still must freeze the exact bytes, run the real committed-
head preflight, verify fresh live paid0/free-credit guardrails and own the
serialized slot before one compile job.

Only local CPU source/configuration, test and transport helpers were exercised.
No Torch/Gluon/CUDA driver import, GPU, service, network, credential or Git
subprocess occurred. Only this review and the source review's card/dtype
binding were written; worker/kernel/controller and old evidence were not edited.

## Exact pair and corrected pointer contract

The new worker compiles exactly `v_after_wait` and `v_before_wait`, with fixed
B32/HKV2/HQ16/M64/N64/D64/page1, stride86008, four warps/one stage and the
same scalars/geometry. V_BEFORE_WAIT is the sole option delta. It calls
`attention_vstage_kernel.warmup` and initializes module handles; there is no
kernel invocation, benchmark or correctness run. Typed one-element pointers
are deliberately undersized and must never execute.

The final eight pointer types are BF16, E4M3, E4M3, **int32 PageTable, int32
HistoryLengths**, FP32, FP32 and BF16. The integer types agree with the actual
official generator at repository-relative
`reviews/attention-v2/official/reference.py:188/239`. A dedicated AST test
asserts all eight types. The old
standalone probe's int64 PageTable specialization is preserved as historical
non-production compile evidence; it is not silently relabeled as this one.

The source/card retain common raw-K prefetch before both waits. Independent
source review and nine CPU tests cover matched math/masks/volume, causal tails,
FP8 rounding/BF16 widening and two-slot lifetimes without artificial dependency,
volatile or new barriers. Actual native V-stage contrast remains a future gate;
identical relevant schedules permit no execution/benchmark. Compiler hoisting
is documented honestly rather than forcing a slow control.

## Review against the prior harness

Direct local diff against the previously reviewed harness shows only new
source pins, selector/API labels, result paths/experiment descriptions,
prospective mechanism wording and the int32 pointer correction. Resource,
credential selection, billing/slot, watchdog, exclusive ledger, transport and
shutdown logic remain the reviewed logic.

Resource gates are zero spills/local bytes/instructions, total shared<=64KiB,
and v_before_wait CUDA-permitted maximum CTA/SM>=v_after_wait using actual
device/function limits. Unknown metadata or SASS is inconclusive; resource pass
is not achieved residency or machine-code ownership proof. Native dataflow,
common-K/matched work, publication and PV drain need independent artifacts.

The default worker checks source hashes and CPU configuration only. Explicit
native `--compile` verifies installed Triton3.8.0 and the same three inventoried
Gluon/Hopper/async-copy files before vendor imports. There is no fallback,
installation, precision rescue or dispatch change. Compilation completion and
mnemonic inventories never auto-admit execution/performance.

The controller uploads six exact allowlisted files (kernel/config/resources/
worker/two licenses) and hash-checks its committed local file too. No credential
or receipt enters the closure-free remote payload. Fresh source-bound slot,
existing protected authentication, paid cap0 and free credits above$.40 reserve
are required. Bounds remain90/100/115/20/140 seconds, one H200/2CPU/8GiB,
one container/invocation and retries0. Uncertain outcomes require reconciliation,
not duplicate launches; explicit stop and stopped/tasks0 checks are retained.

Artifact names are the twelve flat V-arm names. Both arms require TTGIR/PTX/
CUBIN; optional TTIR/LLIR/SASS transport does not make missing ISA a mechanism
pass. Strict gzip/base64 framing, hashes/sizes, duplicate-key/allowlist checks,
2MiB report/8MiB raw/4MiB per-file bounds and exclusive flat persistence remain
unchanged. No archive path or credential expansion was introduced.

## Independent local verification and final binding

After the dtype correction, all22 native methods passed independently:
10 worker/resource tests plus12 controller tests. The nine unchanged source
tests also passed during source review, for31 local methods total. The actual
corrected worker default CPU preflight and source/config helpers passed.

Actual worker artifact packaging round-tripped through the actual controller
decoder/completion validator for both six required files and all twelve formats.
All six real payload files plus controller passed the parse/hash binding helper
with Git responses mocked from current bytes and auth/network helpers patched
to fail. No actual Git or remote preflight was substituted by this test; the
parent's real frozen-head check remains required.

The final card's int32 wording matches the corrected worker and original input
contract. No working selection, old source/artifact, official score or prior
failed-control interpretation changed.

| Reviewed artifact | SHA256 |
|---|---|
| `../kernel.py` | `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e` |
| `../config.py` | `686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0` |
| `../test_source.py` | `156508501f5d48a4f0845caccffa5fee2189b2080b902f2f0689e7382f66824b` |
| `../EXPERIMENT_CARD.md` | `209082108de413aeefb114c963096519451f60908934e12ea9aac15c30fefc1d` |
| `worker.py` | `19b135581617337b159944e000425d34fe9e0a13a4c7037a2b982f886d02a254` |
| `resources.py` | `1590dcf51f1c0ce59419d9d90ee397747556e50af8c2fb773f95b5a2fa5f1c59` |
| `controller.py` | `4d81fbd625f3674a8929742581e23d5e3738e99e6fc6afe79c35b336da601585` |
| `test_worker.py` | `493b0add8bf4da1b4a0111782048f7e7d330ddb56009a7861804a977c01d6050` |
| `test_controller.py` | `52462142b26ce74a14d1fdea93c2cc3ea4554ec86e9a0bc5294a07fc2c18a33c` |
| `CONTROLLER.md` | `4cc55c9ac5e9cd6e5449724bf4cf72c1ec3b124383019185ebc6174a76eb59d0` |
