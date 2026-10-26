# Independent results narrative review: PASS

Reviewer: `/root/offline_transpose_review`.

This review is bound to `RESULTS.md` SHA256
`e46c9ea1a887db8575a3cab20d62a5e17093b4badac1711c09f753e9f8fafa41`.
It checks that the narrative accurately describes the frozen one-tile offline
result and retains its limits. No source, compiler, container, service, GPU,
network or Git action was performed. Only this review file was written.

## Evidence checked

- `ARTIFACT_REVIEW.md` remains SHA256
  `af60a392d852c6b2c3f7a89635d680d33da399c752a8a198d08d28a0e02b9d2b`.
  Its independent PTX/SASS/dataflow findings support the narrative's native
  QK/PV M64N8, precision, reduction, publication/lifetime and zero-spill/local
  statements for this one unbuffered tile.
- The copied `evidence/compiler-manifest.json` matches the reviewed original,
  SHA256 `ca18c06e6aaf03f0721985c0e6e8ef86dc3c8f0b3e847215264302a8d6d451c9`.
  All six copied text artifact hashes match its digest table. The seventh,
  cubin, remains local-only and independently matches
  `449d38c546da620b172d6ba50782fd79ed0801022527d4023b5e5712c0155ac7`.
- Frozen kernel/helper still match `118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033`
  and `e7a489a899d7eae222250ca546d00bfefbb3435cf4ad25c8a79c96d0a099fc37`.
- `evidence/execution-receipt.json` SHA256 is
  `6dcbbec234bd4548a08a60b5a138d1593364c33f8953d4c37f29ae3f92d44685`.
  The source owner's receipt records exit0/removal of the Attention-only
  container, no network/GPU devices, pinned image, constrained mounts/resources,
  no host package installation/download and paid spending0. Exact Attention
  container absence was verified while the shared engine was running. The
  task did not request the engine's stop; a later read at2026-10-26
  23:24:03.548662UTC observed it Stopped, without inferring who stopped it.
  The revised narrative accurately says it was not restarted for the report.
  This reviewer did not independently query the service; closure and later
  engine state are attributed to the owner's receipt and direct observations.

## Narrative and prospective gates

The report clearly limits its result to compiler expressibility. It does not
claim GPU numerical correctness, useful asynchronous overlap, actual residency,
speed, a score improvement, full-producer admission or judge-binary identity.
The host compile/disassembly wall time is correctly distinguished from kernel
latency. The selected1.233633 source and A2-11 rollback are preserved.

The 32-register tiny-primitive count is explicitly **not** presented as a
reduction from the full producer's96-register count. The later96-register
parity/resource gate remains prospective and separate, with buffering/shared
limits still requiring review. The 18432-byte dynamic shared request and cubin
SHARED1024 field are retained separately; neither is treated as throughput or
occupancy proof. No unsupported32-versus96 performance inference was found.

The added warning about Gluon and `num_stages=3` is supported by a read-only
ZIP inspection of the pinned cached wheel. Its
`triton/backends/nvidia/compiler.py` SHA256 is
`e898d9a654e9e51aecb43b263fe61102f7162375c4275307fd1830d18b814eb8`.
Lines598–605 choose ordinary `make_ttgir` for Triton and the separate
`gluon_to_ttgir` for Gluon. Ordinary SM90 scheduling/pipeline passes appear
at288–291; the Gluon pass list341–365 does not contain those passes.
Gluon still has its own encoding/layout resolution passes, so this is not
a claim that it has no compiler transformations. Matching the stages option
alone cannot establish identical K/V buffering or software pipelining.

The four-path coverage and3.03%/20.20% figures agree with the predeclared
`EXPERIMENT_CARD.md` scenarios. They are labeled conditional complete-call
model arithmetic, not forecasts or recovered latency. The next integration
requires the original full loop, FP32 online carry, metadata reads, tails,
last-split ownership, buffering and partial/merge ABI. It explicitly retains
the untested N16-tail issue and stops before an official run if integration
gates fail. No experimental success is claimed from an inferior custom control.

No narrative blocker was found for publication of this frozen result. This
review does not authorize or validate a later producer implementation, GPU run
or performance claim; those require their own source-bound evidence.
