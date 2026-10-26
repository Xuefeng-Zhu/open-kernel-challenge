# Independent local B1 SIMT compile-harness review

**LOCAL PASS; no concrete harness blocker found. B1 has NOT been admitted.**
All new GPU launches remain on hold for the parent's methodology handoff.
This report establishes preparation readiness only: no native compilation,
instruction/resource finding, numerical pass, speedup or execution admission.

The reviewer ran six CPU tests and the actual default source/shared-merge
preflight: all passed. Only this report was written. No Torch/Triton import,
CUDA-library load, GPU, upload, service, credential, Git, commit or push action
was performed. Candidate/control and old evidence were not edited.

## Exact compile-only contract

The worker pins candidatef516 and selected baseline3da, plus both root license
files. Default behavior hashes these files and parses the shared merge AST;
it imports neither module. `_merge` source segments match exactly, SHA256
`2ded746d47fdd7f8c213d55e0db9ddddb50e6cb0ff9081c03e026adbf7ac629d`.

An explicit future --compile branch contains exactly three fixed warmups:
baseline `_fused`, candidate `_fused_b1_simt` and baseline `_merge` once.
Module-handle initialization and resource queries are distinct from invocation.
There is no bracket launch, kernel.run, capture, correctness or timing call.
Typed one-element pointers are undersized and must never execute.

The21 producer runtime arguments match both signatures:13 pointers followed
by theta500000, eps1e-5, QKfactor5.4730064863875, SMscale0.125, window2048,
page1, token stride2304 and PageTable stride80001. History/PT are int32,
cache_loc int64, QKV/output BF16, caches E4M3 and scales/partials FP32.
The unused ticket pointer aliases output under SEPARATE_MERGE=True.

Both producers retain S16/N64, grid(1,2,16), four warps/three stages, RoPE/window
and the same DOT=BF16/QK_DOT=E4M3 settings. Candidate internal M8/PARTIAL_M16
and baseline M16 feed the identical M16/S16/BS16 merge at grid(1,16).
The candidate's source-level arithmetic/window/tail/partial ownership review
is recorded separately in `../SOURCE_REVIEW.md`; this harness review does not
substitute for it. This specialization is page1/unit-scale only. Source support
for page64/nonunit fixtures remains a later native-validation obligation.

## Prospective gates and bounded artifacts

The screen enforces candidate producer<=192 registers, zero spills/local bytes/
local instructions, shared<=65536 bytes and actual legal maximum>=2CTA/SM at
128 threads. It also retains control/merge observations and applies the shared,
spill/local and legal-CTA checks to those units. Candidate Tensor MMA in PTX
or SASS rejects. Missing exact CUDA metadata or readable SASS is inconclusive;
compile/resource success never admits execution.

CUDA enums come from existing bounded headers. Existing driver APIs query the
compiled function, actual device limits, registers/static local/shared data and
occupancy maximum with launch dynamic shared bytes. This is a legal maximum,
not observed residency or proof that the32-CTA grid uses the device well.
The192-register estimate is a prospective source-shape budget, not a native
liveness result or a precision/register-cap rescue.

Actual eight-head state, absence of padded16-head/3D work, no eightfold global
KV reload, window/last-split ownership and partial-to-merge dataflow remain
mandatory independent TTIR/TTGIR/PTX/SASS gates. Textual opcode counts are only
inventory. No inference of native reuse or performance follows from source
loads being outside static head loops.

The worker retains90s alarm/deadline, atomic partial reports,2MiB final report,
8MiB raw gzip/JSON,4MiB per file and18 exact flat artifact names. Existing
cuobjdump alone is used, bounded10s, without installation. No remote controller,
billing or credential logic exists in this worker. Any later upload/admission
needs a separately reviewed frozen payload and bounded controller; this local
PASS does not lift the current launch hold. Native failures reject or remain
inconclusive rather than triggering a sweep or automatic runtime/official check.

## Reviewed SHA256

Paths are relative to the probe directory.

| Artifact | SHA256 |
|---|---|
| `baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| `candidate.py` | `f5161be2e91b05fceb01e28564f96eecff471244704d036fa4cc7950dbe0ef34` |
| `SOURCE_REVIEW.md` | `ecf1b1090db2d4ffd9b4fb8d8d2fdc4245cffaea94438b3f08bacb9e39e630b9` |
| `EXPERIMENT_CARD.md` | `7672a57a349b84f48f5e9c09ed8ecb480daad2514a792d9d0e1b0852c643f501` |
| `native/worker.py` | `e75d54a10342fe0d982a85d1db68a21422dc63714abf4320bd27ce14956797f8` |
| `native/test_worker.py` | `eecddfe583352f70364a247be0d54233b79f6defccd8ad05fa8f9247830c7319` |
| `native/CARD.md` | `6f7448fe95227246cf6d5968c3e61bd1396fff86a756402cbdc1246b597a1259` |
