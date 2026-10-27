# Preparation-version correction review: PASS

Reviewer: `/root/offline_transpose_review`. This corrects the preparation
protocol, not kernel code or compiler/resource gates. The historical
`HELPER_REVIEW.md` remains unchanged at SHA256
`3baad6a36d1e951487ff707f5470f8f23bf3c7e550d685eb0896a6c6e44641cb`.
Only this new review was written. No container, compiler, service, GPU,
authentication or Git action was performed by this reviewer.

## Attempt1 stopped before the compiler

Preserved payload: `local-notes/attention-m64n8-producer-offline/`.
Its manifest SHA256 is
`32774b9cdcd4b8c14813ea4dc8e116e19efc3a891e5bcfb2c04e67107d603c64`.
`artifacts/container.stderr.txt` SHA256 is
`02debd4b55ea91b40784265e4dbee5bcfa1f756693e2619a8a051c7128d112bc`.
The traceback stops at helper `validate_manifest` line273, called at307,
with `ValueError: Control contract changed`. Compiler imports begin later
and the `triton.compile` call is at355. There is no compiler manifest, unit
or native artifact; the directory contains only stdout, stderr and receipt.

The execution receipt SHA256 is
`c2f99c4d15fb6e8391772951285475df878d9ad589e020489059bee619e39fd6`.
It records exit1, verified removal of this exact Attention container, no GPU,
paid spending0 and no shared-engine stop request. This reviewer checked the
saved evidence, not live container/service state.

This is a **precompiler harness validation failure**. It is neither algorithm
compilation failure nor a resource-gate rejection. The algorithm's compile
gate remains untested; no native result or performance observation exists.

## Independent CPU reproduction

Host `python3` is3.14.3. Its default `ast.dump` omits empty fields; the3.12
serialization used by the Linux compiler environment retains them. Whole
source bytes match, but the manifest's function AST digests therefore differ.
The host's expanded-field dump reproduces the3.12 function digests below.

Using the already installed interpreter
`/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python` (3.12.13),
I independently validated both payloads without importing Triton or compiling:

| Payload | `validate_manifest` result |
|---|---|
| Attempt1, prepared under3.14 | `ValueError: Control contract changed` |
| Fresh `local-notes/attention-m64n8-producer-offline-py312/` | PASS |

Fresh manifest SHA256:
`f1bef1b5686f1ea5f0a8f5d55e6f6d5c42457bd0b15d7a28dbee10423953c7ce`.
The only changed top-level fields are `control_contract` and
`candidate_contract`; within each, **only `functions` digests changed**.
All other ABI, source, dependency, options, order and gate declarations match.
For example, the fresh control digests are:

- `_norm_rope`: `970065927be62705da09115c3917f7cb3fd98fe4628bdb1f6575590f0f1579e4`
- `_fused`: `5cd279b82ececbeca2a1d429034afd94986e3e309d610d510882edff8dcfe265`

All16 helper tests also independently pass under this existing3.12 interpreter.
The fresh artifact directory was empty when checked. Linux3.12.15 must still
run the unchanged validation before any compiler call; this CPU reproduction
does not claim that the fresh container has already executed.

## Minimal correction and unchanged boundaries

The solution is preparation with existing Python3.12, matching the compiler
environment's AST serialization generation. No helper patch or skipped contract
comparison is needed. Independent byte comparisons prove both payloads retain:

| File | Unchanged SHA256 |
|---|---|
| Kernel | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| Extracted control | `cfefc9b9a9b2b175945fe7e6990d7c5fe46efb6e752acf88683c0dee4bb05356` |
| Helper | `f8cc6a6b72a0ba9076d2a4ece018a4ecce2140a9161ac8aa35b4eb9e4442f5fd` |
| Execute wrapper | `6a1256a05847d339e0ab08e80217b3f540d9496e32315d10e864899a97d6486d` |

Tests remain SHA256
`ac24e146e5969e6bb187a439cb2ed37e339b8bf6bb756126a1a5972d92bfdb8d`.
The fixed four-controls-first order, D32full-first candidate, first-failure
stop, precision,96-register/zero-spill/local gates, dynamic40KiB gate, timeouts
and no-driver boundary are unchanged. No tile/flag/precision rescue was made.

No protocol blocker remains for the planned first actual compiler invocation
using the fresh3.12-prepared payload after root's final freeze. This approval
does not erase attempt1 or reinterpret it as a native result. Future payloads
using this helper must prepare under the matching Python AST version, or
separately review a version-independent serialization change before execution.
