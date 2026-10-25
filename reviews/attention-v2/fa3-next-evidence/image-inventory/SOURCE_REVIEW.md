# Independent source review: bounded CPU vendor inventory

**PASS for the reviewed worker's local source and guard tests.** No remaining
concrete path-scope, read-bound, timeout or vendor-code-execution blocker was
found. This review did not run an actual image inventory, query a service,
import vendor/GPU packages or inspect credentials. It does not establish kernel
availability, capture/precision eligibility, correctness or performance. The
controller and any eventual execution/admission have a separate review scope.

| Reviewed artifact | SHA256 |
|---|---|
| `worker.py` | `5a287e350612a5f10356a5645e50689c51cdce80c175bb8d22f017cae1321f23` |
| `test_worker.py` | `f9d644813e9a81b538d087614dfd186f45592ba2b42c08ca641e8d50d21e4511` |

The reviewer independently ran all16 worker tests against these final bytes;
they passed. The controller's eight tests were not counted as independently
executed by this worker review.

## Discovery and execution scope

Only stdlib modules are imported. The eight distribution queries are fixed to
Torch, Triton, SGLang, sgl-kernel, flashinfer, flashinfer-python,
flashinfer-jit-cache and cutlass-dsl version metadata. The five `find_spec`
queries are fixed, **top-level** Triton/flashinfer/SGLang/sgl_kernel/cutlass
names; there is no dotted discovery that would first import a parent package.
Module presence and distribution metadata remain separate observations.

Vendor files are decoded and processed with AST/text operations only. Function
signatures use AST arguments/unparse; annotations/defaults are not evaluated.
No vendor source is imported or executed, and no GPU operator, subprocess,
network, credential/config API or HOME lookup is added. A synthetic source
containing an immediate exception is successfully analyzed without executing it.

Allowed file candidates are fixed relative vendor-source paths, bounded
allowlisted header basenames beneath discovered vendor attention roots, and
five fixed attention source directories under`/opt/sglang`. The final three
additional names—`paged_kv.h`, `softmax.h` and
`flash_prepare_scheduler.cu`—retain the same roots and budgets. The large
`flash_ops.abi3.so` is **stat-only**: it is not opened, hashed, imported or loaded.

## Path and symlink guard

Module-derived canonical roots must be permitted vendor paths. User/root/HOME-
style, system/config/cache/credential paths are rejected. Each inspected file
resolves beneath its validated canonical allowroot; outside symlink targets
are skipped, and canonical paths are deduplicated. Header traversal does not
follow symlink entries.

An initial concrete blocker was found before freeze: resolving the fixed
`/opt/sglang` root without validating its target could accept an outside root
if that directory itself were a symlink. The final `resolve_allowroot` helper
fixes this in both inspection and traversal: the fixed root must resolve
exactly to`/opt/sglang`, and every root must pass vendor-location validation.
The exact root itself is explicitly allowed, avoiding accidental rejection of
all legitimate fixed-header scans. The new fifteenth test covers a redirected
fixed root and an outside synthetic symlink root; neither is read. Final
per-file vendor-location checks also reject protected path components below
an otherwise allowed vendor root, and traversal skips those entries before
descending. The sixteenth test confirms that a `.config/page.cuh` fixture
inside the vendor tree is neither opened nor traversed as source.

Bounded reads use a no-follow file open where supported, then compare regular
file type, size, device and inode against pre-open metadata. The read requests
only the previously bounded size, then checks returned and final size. This
limits growth/race exposure; it is not a formal guarantee against arbitrary
concurrent mutation of the installed filesystem. The intended input is the
existing trusted image's vendor tree, with no writes to it.

## Limits and failure behavior

* At most64 regular source/binary candidates examined,2MiB per source file and
 8MiB total source bytes read/hashed. The explicit eight metadata queries and
  uploaded-worker self-hash are separate stdlib/control operations, not claims
  that every interpreter filesystem read is included in those source counters.
* Traversal is bounded globally to depth4,32 directories and256 entries, using
  exact header basenames. Missing and skipped records each cap at64 entries.
* Each source excerpt caps at4096 UTF-8 bytes, with64KiB aggregate excerpts and
 24KiB encoded signatures. Signatures and page-guard records have per-file caps.
  Serialized JSON is checked against256KiB, including escaped output bytes.
  A file row can be dropped with an explicit limit marker to respect export size.
* CLI deadline must be finite and in(0,30] seconds. Cooperative checks occur
  between bounded steps; a30-second-or-shorter SIGALRM covers inventory/output
  work. TimeoutError is explicitly rethrown before broad OSError handlers and
  produces an incomplete report, never a passing inventory. Overall process/
  startup and service watchdog bounds remain the controller's responsibility.
* Output is a bounded JSON file written through a temporary and atomic replace;
  symlink output/temporary paths are rejected. The controller must keep that
  output path within its declared temporary area.

The tests exercise import/discovery scope, module-vs-distribution separation,
AST-only analysis, fixed API files, outside symlinks, the root escape fix,
stat-only large binary, file/byte/count/directory/depth limits, excerpt/signature
budgets, vendor root rejection, timeout propagation and bounded output.

An inventory marked complete means the bounded operation finished, not that all
vendor files were discovered or every API is usable. Missing, skipped or export-
limited source cannot establish absence or eligibility. Returned source is
evidence to review, never authority to install packages, expand credentials,
launch a kernel or change a candidate. No such action occurred in this review.
