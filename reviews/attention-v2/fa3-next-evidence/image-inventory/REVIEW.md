# Independent CPU-only image inventory review

**PASS: no concrete code or bounded-inventory contract blocker.** Final worker
and controller files were confirmed stable by their owners. Sixteen synthetic
worker CPU tests and eight independent controller CPU tests pass. This review
read local code and ran CPU tests only; it did not invoke the real inventory,
authentication, a service, network, GPU or a commit. Only this report was added.

| Reviewed file | SHA256 |
|---|---|
| `worker.py` | `5a287e350612a5f10356a5645e50689c51cdce80c175bb8d22f017cae1321f23` |
| `test_worker.py` | `f9d644813e9a81b538d087614dfd186f45592ba2b42c08ca641e8d50d21e4511` |
| `controller.py` | `382b62b4a6200a4561b62593cb1b7360c047ea93c01b3cbbeb9e38584bb00783` |
| `test_controller.py` | `e43f5ed94510be36c48eb08e07e22a931b17599faa61b2ce7b0536b7390afbe5` |
| `CONTROLLER.md` | `ca25ad48fd6a059fc6f45fbc60d07732725b0f20b3cca581d24c6d67064da85a` |

## Read-only worker scope

The worker imports only the standard library. It queries version metadata for
the explicit distribution allowlist and calls `find_spec` only on the five
named top-level modules; no dotted discovery imports package parents. It never
executes vendor modules, creates a CUDA context, starts a process, calls an
operator, reads HOME/environment credentials or makes a network request.
Module presence and distribution metadata are recorded separately.

Only fixed relative vendor source candidates and a fixed header-name allowlist
under declared vendor directories are inspected. Python code is parsed with
AST/unparse, not evaluated or imported. The limited output is signatures,
page-size guards, hashes and selected source lines. No recursive machine-wide
search or arbitrary source/API invocation is present.

Resolved module roots must be permitted vendor locations. User/config/cache
roots are refused; the fixed `/opt/sglang` allowroot cannot resolve elsewhere.
Every candidate is resolved beneath its allowroot and protected subtrees are
refused again. Directory traversal skips symlinks. Reads require regular files,
use read-only/no-follow opening, and compare device/inode/size with the opened
descriptor before reading; post-read length/size is checked too. Canonical
resolved paths are deduplicated. The immutable image/source setting is the
scope of this guard, not a claim of a general hostile-filesystem sandbox.

Bounds are64 examined files,8MiB total source bytes,2MiB per file,4KiB per-file
snippets,64KiB aggregate snippets,24KiB aggregate signatures and256KiB whole
JSON. Header discovery is additionally bounded by depth4,32 directories and
256 retained entries. Oversized files are skipped. The known large
`flash_ops.abi3.so` is stat-only, with no open, hash or binary content export.
Missing/limited candidates remain explicitly missing or limited.

A finite worker deadline of at most30s is enforced by cooperative checks and
an alarm; TimeoutError is rethrown rather than swallowed by the OSError paths.
Partial error reports are bounded and atomically saved, with symlink output
paths rejected. Synthetic tests cover wrong roots, escaping symlinks, protected
subtrees, non-executed source, file/read/walk/export bounds and timeout handling.

## CPU controller, payload and closure

Default execution is offline committed-byte/hash checking and Python parsing;
it does not import the worker/vendor packages, Modal or credentials. Actual
source freeze and offline preflight on that commit still precede admission.
The upload allowlist contains exactly worker/LICENSE/THIRD_PARTY_NOTICES;
flattening them beneath `/opt/fa3-image-inventory` matches the supervised
`worker.py` path. The remote entry has no closure containing local credentials
or receipts. No installation or broad local-directory upload is present.

The function declaration requests CPU1/2GiB-host, max one container and zero
retries, with **no GPU keyword**. Bounds are30s worker,35s hard child timeout,
45s function,10s startup and60s app-stop watchdog. The supervisor may start the
single bounded Python worker and kill its process group on timeout; the
inventory worker itself starts no subprocess. Network is blocked and source
inclusion is explicit.

The$0.02 reserve uses CPU and memory rates only, doubled over the60s watchdog
plus$0.005 slack. Fresh live admission checks current paid0, a verified paid0
cap, sufficient adjusted free credits and no active Modal apps/tasks. Existing
protected supported configuration is used without printing/copying/renewing
credentials or adding credential environment variables. It is never payload.

One exclusive request ledger prevents automatic duplication. Returned JSON is
bounded to256KiB, transport-hashed and tied to the frozen worker, with explicit
no-GPU/no-context/zero-launch flags. Stop/watchdog/finally paths retain initial
shutdown observations and require independently verified stopped/tasks0;
uncertain or stopping state must be reconciled rather than retried. This CPU
read does not allocate a GPU or change official measurement scheduling.

## Interpretation and remaining admission

The parent may freeze and admit the one already scoped CPU read after actual
offline preflight and live guardrails. No further user-approval question is
introduced by this local review. No launch or cost has occurred by this reviewer.

Presence, version strings, source signatures and fingerprints do not establish
an eligible compiled kernel, exact installed FA3 revision, precision, graph-safe
planning, page1 support or speedup. Missing source is a real evidence limitation;
the report must preserve it. Further architectural work must use the actual
returned evidence and cannot infer a future GPU/kernel launch from this review.
