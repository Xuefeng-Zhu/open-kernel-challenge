# Independent prelaunch review

**PASS for the bounded identity recapture implementation below.** This is a
source/CPU acceptance review, not runtime identity closure or a performance
result. No services, GPU, authentication, browser or Git operations were used
by this reviewer. No source or old evidence was changed.

## Frozen review inputs

| File | SHA256 |
| --- | --- |
| `worker.py` | `a9d05c9a212f5e4c2ab1e2b5fa4189b6632384d34d2af0442ef39fcfb88e3a01` |
| `controller.py` | `4b528de5ffc90572edf6aff29e89e8c55ffb54c92dfa59dd5ad688ce830dc053` |
| `csv_helpers.py` | `a00f901febe900e4ed9f48680b990acf813f0c0d76680b1c798a664f017759df` |
| `test_worker.py` | `ce3e1496e8e30a7510123420feae83cb38a5dae612ffdd527f5327293618a546` |
| `test_controller.py` | `d8bede8516cc989a2e19dc288ff35901f643376a09b844fe25d112a850151201` |
| `test_report_roundtrip.py` | `50f3ef64f0fc53dab3ed1f2dc0dd069d2e0da04457034660b8951d8369a0aac2` |
| `test_csv_helpers.py` | `0424949353c121377c097a2558fe1ab8c77f597ba5e85a62e2b1a6fc3509db19` |

Selected source remains SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
The saved installed NCU help used for the command audit is
`../fa3-b32-counter-diagnostic/1-stdout.txt`, SHA256
`ac75d8e8b9801ec2b43d18ffa925b9d9e6917ea5ed0c47634af7d67ae1eae7c9`.

## Acceptance reviewed

- Every NCU path disables default configuration loading with `--config-file 0`.
  Capture and import disable name simplification and print demangled names.
  Capture uses the fresh graph's literal full symbol with demangled matching;
  there is no regex or fallback. Skip/count select the second matching child
  call. The child has only pre/measured/post selected calls and kernel replay.
- Capture explicitly exports a fresh absolute `baseline.ncu-rep`. The report
  must be regular, nonsymlink, nonempty and at most 64 MiB. Hashing checks file
  identity/size/mtime stability; byte count and SHA256 must remain unchanged
  across import. No overwrite flag is used.
- Import reopens that same report with no Python/application target, replay,
  metric collection or export operation. Both commands must complete with exit
  code zero. Original and imported CSV must each contain exactly one actual
  row and agree on exact name, grid, block, all four requested metric values
  and units. No type/namespace rewriting repairs a mismatch.
- The controller requires all three durable artifacts, bounded lossless
  transport and matching raw/compressed hashes. It reparses the transported
  CSV independently of summary booleans. Missing report, failed import,
  corrupted/oversized transport or false equivalence cannot satisfy completion.
  Verified summary and durable outputs are saved before trace validation;
  partial evidence remains a failed/incomplete diagnostic.
- Original oracle, byte/cache checks and graph-versus-eager fixture, output,
  wrapper and mapped-binary checks remain required. Identity mismatch returns
  failure. The 100-second remaining-time guard precedes the 75-second capture;
  import and enclosing worker/controller watchdogs remain bounded.

Independent CPU run: `python -m unittest discover -s <this folder> -p
'test_*.py'` completed **48 tests, PASS**, including missing/empty/symlink/
oversized/mutating reports, target-free import, row/name/geometry/metric
mismatches, import failure, corrupt transport and false CSV-equivalence claims.

Also reviewed root's `CPU_PREFLIGHT.json`, SHA256
`8af9b50851faa683632ee716456c1e6d0c99c67c0e4f58d0f2371e66b9e3c56c`.
Its seven source hashes match this review; constructed capture/import flags
match the saved help, artifact names/bounds match the controller, and the saved
prior graph symbol is explicitly a CPU fixture only. Runtime binds a fresh
graph symbol and runtime Python executable. This preflight records zero secret
pattern matches and no service execution; it is not an actual NCU report.

## Limits and launch prerequisites

No prelaunch blocker was found in these hashes. The controller must still
verify fresh free-credit eligibility, paid cap $0, existing authentication and
the serialized slot before launch, then independently verify stopped/tasks 0.
Actual exact-filter matching, valid NCU report generation and successful import
remain runtime gates; this review does not claim they have occurred.

`--import-sass off` does not preserve cubin/SASS evidence. Even successful
symbol/launch/input/output/binary parity does not resolve opaque pack-GQA,
split, graph/PDL or scheduler state. Counter denominators differ; the four
percentages are neither additive nor sufficient alone to prove a bottleneck.
The previous CSV-only capture and its failed identity gate remain unchanged.
