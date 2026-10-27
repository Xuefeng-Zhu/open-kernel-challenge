# Independent local result review

**INCOMPLETE / FAILED identity recapture; acceptance correctly failed closed.**
App `ap-Wg106ohTxcewSqroHjDItS`, frozen source commit
`8e6ad398d8f89859e570303e66bbcee4086a9d9c`. No new counter row or durable NCU
report was captured. This review used local artifacts only, with no service,
GPU, authentication, browser or Git operations and no source/raw-data edits.

## Exact failure

The NCU capture returned code 0 after 21.7684 seconds, but the retained output
explicitly says `No kernels were profiled.` at line 14. There is no actual raw
CSV header/data row; the frozen CSV parser independently rejects this file with
`No NCU raw wide CSV header`. A zero CLI exit code therefore did not establish
successful profiling.

The worker then failed with `FileNotFoundError` at `durable_report_validation`
for `/tmp/attention-fa3-b32-identity-recapture/baseline.ncu-rep`. The durable
report and imported CSV are absent, and no import command ran. Worker status is
`incomplete_or_failed`, exit 1, `passed=false`; controller
`completion_verified=false`. No CSV-only success was accepted. There is no
observed `ERR_NVGPUCTRPERM`, timeout or permission denial in this attempt.

The literal filter came from the current Torch graph's 1,380-character symbol,
SHA256 `33239a4fe9825c98df5c44acd8bc412ee2c441fab7b704ffa6124fb68de96711`.
NCU's unrenamed list contains one FA3 mainloop spelling, 870 characters,
SHA256 `e74ccf26278f6fddc906194a2a415767a3f139941f63d0bc0cf67caa03ccbe87`.
It uses `(int)`/`(bool)` template casts and `T1::Params`, whereas the graph name
uses uncast literals/booleans and expanded parameter type. These names fail
both exact and whitespace-only comparison. This explains the filter-selection
failure without proving that the two spellings denote identical execution.
No posthoc rewriting or relaxation passes the existing identity gate.

## Retained evidence independently checked

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `result.json` | 168955 | `a611d400b95999274c72dcd1b2b5a7b98b4c2eea92a8a629c370ad7663ece85f` |
| `artifacts/profile-original.csv` (log, not a counter table) | 31827 | `e7b06eb33263ac19cf7c403d5fe22378eaa92093d64dc1ad358a206e017e6f5b` |
| `durable-artifacts.json` | 317 | `9529f7e2ebc59db46135a49d074d4b624e4e5b613841aa37fac20153098da32c` |
| `activity-trace.json.gz` | 3913 | `fba62378a059b891841d0e6dd9a09da546e765fdf179fccfaf45a2352ca82b74` |
| `trace-manifest.json` | 216 | `c6aa0ec31d0f5bf36ae049ebd1b135f253da68b3467558a41f8f346faeac34e1` |

The transported log matches its summary and durable manifest. Both compressed
and decompressed graph trace lengths/hashes match its manifest.

All 14 main-process checks and 3 ordered child checks passed, including 3 main
and 2 child calls to the unchanged original oracle. Child versus graph input
fingerprints, exact output/new-KV byte hashes, wrapper and mapped-binary
environment facts match. The selected candidate SHA256 remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The current source-bound graph again records the literal `64,128,64` mainloop
tuple, grid `132,1,1`, block `256,1,1`, and dynamic persistent scheduler name.
It does not establish resolved pack-GQA/split/PDL state or QK/PV instruction
semantics. This is not proof of upstream `192,160` dispatch or of the official
judge binary. No new counter percentages exist to interpret or compare.

Uninstrumented whole-call graph medians were 379.232 us before the Torch trace,
380.768 us after it, and 380.448 us after the failed NCU attempt (three samples
per group). These are baseline confirmations, not an optimization comparison;
trace durations and counter/replay timing are separate. No performance claim
or candidate change follows.

## Closure and remaining blocker

The controller ledger reports stopped state, zero tasks for this app and all
listed apps, observed gross account cost delta $0.09, and paid cost $0. Root
separately reported service-side verification at 05:38:54 UTC of stopped/tasks 0
for this app and all apps, with paid cost $0. This reviewer did not repeat those
operations. The account ephemeral-app delta was $0.08875238; these account
observations are not a final per-app invoice and may include delayed metering.

The blocker is verified kernel-name filter mismatch, not authentication or
hardware-counter permission. Offline diagnosis now has the actual NCU spelling
to examine. Any future capture needs a separately declared selection/identity
method supported by that evidence; it must not retroactively validate the old
four-counter row or call this failed capture complete. No additional GPU run
was performed or authorized by this review.
