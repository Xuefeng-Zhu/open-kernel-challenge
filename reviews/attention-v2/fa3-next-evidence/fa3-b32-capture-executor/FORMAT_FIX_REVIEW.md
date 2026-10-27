# Independent formatter-fix review

**PASS for the future formatter fix and offline artifact replay.**

| File | SHA256 |
| --- | --- |
| `capture_plan.py` | `2b6f7639ca51ca30a299f67a4787a7252debce0d69056dbb944ab67d78eb4c76` |
| `worker.py` | `346982496afad9a982687e8d6132964967e39b5a1fcbaca4b7506e82123e2567` |
| `controller.py` | `bad886806ebaba640ba3a981639e82d2fd71e44fbbe675dba78f543ebeb0aa71` |
| `test_worker.py` | `8c137dcad02955e69e6876770ec1422faf893d0ad276c360ed81e1caa76898fc` |
| `dry_run.py` | `494e1cec299558afce48f5d0069ee2f4d3c257e3414a48e610230e153a1bbc6f` |

Read-only comparison with `c1208f41841a560a08726a4c95b94e7a44dcf4f3`
confirms the production delta: import adds the empirically proven
`--print-units base --print-fp`; the worker delegates its actual import builder
to that helper; worker/controller update the helper hash pin. Path validation
remains in the helper. Selection, numerical gates and candidate math do not
change. The delegation is necessary because the executed worker previously
used its own hardcoded import argv.

Independently verified all **15 native payload hashes against the c120 Git
blobs**, matching the native admission ledger. Current working sources are a
normal versioned fix; they are not represented as the sources executed by that
older native run. Raw native result remains SHA256 `2dd7a86c…`, report remains
`4104eaf5…`, and the original failed verdict/default-import CSV are preserved.

All **40 CPU tests passed** independently. Four added regressions check actual
canonical exports, keep the legacy six-decimal comparison failed, verify the
production import argv and validate actual installed-help support. The new
optional real-fixture replay is explicitly marked as no new measurement.

Also independently reran the collector, packaging and materializer using the
saved native report/capture and actual CPU canonical import: completion passes,
while empty capture and corrupt transport reject. Materialized hashes remain
`4104eaf5…` (report), `aed2fe38…` (capture), `80177532…` (canonical import).
Root's `CANONICAL_REPLAY_SUMMARY.json` SHA256 is
`a808aa69651c6e09b682d95e14926148500a1828e8f52f43c6d413d2a7148377`;
the independent replay's source hashes match it. NCU/native boundaries were
substituted; no GPU or service was invoked by this replay.

`CANONICAL_ANALYSIS.json` SHA256
`1832624b55691cfe2c2e51f6b2d9ed63498f0303a81315809ee9dccbe75fe383`
agrees with independently checked raw artifacts: exact 870-character native
name, grid132/block256, four raw strings/units `11.42%`, `36.55%`, `43.37%`,
`12.49%`, and unchanged report bytes/hash. No tolerance or arithmetic rounding
was applied. Native and CPU cleanup receipts record stopped/tasks 0/all apps 0
and paid cost $0; provisional account readings are not per-app invoices.

The frozen c120 gate remains false. Recovered artifact verification is separate;
graph ABI/internal scheduling state remains unresolved. This review adds no
launch authority or speed/bottleneck claim. Only local tests, artifact reads
and the specifically requested read-only Git blob comparison were performed;
there were no Git mutations, GPU, authentication or service operations.
