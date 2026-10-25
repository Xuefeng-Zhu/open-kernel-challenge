# Reproduce the offline profiling audit

Use Python 3.12 or newer from the repository root. This postprocessor uses only
the standard library and makes no CUDA, Torch, Modal, browser or network calls.
It does not import or execute the candidate.

```sh
python -B reviews/attention-v2/architecture-probe/profiling-run/analysis/analyze.py --self-test
python -B reviews/attention-v2/architecture-probe/profiling-run/analysis/analyze.py
```

The default raw folder is `../gpu/artifacts`. A published
`../gpu/launch-receipt.json`, when present, supplies transport hashes and the
frozen payload/commit checks. There is no implicit ignored-local-ledger fallback.
An alternate receipt requires `--receipt PATH`; alternate raw inputs require
`--artifacts PATH`. Absence of a published receipt is stated explicitly in
`transport_audit`; local frozen source and summary/raw-trace hashes are still
verified. Do not interpret that limited state as a re-audited transport receipt.

The analyzer independently recomputes fenced intervals from the unmodified raw
Chrome trace, rather than trusting the compact worker calculations. It verifies
all four pre/post official-oracle, stable-address and immutable-input checks,
exact post-profile output/new-KV return, one activity session, one graph replay
per arm, complete expected FA3/custom stage coverage, and agreement with the
worker's compact interval unions. Actual source and trace bytes are unchanged.

It retains full event names, raw IDs, CPU graph-launch API activities, pairwise
stage separations and union-frontier uncovered intervals. Pairwise separation
does not establish a global idle gap. CPU annotation/API durations and all GPU
activities are instrumented observations, not ordinary host-launch overhead or
replacement unprofiled latency.

Optional NCU parsing retains long/wide raw CSV values and literal units. Numeric
admission requires finite values/units, the exact requested/returned metric set,
selected main-kernel identity, raw report checksum and pre/post target oracles.
Explicit tool-log pass counts remain separate from requested session count.
Unavailable counters never become a bandwidth/tensor/stall/residency diagnosis.

For this actual run, the fixed B32/L8/page1 activity session completed and passed
all four checks. The first FA3 NCU session timed out at 60 seconds, returning no
metrics, target validation, raw report or captured stdout/stderr. No second arm
was attempted. Permission status and hardware bottlenecks remain unknown; no
permission bypass or additional experiment was performed.

`RESULT.md` is the readable outcome, `analysis.json` retains all recomputed
evidence and exact source/artifact hashes, and `manifest.json` hashes local
deliverables. The frozen execution commit is
`e26bb0cf291dfdbfc6e70065b6d121bf315423ca`.

The separately authorized remaining counter-access session has its own report
and immutable inputs. Reproduce that supplement without rewriting the original
activity artifacts:

```sh
python -B reviews/attention-v2/architecture-probe/profiling-run/analysis/analyze_access.py --self-test
python -B reviews/attention-v2/architecture-probe/profiling-run/analysis/analyze_access.py
```

It reads `../counter-access/artifacts` and the published access launch receipt,
verifies retained log bytes/base64 hashes and the same fixed FA3 target, and
emits `COUNTER_ACCESS_SUPPLEMENT.md`, `counter-access-analysis.json` and
`counter-access-manifest.json`. The second session also timed out, at 55 seconds,
with target oracles passing and complete logs retained but no NCU report or
metric. It supplies no hardware permission proof, bandwidth result or BF16
comparison. The original 60-second timeout remains unchanged.
