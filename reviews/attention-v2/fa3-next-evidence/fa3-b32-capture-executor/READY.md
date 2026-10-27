# Integrated executor: offline verification complete

The existing collector, remote artifact packaging and local materializer are
connected. Selection uses `regex:.*FlashAttnFwdSm90.*`; acceptance uses the
predeclared complete NCU-native signature, grid `[132,1,1]`, block `[256,1,1]`, all 15 source,
wrapper, binary and fixture hashes, and same-report signature/launch/metric
parity. There is no cross-demangler string-equality requirement. Graph ABI and
internal scheduler-state equivalence remain explicitly unresolved.

**31 CPU tests passed.** The real command/file/packaging/materialization flow
also passed a local dryrun with three outcomes: complete simulated report
roundtrip accepted; saved exit-zero empty capture rejected before import;
corrupt report transport rejected. The successful tiny-counter CSV is a real
parser control. The generated report is explicitly synthetic, not a valid
Nsight report or a new GPU measurement. See `DRY_RUN_SUMMARY.json`.

Reproduce locally with a new output directory:

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python \
  reviews/attention-v2/fa3-next-evidence/fa3-b32-capture-executor/dry_run.py \
  --output-dir /tmp/attention-fa3-executor-dryrun-fresh
```

The future native verification command is below. Use the reviewed remote SHA
reported with this change, plus **fresh** source-bound slot and read-only
billing receipts. This step did not create those receipts or launch anything.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python \
  reviews/attention-v2/fa3-next-evidence/fa3-b32-capture-executor/controller.py \
  --frozen-head <reviewed-remote-SHA> --launch \
  --slot reviews/attention-v2/fa3-next-evidence/fa3-b32-capture-executor/slot.local.json \
  --billing-receipt reviews/attention-v2/fa3-next-evidence/fa3-b32-capture-executor/billing.local.json
```

All 14 uploaded files are listed by `controller.payload()` and bound by
`EXECUTOR_MANIFEST.json`. Credentials and this dryrun script are excluded.
Native admission retains existing-authentication, free-credit/$0-paid and
no-active-app guards; one H200 invocation, a 190-second worker deadline,
enclosing watchdogs, $0.70 reserve and independently checked stopped/tasks 0.

**Remaining native check:** the actual installed NCU version/help and metric
queries must pass, then the family selector must yield exactly one expected
row and a nonempty real `.ncu-rep`. Its real offline import must reproduce that
row and all four values/units, with original correctness before/after and
complete local retrieval. CPU simulation cannot establish any of these native
facts or a latency bottleneck. No candidate or selected/ranked source changed.
