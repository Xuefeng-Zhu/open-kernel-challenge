# Independent prelaunch review: PASS

Reviewer: `/root/counter_repair_review`. Local source review and CPU tests only;
no service, authentication, GPU, browser or Git action by this reviewer.
All **27 CPU protocol tests pass** independently on existing Python 3.12.13.
This is prelaunch evidence, not proof of installed CLI compatibility or counters.

The profiling argv removes the unsupported `--units base` pair. Before any metric
query or profiling target, the worker runs the installed NCU `--version` and full
`--help` with bounded logs, requires successful exits, and checks all exact NCU
option tokens in both query and profiling commands against that help. It rejects
`--units` even if a fixture advertises it. Required choice names are checked too.
Target Python flags are excluded from NCU grammar validation. If help fails or an
option is absent, no metric query or profiling target is invoked. The captured
help and its SHA256 remain evidence; this guard does not claim to implement the
entire vendor command parser. Actual execution remains the integration test.

The same fixed `sm__cycles_elapsed.avg` query, deterministic eager tiny kernel,
kernel replay and application-only process scope are retained. A unique kernel
filter with skip1/count1 profiles the middle of three matching calls. Ordered
pre/measured/post correctness markers, exit0 and exactly one finite positive
counter are required for PASS. Import, CUDA-init and per-call progress markers
localize failure. There is no FA3, CUDA graph, NVTX, application replay, candidate
change, clock change or security/permission change. A timeout does not establish
permission denial. The shared helper's old profiling/run-gate functions are not
called; only its tiny kernel, metric query/parser and bounded command runner are
reused under its frozen SHA256.

Focused regressions verify unsupported/missing options stop the gate before any
query/profile call, a longer similarly named option cannot satisfy an exact flag,
and query flags are checked too. The upload mount and execution root both resolve
to `/opt/attention-eager-counter-repair`; a regression protects this integration
boundary. The four-file payload is only worker/helper/licenses, and the remote
entry has no closure variables that could carry local credentials.

The existing controller retains the reviewed one-H200, 2CPU, 8GiB, zero-retry,
65s worker / 75s child / 85s function / 20s startup / 110s watchdog envelope.
Fresh existing-auth/free-credit/$0-paid guards and independent stopped/tasks0
verification remain the root owner's launch and cleanup responsibilities.

| File | SHA256 |
|---|---|
| worker.py | `7391280b4577a2d1f242bdffdc42777a26a180b1cdcd6e021dc47c2866c1137a` |
| controller.py | `af759a477f7916aa48b0e50fb221aaa41c449b054dcb905d45e22add531f90a5` |
| test_worker.py | `68879e6eea5abe85c429294e9cb70d9431c8453f1ce2e3de34cc1e4a062ae38d` |
| test_controller.py | `43b6f5f02fd92cad43a3bd99aec2012ff485fcbdc114e9a5779af17e6a1d50ca` |
| test_help_contract.py | `a17ffc18aa62134d6b1fa7ab046daaebff385717ce8c81c78ee7a66a603b10d8` |

Frozen helper: `d984cbd44691cfead1c889cbb2ae8b0bde2e11e0a1d703bbdd78a7ecf1b30637`.
No remaining prelaunch code blocker identified for this bounded capability test.
