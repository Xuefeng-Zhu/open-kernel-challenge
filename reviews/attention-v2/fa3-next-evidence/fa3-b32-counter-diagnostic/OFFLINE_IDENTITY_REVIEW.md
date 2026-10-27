# Offline identity audit: the exact gate remains unresolved

Independent read-only review of the saved B32/L8 baseline artifacts. No service,
GPU, authentication, browser or Git operation was performed; no source, raw
result, counter value or original identity verdict was changed.

## Offline re-export prerequisite is missing

The current artifact directory and the original result manifest contain CSV
stdout/stderr, JSON summaries and a compressed Torch activity trace. A bounded
file-name search across `reviews/attention-v2` found no `.ncu-rep`,
`.nsight-cuprof-report` or equivalent saved profiler report.

The exact profiling argv in `result.json` uses `--page raw --csv`, with neither
`--export` nor `-o`. The controller retrieves the JSON summary and Torch trace;
it has no binary NCU-report transport branch. The installed NCU help captured
in `1-stdout.txt` is decisive:

- Lines 337–338: `--export` chooses the profile-results file; without it NCU
  uses a temporary file which is removed afterward.
- Line 341: `--import` reads a profile-results file.
- Lines 252–256: `--rename-kernels` defaults to 1 and simplifies demangled names.
- Line 386: `--print-kernel-base` controls name output and defaults to demangled.

Thus an offline import/re-export is **not executable on the saved data**. This
is a missing-input prerequisite, not a permission denial. No invalid attempt
was made to treat CSV or the Torch trace as an importable NCU binary report.
If an original binary report is later found, the captured help supports an
offline command of the following form, without launching a target application:

```text
ncu --import existing.ncu-rep --rename-kernels 0 --print-kernel-base demangled --page raw --csv
```

That conditional command was not executed: `existing.ncu-rep` is absent.

## The lost identity fields cannot be recovered uniquely from this CSV

The trace preserves a 1380-character fully qualified symbol. The NCU CSV
preserves a 425-character simplified name. Namespace qualifications are absent,
boolean template arguments are printed as 1/0, and the parameter type appears
as `Params` rather than the fully qualified dependent type.

The CSV contains one data row. Its `launch__kernel_name` field is empty;
`launch__function_pcs` contains `1`, which supplies no usable full-symbol binding.
There is no retained nonempty original mangled/full-signature column linking
this counter row to the graph symbol. Device unified-addressing attribute
columns are not kernel identity addresses.

Matching grid/block, byte-identical full inputs and output/new-KV, wrapper/file
mapping and complete binary-file SHA256 remain useful evidence. The unchanged
binary is `cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`.
Those facts do not restore the missing original symbol or establish equal
resolved split/capture/scheduling state. The apparent naming correspondence is
consistent with simplification, but manually deleting namespaces or rewriting
booleans after collection would relax the predeclared gate.

**Verdict:** `necessary_identity_checks_passed=false` and
`inconclusive_eager_graph_proxy_mismatch` must remain unchanged. No graph
bottleneck is established by this review.

## Smallest justified follow-up, if separately admitted

First recover any independently retained original `.ncu-rep`, if one exists;
there is none in the audited saved artifacts. If recovery is impossible, the
missing data require a new targeted capture, not another formatting operation
on the existing CSV. One same-source, same-fixture, same-launch session should
disable name simplification and persist a lossless NCU report through explicit
`--export`, retrieving that binary along with the full or mangled kernel name,
launch metadata and the counters intended for interpretation. Validate all
options against the already captured installed-version help. Preserve the
original oracle, input/cache/whole-call checks and timing separation.

Even a subsequent exact name/grid/block match is necessary evidence rather than
proof of equal eager-versus-graph split/scheduler state. Confirm those launch
and auxiliary-state differences before transferring an eager diagnosis to the
timed graph. Do not label the old four percentages as graph-validated through
a separate later name-only probe. No such follow-up was run here.

The four retained percentages also have different aggregation/denominator
semantics. Tensor-pipe active cycles are normalized to sustained peak over
elapsed cycles; this is not useful FLOP utilization. DRAM throughput is relative
to sustained peak over elapsed cycles. Long-scoreboard is a per-active-warp
stall percentage, whereas active warps is normalized using active rather than
elapsed cycles. They cannot be added together, and they do not alone establish
latency, HBM saturation or register causality.

## Evidence hashes

- Installed help `1-stdout.txt`:
  `ac75d8e8b9801ec2b43d18ffa925b9d9e6917ea5ed0c47634af7d67ae1eae7c9`
- Original `result.json`:
  `3abcaeb2b52b57ccd94f8dd56110c4d2908815e8039b2fb1315f53475c71983b`
- Original `RESULT_MANIFEST.json`:
  `f2933df8c788018e5caa45112dc21ac2aa6abd32729bf061042fc684c057c0cc`
- Trace full-symbol string:
  `33239a4fe9825c98df5c44acd8bc412ee2c441fab7b704ffa6124fb68de96711`
- NCU simplified-symbol string:
  `cde43affc2c7836b1e5177403df89475e2bf82f2e1f0451ff1e3e56facf5037a`

## Independent review of the root audit

**PASS on the factual identity conclusion.** I independently checked
`OFFLINE_IDENTITY_AUDIT.md` and `OFFLINE_IDENTITY_AUDIT.json` against the stored
help, original argv, raw CSV, graph record and manifest. All 34 original
manifest-bound files still match their byte counts and SHA256 values.
No offline re-export was executed, no importable report is present, and the
identity gate remains false. The metric denominators and conditional wording
match the preserved runtime query descriptions; no latency cause is established.

Reviewed root document hashes:

- `OFFLINE_IDENTITY_AUDIT.md`:
  `42451d6470c23991c9b65b2ba9f7d416afb70d3757e513f01422a2da6281ab07`
- `OFFLINE_IDENTITY_AUDIT.json`:
  `990e9283776f68ad0a126266ad7141f702e2f825cd99df81395ef1b9b4688047`

The minor future-command wording correction is resolved: the root now uses
`--import-sass off`, explicitly listed by the captured `on/off` and `yes/no`
help. I verified that this single spelling replacement is the only change to
the already reviewed Markdown; the JSON is unchanged. No proposed command was
executed. Final audit PASS retains the missing-report blocker and false gate.
