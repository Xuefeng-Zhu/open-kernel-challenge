# Offline identity audit: gate still not passed

No service call, app, GPU workload, credential action, candidate edit or ranked
submission was made. The 34 files bound by the original result manifest were
verified byte-for-byte. The original result, raw metrics and failed gate remain
unchanged.

## Why offline re-export cannot run on these artifacts

The saved NCU 2025.3.1 help supports `--import`, `--rename-kernels 0`,
`--print-kernel-base demangled`, `--page raw` and `--csv`. If a lossless report
were available, the supported offline path would be:

```text
ncu --import retained-profile.ncu-rep --rename-kernels 0 \
    --print-kernel-base demangled --page raw --csv
```

This command was **not executed**: its required report file is missing. The
original profiling argv contains no `--export`/`-o`. Installed help lines 337–338
explicitly state that without export the temporary profile is removed afterward.
No `.ncu-rep` or older supported report extension was found in the saved Attention
evidence. The controller retrieved JSON/stdout/stderr and the Torch trace, with
no binary NCU report in its allowlist or manifest. Local Darwin also has no NCU
executable on PATH, but missing input is the decisive blocker; adding a tool
would not reconstruct discarded information.

The CSV's `Kernel Name` contains only the 425-character simplified rendering.
Its `launch__kernel_name` field is empty; no retained full/mangled identity can
be re-exported. The 1380-character graph name comes from a separate Torch trace,
not from the counter sample. A CSV is not a replacement for the missing lossless
report. No GPU was rerun merely to change formatting.

## What does and does not match

| Evidence | Current result |
|---|---|
| Selected B32/L8/full dispatch | `_fa3_full_verify_fused`, unchanged selected source |
| Input fingerprints and output/new-KV bytes | Match |
| Wrapper and loaded binary mapping | Match |
| Grid / block | Match: 132×1×1 / 256×1×1 |
| Unrenamed full function identity for counter sample | Missing |
| Exact graph/eager name gate | **False**; not relaxed after observing the data |
| Resolved split/packing/launch-dependency state | Not established |

Default NCU simplification is a plausible explanation for the different strings.
It removes information and is not a unique inverse mapping. Namespace removal,
boolean rewriting and a `Params` alias must not be normalized away to manufacture
a passing gate. Even a later exact function match would not alone prove identical
internal eager/graph scheduling state.

## Counter interpretation remains bounded

The four percentages have different denominators and cannot be added:

- Tensor 11.47%: average tensor-pipe active cycles normalized to peak sustained
  rate during elapsed cycles across hardware instances, not useful FLOP rate.
- DRAM 36.77%: throughput normalized to peak sustained rate during elapsed cycles,
  not a direct byte count or a share of whole-call latency.
- Long-scoreboard 43.40%: a warps-per-cycle ratio waiting on an L1TEX dependency;
  it does not identify an instruction, prove HBM involvement, or assign 43.40% of
  the 383µs whole call to memory latency.
- Active warps 12.49%: average in-flight warps normalized during unit-active
  cycles, not a per-CTA residency trace or proof of register-pressure causality.

Thus the sampled eager launch supplies a dependency/latency-hiding lead, not a
confirmed bottleneck for the timed graph. The current identity verdict remains
`inconclusive_eager_graph_proxy_mismatch`.

## Smallest justified follow-up, not executed

If separately admitted, collect one targeted same-case report that attaches the
needed counters to the original full/mangled function identity and launch metadata.
Use help-validated `--rename-kernels 0` and explicit `--export`, then retain the
bounded lossless report and its digest. Re-export that same report offline to
compare the full signature with the graph trace. If report size is a concern,
the installed help advertises `--import-sass off` to omit embedded cubins; validate
this output-only choice before use. No security setting needs changing.

A name-only later capture cannot retroactively validate the earlier four-counter
row. Any new counter interpretation must be bound to its own retained report.
If full-name parity succeeds but split/capture state remains unresolved, continue
to label graph transfer conditional. Do not introduce a candidate rewrite or
another ranked submission to resolve this evidence gap.
