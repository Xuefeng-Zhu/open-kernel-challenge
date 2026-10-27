# Offline-only capture repair

**Not executed. No further GPU launch is authorized in this task.**

The failing selector was the complete 1380-character Torch graph name under
`--kernel-name-base demangled`. NCU's available name retains the same visible
FA3 class/tile family but renders casts and parameter aliases differently:

```text
Torch: ...CollectiveMainloopFwdSm90<2, ...C<64>..., true,...>(fully-qualified::Params)
NCU:   ...CollectiveMainloopFwdSm90<(int)2, ...C<(int)64>..., (bool)1,...>(T1::Params)
```

These excerpts illustrate the mismatch; `analysis.json` preserves both full
strings. Disabling name simplification does not make the two demanglers equal.

Use the already successful family selector
`--kernel-name-base demangled --kernel-name regex:.*FlashAttnFwdSm90.*` to avoid
cross-tool spelling as a selection requirement. Keep three unchanged selected
eager calls, skip one matching launch and collect one. The original reference
check does not call FA3 for this case; nevertheless validate the observed row,
not an assumed count. Keep config-file0, rename0, explicit export, bounded
report retrieval and target-free offline import. `capture_plan.py` is a CPU-only
command builder and early empty-capture validator, with real-name regressions;
it does not modify the frozen failed worker or launch anything.

Postcapture require exactly one FA3 row, the predeclared 870-character NCU-native
signature, grid `[132,1,1]`, block `[256,1,1]`, unchanged selected/wrapper/binary/
fixture hash bindings and original correctness. A different same-family row
cannot pass just because its own capture/import outputs agree.
Require the same report to import successfully with identical NCU signature,
launch, four metric values and units. `No kernels were profiled` or no actual
CSV row/report must fail immediately even when NCU exits 0.

The plan and regressions pass **59 combined CPU tests**. It is not wired into a
new executor snapshot; the failed frozen worker must not be relaunched. A later
admission needs a new source-bound integration and acceptance review.

**Still unmet:** a raw common ABI/mangled-symbol binding between the Torch graph
name and the NCU row. The proposed family capture can preserve that missing
representation through a mangled-name re-export of its *same retained report*,
using help-supported `--print-kernel-base mangled`; mapping it back to the graph
requires separately verified binary/demangler evidence. No cast removal,
`T1::Params` expansion or later name-only run may retroactively validate old
counters. Resolved graph/eager split, packing and scheduler-state equivalence
also remain unknown even after static identity parity.

A no-filter `--launch-count 1` is unsafe in the current child: fixture/oracle
work and `kernel_fn` launch many kernels, and the selected call itself includes
preparation, scheduler, mainloop, combine and output work. The public wrapper
does not expose an isolated native mainloop. A blind ordinal would select an
unproven target; no no-filter variant is proposed without its own observed
launch-isolation evidence.
