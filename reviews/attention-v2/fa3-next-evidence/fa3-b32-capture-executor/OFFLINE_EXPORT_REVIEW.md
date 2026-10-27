# Offline retained-report export review: PASS

Reviewed source SHA256 `61c60c85afb65db6971f94ea1e1ee4d6719c05f2d14074f22b957d732cee0fd9`.
Focused tests SHA256 `3baff7063eba1b02baae0c5e467d931052b810015170da97c1353dd9663752ca`.

Five CPU fixture tests pass. They invoke the actual closure-free entry with only
its container report path remapped and subprocess calls substituted; the retained
35,756-byte report, saved installed help, version and imported CSV are real saved
fixtures. No NCU process, service, authentication or GPU action was performed by
this reviewer. Original report SHA256 remains
`4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6`.

The remote entry uploads only that report, requests `gpu=None`, validates its
size/hash, reads installed version/help, and executes two identical target-free
`--import` commands. Configuration, rename, base-unit and floating-point-print
flags are checked against actual saved help before imports. No profiling target,
export, metric query, security setting, install or credential enters the entry.

Nonzero imports, missing help options and timeouts now retain structured phase,
command argv and partial stdout/stderr. Tests cover these paths and reject a
changed report before commands. Before/after report hashes remain mandatory.

`--print-fp` availability does not guarantee numerical precision or recover a
passing frozen native gate. The two canonical CSV imports and their numeric
representation must be observed in the next CPU-only run; command completion
and `canonical_stdout_identical` are separate outcomes. Prior native failure
records remain untouched. The controller records its initial cleanup state;
root must independently confirm stopped/tasks 0 as planned if transition is
still pending. Paid spending is checked before and after.
