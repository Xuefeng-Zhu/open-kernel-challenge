# Independent review of the offline capture plan

**PASS as an offline plan and regression suite. No new execution is approved
or claimed by this review.** No service, GPU, authentication, browser or Git
operation was used. Only this review file was written.

| Reviewed file | SHA256 |
| --- | --- |
| `capture_plan.py` | `be067afbc933020f41ea12f103c15263bc0e1c991c4a80ec1a1b6c1b763c26db` |
| `test_capture_plan.py` | `3fa01de9d3798819d5bc022e21f01e54a2bc40daf2f9384cb4afbe318757fc21` |
| `NEXT_CAPTURE_PLAN.md` | `d2e8a461ba081b902bd50f181c58bcf1988eeea1df64ba7fd50e53284e3650c5` |

Independently ran all **59 CPU tests: PASS**. The new eleven tests use the real
1,380-character graph spelling, 870-character NCU available-name spelling and
empty-capture stdout extracted from the saved failed `result.json`.

The family selector matches both spellings while excluding the combine kernel.
It is a selection mechanism, not cross-demangler identity proof. Config loading
and name simplification remain disabled; explicit export and target-free
same-report import remain in the proposed argv. Optional mangled-name output
is an offline view of that same report, not another profiling invocation.

The pure acceptance helper requires the separately frozen expected NCU-native
signature, grid `[132,1,1]`, block `[256,1,1]`, and selected/reference/wrapper/
binary/all-fixture-tensor hashes. Identical but wrong same-family signatures,
geometry or bindings cannot pass merely because capture and import agree.
Report metadata, exact NCU names, geometry and all four metric values/units
must match; missing report/row or an empty-capture warning rejects even exit 0.

Final completeness correction: `REQUIRED_BINDING_KEYS` now requires exact set
equality for all 15 saved bindings: four source/vendor hashes and eleven fixture
tensor hashes. The previous nonempty-dictionary check was insufficient and was
missed in the initial review. The added regression independently rejects each
missing key, an unknown-only dictionary, and an extra unknown key even when
capture, import and expected identity all carry the same deficient set.

The helper assumes its caller obtains rows from the strict CSV parser and
metadata from verified report bytes. It is not a standalone file/transport or
correctness verifier. A future integration must retain the existing bounded
report checks, verified transport, source binding, oracle checks, resource
guardrails and cleanup; it requires a new frozen snapshot and review.

The no-filter alternative is correctly rejected for the current child. The
saved selected graph has five ordered kernels (preparation, scheduler,
mainloop, combine, norm/gate), and fixture/oracle work adds other launches.
An unfiltered launch ordinal does not identify an isolated native mainloop.

The failed worker is unchanged at SHA256
`a9d05c9a212f5e4c2ab1e2b5fa4189b6632384d34d2af0442ef39fcfb88e3a01`;
controller is unchanged at
`4b528de5ffc90572edf6aff29e89e8c55ffb54c92dfa59dd5ad688ce830dc053`.
Neither should be relaunched as the proposed repaired executor.

Also checked `RESULTS.md`, `analysis.json` and `COSTS.json` against the raw
failure; all 14 extracted command stdout/stderr streams match their original
byte counts and hashes. Failure/no metrics/no permission denial and provisional
account-level cost qualification are preserved.

Graph ABI/mangled-symbol equivalence and resolved graph/eager packing, split,
PDL and scheduler-state equality remain **unmet/unknown**. The old identity gate
remains failed. No posthoc cast/alias normalization, counter reinterpretation,
performance gain or new GPU measurement follows from these CPU tests.
