# Independent final compile-result review

**PASS for evidence integrity and the qualified narrative. The complete
predeclared mechanism gate FAILS; execution is not admitted.** Both arms
compiled with feasible reported resources, but native scheduling already
prefetched next-K payloads in the nominal serial control. This is a useful
compiler result, not a correctness failure or a performance measurement.

This review used saved local bytes and CPU-only parsing/transport/resource
helpers. No source edit, vendor/GPU import, kernel, service, network,
credential or Git action occurred. Only this report was written.

## Raw result and static findings

The retained result SHA256 matches its launch receipt and analysis. All seven
receipt-bound current payload/controller file hashes match. Freeze is
`605e815462b56768fca9c84c68a8aada639bbb0b`, app
`ap-AECevTBOrJ7vVRE4EWgk3R`. Actual controller decoding reproduces all ten
artifact bytes, hashes and sizes, matching the saved artifact manifest. There
are five files per arm: TTGIR/LLIR/PTX/CUBIN/SASS; no TTIR was emitted.

Each arm has eight explicit BF16-input/FP32-output WGMMA instructions, two
commit and two wait sites. Recorded resources are serial167/overlap168
registers,32768 shared bytes each, zero spills/local bytes/instructions, and
CUDA legal maximum3 CTAs/SM each. The actual resource helper reproduces PASS;
that maximum is not achieved residency. Both units record no invocation,
correctness or performance test and no execution admission.

The artifact, SASS and independent mechanism reviews agree on the emitted
precision, matched logical PTX work and static buffer addressing/lifetimes.
Direct checks of the retained SASS corroborate the decisive boundaries:

| Boundary | Serial | Overlap |
|---|---|---|
| Next-K payload LDG example |0x2980, before QK drain|0x2a10, before QK drain|
| Next-V payload LDG example |0x2db0, after QK drain|0x2d80, before QK drain|
| Next shared stores |0x3510–0x49c0, after drain|0x3320–0x3830, before drain|
| QK drain |0x2d50|0x3920|
| PV drain before loop reuse |0x52a0|0x5250|

The reviews trace K/V bases and load-to-widen-to-store chains, rather than
equating arbitrary LDG counts with next-tile work. The overlap arm retains
more completed staging before its drain, but the serial arm is already partly
overlapped. Source/PTX serial ordering does not establish a fully serial final
ISA control. `analysis.json` correctly records strict_serial_control_gate_passed
false and execution_admitted false. Static ownership/precision findings do not
prove runtime race freedom, numerical tolerance, dynamic overlap, saved cycles,
an FA3 feature gap or a whole-call benefit.

## Closure, costs and final narrative

The initial receipt's stopping/tasks1 observation remains unchanged. Separate
23:44:21.845040UTC reconciliation confirms this app stopped/tasks0 and all
listed apps stopped/tasks0. The app's recorded stop time is23:43:37UTC. Saved
23:47:17.887804UTC cutoff reports no new other app IDs and all listed apps
stopped. Decimal arithmetic from initial/final detailed workspace billing
reproduces+$0.01044224, with paid$0. The$.40 reserve is a bound, not a charge;
the delayed workspace delta is not an exact app invoice. Recorded child/client
times5.811733s/15.400792s agree with the narrative.

`../RESULTS.md` accurately separates compile/resource/static-mechanism findings
from the failed strict native control and unknown runtime behavior. It preserves
working3da/A2-11, records no official check or formal entry, and proposes local
diagnosis rather than automatic execution or a posthoc gate relaxation. No
evidence or publication blocker remains; the failed control remains a blocker
to onward execution under this declared pair.

## Reviewed SHA256

Paths are relative to the probe directory.

| Artifact | SHA256 |
|---|---|
| `RESULTS.md` | `effed23e9a437f8411f5ca7cb0e35bb57a3d9868a7b812e620b297d19777d3e9` |
| `NATIVE_MECHANISM_REVIEW.md` | `635ec48286908303a2bdce5e80f658f2a234c35728e749e9dfd99802a63c7034` |
| `native/ARTIFACT_REVIEW.md` | `f07915aa3fe76f57ff0ce7104c755ee5b39e938b88d21ce3b268447ace5ed606` |
| `native/SASS_REVIEW.md` | `b19291b86841d55b0b59df75b3c41f1fd2b8ec552ca658a25c3f235b6bfc5fdb` |
| `native/result.json` | `8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea` |
| `native/analysis.json` | `ebca7a8c3953f666602324fee88427deca675b9265ba835f4c1d3f9a283d529b` |
| `native/COSTS.json` | `feb566e55b0a23e14abfc53c5695c3dfc4a2247ce7200defbbdad1c294873952` |
| `native/launch-receipt.json` | `b09dfda89db604c54fac02acc4cbb8d25c49a0d089e38f84dcdec02b65b02e63` |
| `native/reconciliation.json` | `b042af5a14ea78fa412e15b62d54250a5f646d600a6c523be5760c873e0db163` |
| `native/accounting-cutoff.json` | `2be83e9b5c6d8bc98dc0d7fbc0fa33d9680f7ad6342cc1c45c32bba4b1531b97` |
| `native/artifacts/manifest.json` | `6fb48f89b276c9c3729e072b08c342359535544181dfb17bdd8281f246ed62e9` |
