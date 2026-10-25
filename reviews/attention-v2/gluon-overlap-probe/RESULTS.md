# Gluon attention-core compile result: qualified mechanism, strict control gate fails

Both fixed arms compiled, with native BF16/FP32 WGMMA and intact double buffers.
The overlap arm retains next-tile payload loads, conversion and shared stores
before QK drain. However, the compiler also hoisted next-tile K payload loads
above the control arm's QK drain. The predeclared fully serialized native
control was therefore not retained. **Do not mark the complete mechanism gate
passed or advance this exact declared pair automatically to execution.**

This is a useful compiler finding, not measured performance or a numerical
failure. Neither kernel was invoked. Working fusion+lookahead source
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
and A2-11 rollback remain unchanged; no official check or ranked entry occurred.

## Frozen source and narrow scope

- Freeze: `605e815462b56768fca9c84c68a8aada639bbb0b`.
- Kernel SHA256: `3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c`.
- Worker SHA256: `608cc5769883baa204c817773da80b1841eab1837c12daef7ef9a2b1518b4a39`.
- Controller SHA256: `3c5cf83169e717dbc9cb1a24d5a2aadcbe1f8a44e7652ef2212186ce8546ab8a`.
- Raw result SHA256: `8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea`.
- Fixed image: `im-cPTbGpNiX3RarKResPgX7h`; actual installed Gluon source pins
  were verified before probe import. Thirty local checks and independent
  source/harness review preceded admission.

The standalone core takes unscaled normalized BF16 Q and an FP8 cache already
containing history plus eight new tokens, then returns raw BF16 attention.
It includes BF16-to-E4M3-to-BF16 Q rounding and all paged KV gather/widening.
It does not implement production normalization, cache append, output norm/gate
or a production split/merge. No complete-attention speedup can follow from
this microcore alone. It is not a claim that FA3 lacks asynchronous overlap.

Both specializations use B32/L8/G8, M64/N64/D64, page1, four warps, one stage
and grid(32,2). Only `OVERLAP` differs. QK factor5.4730064863875 is applied
after QK; live K/V descales are applied once. Both use zero-C BF16 PV followed
by common FP32 `acc*alpha+PV`, without claiming carry-C bitwise equivalence.

## Observed gates

| Gate | Observation | Disposition |
|---|---|---|
| Compile transport | Two units; ten hash-verified TTGIR/LLIR/PTX/CUBIN/SASS artifacts; Gluon emitted no TTIR | Pass |
| Arithmetic | Eight BF16/FP32 WGMMA instructions and two commit/wait groups per arm; matched common PTX math | Static pass; numerical behavior untested |
| Shared ownership | Current K at `base+(slot<<13)` and current V at `+16384`; opposite-slot next staging; completed PV before reuse | Static pass, not a runtime race certificate |
| Resources |167/168 registers;32,768 shared bytes each; zero spill/local; CUDA legal maximum3 CTA/SM each | Pass; no achieved-residency measurement |
| Strict serial control | Actual control K payload LDGs precede its QK drain | Fail |
| Execution/correctness/timing | Zero candidate invocations | Unknown, not admitted by the complete declared gate |

After normalizing debug text, the PTX body is identical after removing only
the first QK wait from each arm (2,189 normalized lines; SHA256
`3c9d56d78a167e20a7d0c033321b24686ce32e485e50cf4f208f9475fd9d3d05`).
This supports matched logical work. PTX instruction order alone misses the
subsequent native scheduling distinction:

- Overlap SASS: QK HGMMA at0x2880–0x2930; next K payload loads from0x2a10,
  V payload loads from0x2d80 and shared stores at0x3320–0x3830 precede QK
  DEPBAR at0x3920.
- Control SASS: next K payload loads from0x2980 occur before QK DEPBAR
  at0x2d50. Next V payload loads from0x2db0 and shared stores from0x3510
  follow it. These early K accesses are payload loads, not just page-table
  addressing.

There is a real remaining V/conversion/store scheduling contrast, but the
control is already partly overlapped. No runtime concurrency or saved cycles
were observed. See the independent [artifact review](native/ARTIFACT_REVIEW.md)
and retained raw artifacts for exact dataflow and wait/buffer evidence.

## Resources closed and cost recorded

App `ap-AECevTBOrJ7vVRE4EWgk3R` was stopped after compilation. Independent
reconciliation at **2026-10-25 23:44:21.845040 UTC** observed stopped/tasks0
and all listed main-environment apps stopped/tasks0. The slot was released
before artifact review. The original receipt retains its initial stopping
observation; reconciliation is a separate record.

The worker child took5.811733s, client15.400792s. At the23:47:17.887804UTC
no-new-app cutoff, delayed detailed workspace credit usage increased
**$0.01044224**, paid **$0**. This is not an exact per-app invoice. The fixed
free-credit reserve was$0.40; worker/child/function/startup/watchdog bounds
were90/100/115/20/140 seconds. No installs, credential changes, billing changes,
additional GPU job, official check or formal entry followed.

## Local diagnosis and next decision

Do not force a slower artificial baseline or relabel this as a clean serial
control. The next local experiment design can explicitly ask about the
remaining V/widen/store schedule relative to the compiler's already-prefetched
K control, with a new declared interpretation before any execution. It must
preserve the failed strict-control result and obtain source/harness review;
no performance threshold has been changed here. A production candidate still
requires original-tolerance correctness/replay coverage, a comparable current
FA3/working-path complete-call benchmark and a costed B1/sliding/full coverage
plan. The public score-gap model remains conditional arithmetic, not a speed
forecast from this compile result.
