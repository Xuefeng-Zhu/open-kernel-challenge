# Independent Gluon native mechanism review

**Overall declared-pair gate: FAIL; no execution admission.** Compilation and
resources pass, and overlap has more complete staging before the QK wait.
However, ptxas also hoists next-tile K payload reads in the serial arm before
its QK dependency wait. The predeclared strict native serial-before-gather
control is not retained. Do not relax that requirement after seeing artifacts
or describe this as a clean fully-serial versus overlapping comparison.

This review used local saved artifacts only. No kernel, GPU, service, source
edit or Git operation was performed. It adds only this report.

## Binding and gate decisions

Native result SHA256:
`8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea`.
The receipt names source freeze`605e815462b56768fca9c84c68a8aada639bbb0b`;
kernel SHA256 remains
`3d847a290d747c204419fd131c8149a21c4afddbdfb1d0a6130f5f522b81847c`.
The exact two fixed options differ only by OVERLAP. All **ten** saved artifact
sizes/hashes match the result/manifest: five per arm, TTGIR/LLIR/PTX/CUBIN/SASS.
Gluon emitted no TTIR; twelve was an allowlist maximum, not twelve actual files.
The276,814-byte compressed bundle and2,653,542-byte decoded payload hashes
also verify. No binary was executed by this review.

| Gate | Decision | Evidence / limit |
|---|---|---|
| Exact asynchronous QK, commit and dependency wait | PASS | Four BF16/FP32 QK MMAs, commit and mapped wait survive in TTGIR/PTX/SASS |
| Next-tile work before overlap wait | PASS, partial mechanism | Actual K/V payload reads, conversion, other-slot stores and fence precede overlap depbar |
| Strict native serial wait before all next staging | **FAIL** | Serial next K payload LDGs occur before its depbar, not just page-table/address work |
| Static buffer ownership/publication/drain | PASS | Distinct slots and V offset, fences, dependent score wait and completed PV before reuse; not a numerical test |
| Strict BF16 operands / FP32 state, matched work | PASS for emitted code | Both PTX instruction streams differ only by one moved QK wait after removing debug/location text |
| Resource gate | PASS |167→168 registers,32,768 shared bytes, zero spill/local access, legal maximum3 CTAs/SM in both |
| Correctness, actual overlap, performance | UNKNOWN / not tested | Kernel launches0; correctness/performance false; static maximum is not achieved residency |

## Trace tile identity rather than counting LDG mnemonics

PTX parameter1 is KCache (`%rd18`), parameter2 VCache (`%rd19`) and parameter3
PageTable (`%rd20`). `%rd96` is PageTable plus the request-row byte offset;
`%rd195=%rd96+512` at line1856 is the next64 i64 page entries. It advances512
bytes per iteration at line3493. The overlap load at2040 and subsequent page
loads feed clamped physical rows, multiply by128=HKV2×D64, and add key/value
base pointers. K payload loads are2104–2146, V loads2536–2578. The resulting
values feed FP8 widening and shared stores through2947, then fence2951 and
QK wait2957. This is actual next-tile dataflow, not arbitrary unrelated LDG.

TTGIR provides the corresponding typed chain: async QK at127; `start+64` and
`next_slot=slot xor1`; live page mapping and E4M3 payload loads; BF16 K/V
stores at148/152; fence153; dependent wait154. Serial places that wait directly
after QK. Normalized PTX differs only by moving its wait from serial line2016
to overlap line2957; all remaining instruction lines are identical.

## Final SASS: useful difference, contaminated serial control

| Operation | Serial SASS address | Overlap SASS address |
|---|---|---|
| Four current-QK HGMMA instructions |0x2870,0x28b0,0x28e0,0x2b30|0x2880,0x28c0,0x2920,0x2930|
| Next K payload LDGs |0x2980–0x2d10, **before wait**|0x2a10–0x2d10, before wait|
| Next V payload LDGs |0x2db0 onward, after wait|0x2d80–0x2f00, before wait|
| Next K/V shared stores |0x3510–0x49c0, after wait|0x3320–0x3830, before wait|
| Publication fence |0x49e0|0x3850|
| Current-QK WARPGROUP.DEPBAR |**0x2d50**|**0x3920**|

In both arms, some next page-table LDG.E.64 instructions are already hoisted
before HGMMA. More decisively, serial `UR16` is loaded from KCache's parameter
slot`c[0][0x218]` at0x25f0. Its physical-row offset is added at0x2950/0x2960,
feeding `LDG.E R83` at0x2980 and seven further K reads through0x2d10, all before
serial depbar0x2d50. These are payload reads, not page entries. Serial V uses
the distinct base`UR18=c[0][0x220]`; its payload starts0x2db0 after the wait.

The overlap arm completes additional V reading, conversion and all next-tile
stores/fencing before its wait. Thus the weaker statement “more staging moved
before the wait” is supported. The stronger declared clean serial control is
falsified at final ISA even though it was present in source/PTX. No observed
runtime overlap or speedup can be inferred from these static schedules.

## Precision, ownership and resources remain separate findings

TTGIR preserves Q BF16→E4M3→BF16 and K/V E4M3→BF16. Every matrix instruction
in both PTX files is `m64n64k16.f32.bf16.bf16`; SASS confirms BF16 HGMMA with
FP32 outputs. Metadata's generic default TF32 field is not the operand type of
these instructions. Both retain four QK and four PV instructions, the same
FP32 online recurrence and zero-C PV followed by the common carry rescale/add.
This is matched probe arithmetic, not bitwise equivalence to production FA3.

TTGIR has separate `2×64×64xbf16` K/V allocations. PTX forms current K from
`slot<<13`, current V at+16,384 bytes, and next stores from `(slot xor1)<<13`.
SASS constructs the current descriptor before toggling `UR4`, then constructs
next-store base`UR7` with8192-byte slot stride; V stores add0x4000. Current QK
therefore does not consume the slot being overwritten. Publication fences are
present, score registers are waited before their use, and PV's separate wait
drains before loop reuse. This supports static ownership without claiming
runtime correctness or a general concurrency guarantee.

Driver resource metadata reports167/168 registers,32,768 dynamic shared bytes,
zero local bytes and legal maximum3 CTAs/SM for both128-thread arms. PTX/SASS
contain no local loads/stores and raw spill fields are0. These pass the stated
resource test; they do not establish achieved occupancy, a bottleneck or speed.

## Decision

Preserve this as a **partial mechanism result with a failed strict-control
gate**. Do not execute or time this exact pair under the declared experiment,
and do not use official quota. Because the gate fails, no minimum future
execution proposal is admitted by this review. Any later differently framed
local hypothesis would need its own predeclared control and independent review;
there is no GPU rescue, precision change or posthoc gate relaxation here.
