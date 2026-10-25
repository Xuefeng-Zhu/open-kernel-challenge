# Native audit: BF16 widening enables QK WGMMA at M64

Offline inspection of the newly retrieved frozen-source H200 artifacts. This
is compile/disassembly evidence only: no candidate launch, correctness/replay
test, paired timing or official score is established by this audit.

Both B16/B32 variants retain A2-13 source SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Only `QK_DOT` changes at the compile call, from E4M3 to BF16. L8/G8/M64/D64,
BN64, split11/chunk8192, four warps, three stages, BF16 PV and the integrated
ticket merge are unchanged. B16/B32 are different launch grids of the same
specialization: cubins are byte-identical within each operand variant.

| Compiled variant | Historical QK | New-token QK | PV | Registers | Shared bytes | Raw driver spills |
|---|---|---|---|---:|---:|---:|
| E4M3 control | FP16 `mma.sync` | FP16 `mma.sync` | BF16 WGMMA | 172 | 33,024 | 0 |
| BF16 widening | BF16 WGMMA | BF16 WGMMA | BF16 WGMMA | 135 | 24,832 | 0 |

This maps the individual source products, rather than inferring QK from any
WGMMA instruction. Recursive TTGIR location aliases map source lines 141 and
171 to QK and 152 and 179 to PV. PTX `.file/.loc` mappings independently agree:

- Historical QK: four static `wgmma.mma_async...m64n64k16.f32.bf16.bf16`.
- New-token QK: four static `...m64n16k16.f32.bf16.bf16`.
- PV: five static `...m64n64k16.f32.bf16.bf16`.

The BF16 cubin contains 13 HGMMA instructions and no HMMA; the control contains
five HGMMA and 36 HMMA. PTX control has 40 static QK MMA instructions; the
different final count reflects compiler/assembler transformations and is not
a direct speed estimate. There are zero PTX local loads/stores and zero SASS
LDL/STL instructions in these files. The raw driver spill field remains zero;
its units have not been reinterpreted as bytes.

The exact query quantization sequence is visible in BF16 TTIR lines 138–140:
`f32 -> bf16 -> f8E4M3FN -> bf16`. Cache storage remains E4M3. Both QK source
calls explicitly retain `max_num_imprecise_acc=0`, verified from the frozen AST.
TTIR omits that attribute when printing these dots; omission is not independent
zero proof. It does show BF16 operands and FP32 dot outputs. No explicit
nonzero imprecise-accumulation attribute or native FP8 WGMMA was introduced.

The matrix instruction gate passes, but the emitted schedule does **not** prove
a more overlapped pipeline. Historical QK TTGIR `warp_group_dot` is immediately
followed by `warp_group_dot_wait` with `pendings=0`. In BF16 B16 PTX,
commit_group at line 1658 reaches wait_group 0 at line 1667 after five scalar
conversion/move instructions. In SASS, QK HGMMA at addresses 0x2a30–0x2a60 is
immediately followed by `WARPGROUP.DEPBAR.LE ...0` at 0x2a70. Historical PV has
the same immediate-wait structure (0x4160–0x4190 then 0x41a0). There are four
WGMMA commit/wait groups in BF16 versus two in the control. Static placement
does not establish dynamic stalls, throughput or useful overlap.

Resources improve relative to this M64 control by 37 registers and 8,192 shared
bytes. This is sufficient evidence of a distinct, precision-preserving requested
mechanism to consider representative correctness/replay follow-through; it is
not a prediction that the custom path can beat existing FA3. The old custom
verify deficit remains substantial, and the control already used WGMMA for PV.
The next runtime comparison must retain the current FA3 route as control and
measure the complete call, including normalization/cache writes/ticket merge.

`analyze_native.py` hash-checks every receipt artifact and emitted
`native-analysis.json` with complete source/artifact hashes, per-dot mappings,
instruction/control locations and resource deltas. All 24 recorded native
artifacts plus the summary are present. Its semantic parser self-tests passed,
including the important negative case where only PV has WGMMA. A regression
against historical A2-13 M16 artifacts recovered QK/PV roles without claiming
QK WGMMA. Installed runtime is Triton 3.8.0, Torch 2.14.1+cu130, CUDA 13.0,
H200 capability 9.0 and driver 610.57.04. Exact Triton Git revision is unavailable;
distribution METADATA/RECORD and compiler Python-source hashes are recorded.
