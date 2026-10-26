# Independent offline artifact review: compile hard gate PASS

Reviewer: `/root/offline_transpose_review`. This review covers the generated
one-tile kernel only. It independently checks file hashes, IR/PTX/SASS dataflow,
instruction geometry, synchronization and static resources. No Docker/OrbStack,
network, service, GPU, Git or source-edit operation was performed by this reviewer.
Only this report was written during the artifact review.

The native mechanism gate passes: QK and PV both use real M64N8 instructions;
the required quantization and FP32 reductions remain visible; shared publication
and accumulator waits are present; no spill/local traffic or MMA fallback is
reported or found. This is not numerical correctness, performance, a complete
producer admission, or proof of the official judge's binary/toolchain identity.

## Provenance and independently verified hashes

Source directory: `reviews/attention-v2/m64n8-transpose-probe/`.
Artifact directory: `local-notes/attention-m64n8-offline/artifacts/`.
All line references below are one-based lines in these exact files.

| File | SHA256 |
|---|---|
| `kernel.py` | `118cdd776cf0f4668a2e3a31539c56f3f7d449748ae3881f9c0adf500344d033` |
| `offline_compile.py` | `e7a489a899d7eae222250ca546d00bfefbb3435cf4ad25c8a79c96d0a099fc37` |
| Earlier `REVIEW.md`, unchanged | `99cc6f94f98d8ab45f24bbebe54663e00ef9063483129263ca31ce5e3fc2a9e1` |
| `manifest.json` | `ca18c06e6aaf03f0721985c0e6e8ef86dc3c8f0b3e847215264302a8d6d451c9` |
| `tile.ttgir` | `fd5c306f61b722dce5d0a6eccad711501d53ccd07c3b1d6b5a6f0692824f45b5` |
| `tile.llir` | `58dc06e04ff219edce9aabba7ba3010d3fce3d9a8d656cf8fa8cf4e40784e262` |
| `tile.ptx` | `822c2b004ffb9066ecec394e3c6448ed68b9f26bc7effe13f123b1ed97a77ecd` |
| `tile.cubin` | `449d38c546da620b172d6ba50782fd79ed0801022527d4023b5e5712c0155ac7` |
| `tile.ptxas.txt` | `bc5fc1e29c3ea6cdc0dcde46c6a27109e8ac51525e3d15d0685b8706e355ec2a` |
| `tile.resources.txt` | `2f2b092782dbe8f6f308d08435027ff18d9bda6ace2e21522465dfabf60fe046` |
| `tile.sass` | `3000d4eaef7b3f4bba348c84336d7aaf991518c591000a25865dfde787515e87` |

Every artifact digest matches `manifest.json`; source/helper digests match its
nested preparation manifest. The earlier source review and selected candidate
binding remain unchanged. The selected source hash recorded by preparation is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The manifest records Python 3.12.15/aarch64, Triton 3.8.0, explicit CUDA target
90 with warp size32, four warps, three stages, ptxas12.9.86 and cuobjdump13.1.80.
The cached wheel/API source identities were checked in the earlier source review.
This report rechecks the produced artifacts, not an installed GPU runtime.

## Both operations are genuinely M64N8

TTGIR line7 declares MMA version3, four warps and instruction shape16x8x16.
Lines61–64 retain distinct K[64,64], Q-transpose[64,8], V-transpose[64,64] and
P[64,8] BF16 shared descriptors. Lines71–72 produce and wait on QK[64,8] FP32;
line96 produces PV[64,8] FP32. The query extent stays eight, including the
output index range at line103. There is no hidden 16/64-query mathematical pad.

PTX QK sites1132,1140,1147,1154 and PV sites1382,1389,1396,1403 are all
`wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16`. Each operation uses
four K16 chunks, covering its full K64 contraction. The first instruction
disables the prior accumulator and the next three accumulate into the same
four FP32 registers. PV uses the transposed-A descriptor flag as required by
V-transpose. The generated native instructions preserve these distinctions:

| Operation | SASS lines | Instruction offsets |
|---|---|---|
| QK | 489,511,521,523 | `0f20`, `0fd0`, `1020`, `1030` |
| PV, `.tnspA` | 713,735,747,767 | `1620`, `16d0`, `1730`, `17d0` |

All eight sites are `HGMMA.64x8x16.F32.BF16`. Independent static counts are
eight PTX WGMMA and eight native HGMMA sites, zero `mma.sync`, and zero old
HMMA/IMMA/MMA instruction-family sites. These are static instruction counts,
not executed counts, tensor utilization or throughput measurements.

## Precision and complete reductions

TTGIR line24 retains the typed ABI: Q BF16; K/V E4M3; scales/output/max/sum
FP32. Q load8x64 at line43 passes through RTNE E4M3 at44, then BF16 at45.
K/V E4M3 loads and BF16 widening are at53–54 and59–60. PTX Q encoding has
`cvt.rn.satfinite.e4m3x2.f32` at92,107,122,137, followed by E4M3 widening
and BF16 conversion at147–207. K/V widening repeats the same conversion
pattern through952. The F16 intermediate exactly represents finite E4M3
values; it is not an additional lower-precision quantization step.

TTGIR79–83 and88–92 reduce axis0 of the entire64x8 FP32 tensor to eight
FP32 maxima and sums. The native lowering is not merely a warp-local partial:

- PTX1195–1232 forms local FP32 maxima and intra-warp shuffle reductions.
  PTX1235–1248 then stores to shared, performs a CTA barrier and reloads
  redistributed values;1249–1270 finishes with FP32 shuffle/max operations.
- With `tid=32*warp+lane`, the store-row address expression at1235–1243
  supplies row bases0/16/32/48 for warps0/1/2/3 and64/80/96/112 for the
  second matrix bank. The load expression1245–1248 supplies all four warp
  offsets to every warp; final butterfly offsets8/4 combine those contributions.
  A CPU evaluation of the actual address expressions confirmed this routing.
- PTX1290–1354 implements the corresponding FP32 add/shuffle/shared-exchange
  denominator reduction, with shared-store/barrier/load at1330–1332.
- SASS max exchange is STSM/BAR/LDSM at597/599/601; sum exchange is at
  673/681/685. The `.16` matrix transfer spelling transports register bits;
  no floating-point down-conversion accompanies these reduction exchanges.

Score scales and log2 conversion remain FP32 at PTX1182–1188. Probability
uses FP32 `ex2.approx.ftz` at1280–1283. P rounds to BF16 at1364–1365 before
PV. Post-PV scaling and division are FP32 at1414–1421; the denominator is
the FP32 sum of probabilities before P rounding. Compiler FP fusion is enabled:
PTX1274–1278 fuses score scaling/subtraction. This is not a bitwise-equivalence
claim, and numerical acceptance still requires the unchanged GPU oracle.

## Publication, shared lifetimes and accumulator waits

K/Q/V shared stores finish before PTX1120 async proxy fence and1122 CTA
barrier. QK has a WGMMA register fence at1127, commit at1157 and wait-to-zero
at1162 before score consumption. SASS has the publication fence/barrier at
465/469, WARPGROUP.ARRIVE at483 and dependency wait at525 (`1040`).

The live shared regions are K at0, V-transpose at8192, Q-transpose at16384,
and P at17408. Reduction scratch reuses the beginning of K only after the
QK wait; it does not overwrite the still-needed V-transpose or P region.
The max/sum shared reuse is separated by CTA barriers. There is no next-tile
writer or asynchronous double-buffer reuse in this one-tile kernel.

P is stored at PTX1366, published with proxy fence1368 and CTA barrier1370,
then consumed after WGMMA fence1377. The native sequence is BF16 packing at
SASS677/679, P STSM at687 (`1550`), async publication fence691 (`1570`),
CTA barrier701 (`15c0`), WARPGROUP.ARRIVE711, then PV HGMMA713 onward.
PV commit1406 and wait-to-zero1410 precede any PV accumulator consumption.
Native dependency wait is at SASS819 (`1970`). Independent denominator
shuffle/add instructions overlap PV issue in SASS; they do not consume the
pending PV accumulator registers. No lifetime/publication blocker was found
for this frozen one-tile program.

## Resources and decision boundary

`tile.ptxas.txt` reports32 registers, zero stack bytes, zero spill stores/loads
and one barrier. `tile.resources.txt` reports LOCAL0, STACK0 and cubin SHARED1024.
Independent scans find no PTX local loads/stores or native LDL/STL sites.
The compiler metadata separately requests **18432 bytes of dynamic shared
memory**. The cubin SHARED1024 field does not replace or reduce that dynamic
launch requirement; both resource records must be retained. Register/shared
counts are recorded, not converted into a performance or residency claim.

The compile hard gate therefore passes without lowering precision or relaxing
zero-spill/local requirements. The 128-register/32KiB references are not used
as production acceptance thresholds. A complete producer still requires its
separate96-register parity/buffering gate and original correctness/performance
tests. This probe has only one full historical tile: online carry, tails,
new-token KV ownership, page gathers, cache writes, split/merge, replay and
concurrency remain unimplemented/unmeasured here. It has never been GPU-loaded
or executed. No native correctness, speed, whole-score or promotion claim is
supported. Selected1.233633 and A2-11 rollback remain unchanged.
