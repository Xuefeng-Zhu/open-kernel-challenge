# Common-PV native artifacts: mechanism/resource gate passes

**PASS for the predeclared compile mechanism and static resource gates only.**
The candidate retains96 registers, original K and V asynchronous double
buffering, one common PV carry chain and exact merge SASS. Its alpha-one edge
actually suppresses numerator MUL execution through SASS predication rather
than unconditional multiply followed by selection. No kernel was launched;
runtime correctness, frequency, performance and promotion remain unadmitted.

## Exact job and artifact binding

Source freeze is`6df535d096aec0e76379b6795bb97b5956d0474b`. Both producer and
merge units for both exact sources compiled under identical options; all24
artifact sizes/hashes were independently checked against the returned records.
The raw result SHA256 is
`de0efb07a8e46e6e261d16db249668619e7d7083c17c599478db5f2aff861e90`.
`artifact-review-evidence.json` retains these offline checks and inventories.

| Binding | SHA256 |
|---|---|
| Working control | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Common-PV candidate | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| Fresh control producer SASS | `44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de` |
| Fresh candidate producer SASS | `4a61b188fc06b893cb2a56842b4ddb7916bd2c645d8ce12fa02b3565b241dbf9` |
| Fresh candidate producer TTGIR | `3610b912316aadef1bbcdcbdeb871f239bbe6afc49cfda728086212f468035d4` |
| Fresh candidate producer PTX | `ef38dc8abeda9797234fc2dff7245b106518719a6c179b0e3867ee9fb61fcd44` |

The compile-only report records H200 sm90, Triton3.8.0, Torch2.14.1+cu130,
CUDA13.0 and zero kernel launches. Producer options are L1/G8/HQ16/HKV2/D64,
M16/N64, CHUNK8192/SPLITS11, PAGE_LOOKAHEAD true, separate merge, four warps
and three stages. Merge uses the unchanged BS16. This is native evidence for
the scored TP8 specialization only; constexpr-false/native DP specializations
were not compiled in this bounded probe.

## Actual predicate and MUL suppression

Candidate TTGIR358–366 compares the computed alpha against exactly1 with
unordered-not-equal, ANDs with valid query`row_ok`, sums integer violations
along axis0 and checks whether any remain. A valid non-unit or NaN alpha
therefore selects multiplication, while padded query lanes are excluded.
This is not a rowmax-equality shortcut or an approximate threshold.

TTGIR367–374 contains a multiply-only conditional. Its rescale edge returns
`old_acc * alpha`; its other edge returns`old_acc` unchanged. PTX1717–1741
contains the exact comparison/shuffle reduction and a real branch around the
eight carry MULs at1744–1751. Both edges reach the same`$L__BB0_4` common PV
block at754, whose eight BF16 MMA instructions appear at780–815.

The native SASS if-converts that short region into **predicated** operations:
all eight historical rescale FMULs at0x2770,0x2790,0x27a0,0x27b0,0x27d0,
0x27e0,0x27f0 and0x2810 have`@P0`. P0 comes from the valid-alpha comparison
and reduction around0x25f0–0x2730. Thus alpha-one suppresses those operations;
this is not eager multiplication whose result is merely discarded by a select.
The eight source-level MULs remain present in the static instruction inventory.

Predicated-off instructions may still consume issue/control resources. This
gate proves the intended narrow mechanism, not that skipping arithmetic is
free or beneficial. Actual frequency and uninstrumented timing remain essential.

## One common PV, unchanged math and ownership

Candidate TTGIR378 has one historical PV`tt.dot`, outside the conditional,
using its returned`%16` as C. Fast C is old acc; slow C is exactly the old
alpha product. P is rounded to BF16 exactly as before; V widens from the same
E4M3 shared buffer into the same BF16 dot operand layout. There is no separate
PV-from-zero result followed by a scalar addition and no duplicated executed
PV. The per-output MMA chains retain the four K16 accumulation steps.

SASS confirms one common historical eight-HMMA chain at0x2cc0–0x3620, with
the same numerator register groups as carry and destination. Static BF16 MMA
count is10 in both complete producers: eight historical instructions and two
new-token-tail instructions. This differs from the rejected variant's two
exclusive historical PV copies and18 static BF16 MMA instructions.

Rowmax/safe-max, alpha and P exponentials, QK conversions and denominator
reduction retain the original dataflow. TTGIR348–354 retains denominator
product, P reduction and add. Its denominator FFMAs remain at SASS0x3870 and
0x3880 after the common PV. Both complete producers retain12 static FP32 FMAs,
16`ex2.approx.ftz.f32` operations and26 FP32 max operations. These counts support
the localized dataflow review; they do not alone establish numerical equivalence.

The last-split branch remains split==10, with original new-KV normalization,
E4M3 quantization/stores, causal mask, alpha/denominator/numerator rescaling
and BF16 PV. Candidate TTGIR437 onward contains this separate original tail.
The seven source tests establish byte-exact tail, denominator expressions,
dispatch and other functions. Partial output/index ownership and merge remain
the original counter-free separate-merge route.

Slow carry MULs retain their original FP32 form without adding FTZ; exponentials
retain their existing approximate FTZ form. Preserved carry structure and finite
multiply-by-one algebra do not certify arbitrary signed-zero/subnormal/NaN
payload behavior. No input was processed in this compile-only probe, so original
oracles, cross-arm cache/output checks and graph replay are still required.

## Original K/V pipeline and resources restored

| Same-job producer observation | Control | Common-PV candidate |
|---|---:|---:|
| Registers/thread | 96 | 96 |
| Shared bytes | 20,480 | 20,480 |
| Driver raw spill field | 0 | 0 |
| PTX local accesses / SASS LDL or STL | 0 /0 | 0 /0 |
| TTGIR async global→shared copies/commits | 6 /6 | 6 /6 |
| PTX async wait sites | 2 | 2 |
| PTX barriers / butterfly shuffles | 18 /34 | 18 /37 |

Candidate TTGIR227–228 retains both`2x64x64xf8E4M3FN` K and V shared
allocations. Its eleven-value loop-carried signature at312 again includes two
K and two V async tokens plus the original compact address vectors, not the
rejected variant's large carried KV-address/mask tensors. At323, the combined
K/V wait retains num2. Both future K and V async copy/commit chains remain
present at426/430 and their two-slot buffer index progression is preserved.
The current V dot operand comes from its indexed asynchronous shared buffer
at355–357, not synchronous global V materialization before a conditional PV.

Thus this gate is supported by buffer/token/dataflow structure, not merely by
equal shared-memory totals. The exact source reformulation restores the
observed pipeline and resource outcome. That result supports the stated
conditional-consumer lowering hypothesis but still does not identify a unique
internal register-allocation cause for the earlier failure.

The additional three butterfly shuffles and integer/predicate work are real
static overheads. Unchanged registers/shared/spills preserve these static
resource bounds; no achieved occupancy, stall, bandwidth or latency conclusion
follows from the inventory.

## Exact merge and admission decision

Control and candidate merge SASS are byte-identical, SHA256
`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.
Both merges report27 registers,1,024 shared bytes and raw spill0, with identical
options. Debug/source records explain differences in other file formats;
the required exact native merge-identity gate passes.

All predeclared mechanism/resource gates pass for this specialization. This
admits preparation of the separately bounded frequency/correctness stage and,
only after its correctness gates, fresh uninstrumented paired whole-call
timing. Instrumented times must never become benchmark claims. It does not
admit a new official check, source selection or a performance claim by itself.
The working source and its official1.233633059× result remain unchanged.

The saved reconciliation reports app`ap-dUTLyzmcwfCtweeterml8r` independently
stopped/tasks0 at2026-10-25T22:28:32.831668UTC, with no active listed app and
paid$0. Worker wall time was30.6593s. These are receipt observations supplied
by the parent; this reviewer performed no service, GPU or credential action.
