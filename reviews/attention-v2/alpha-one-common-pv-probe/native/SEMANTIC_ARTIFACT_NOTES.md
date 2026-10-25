# Independent fresh-artifact gate: common PV passes this compile stage

The exact common-PV source passes the predeclared **static compile admission**
for this specialization:96 registers,20,480 shared bytes, raw spill field0,
original two-slot K/V async pipeline, correct masked exact-one predicate,
conditionally executed numerator multiplies, one common PV carry-C chain and
byte-identical merge SASS. This is not a native correctness or performance
result. The returned result records zero kernel launches and no correctness or
performance testing. The reviewer performed local reads/offline analysis only.

Source bindings were independently checked against current bytes and exact Git
freeze`6df535d096aec0e76379b6795bb97b5956d0474b`:

* control: `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
* common-PV candidate: `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.

All24 compiler/disassembly artifacts match their returned exact size/SHA256.
The raw result SHA256 is
`de0efb07a8e46e6e261d16db249668619e7d7083c17c599478db5f2aff861e90`.
All following line numbers refer to fresh `native/artifacts/` files.

## Real rescale bypass uses predication in final machine code

Candidate TTGIR358–366 compares computed alpha with1 using unordered inequality,
ANDs the flags with `row_ok`, sums them and tests for nonzero. A valid-row NaN
therefore requests original rescaling; padded rows do not control the decision.
The compiler's generated name `all_alpha_one_247` has the inverted value
`sum != 0`, which correctly corresponds to source `if not all_alpha_one`.

TTGIR367–374 yields either old carry×alpha or unchanged old carry. The selected
carry is the C operand of **one common** BF16 PV `tt.dot` at378. There is no
PV-from-zero followed by a final carry add and no duplicate conditional PV
consumer in this lowered loop.

At PTX1717–1741, alpha comparison and three shuffle reductions select the
zero-sum path. Eight carry multiplies exist at1744–1751 only on the other
path, with both paths joining at`$L__BB0_4`. The common PV at780–815 uses the
selected registers `%r713–%r720` as C through its eight BF16 MMAs.

Final SASS lowers that source/PTX branch to **predicated multiplies**:

| SASS address | Actual operation |
|---|---|
|`0x25f0`|Compare valid alpha R53 with1, unordered-NE |
|`0x2630/0x2680/0x26d0`|Three butterfly shuffles for the row-flag reduction |
|`0x2730`|Set P0 from the nonzero flag sum |
|`0x2770/0x2790/0x27a0/0x27b0`|`@P0 FMUL` on carry R28–R31 |
|`0x27d0/0x27e0/0x27f0/0x2810`|`@P0 FMUL` on carry R36–R39 |
|`0x2cc0/0x2cd0`, continuing through`0x3620`|One common chain of eight BF16 PV HMMAs, C blocks R28/R36 |

With P0 false, those multiplies do not perform arithmetic/write the carry.
This is not unconditional multiplication followed by choosing an unscaled
result. It is also **not** a literal machine-code branch that jumps over the
eight instruction slots. Predicated-off instructions remain in the static
stream and may still consume issue capacity. Therefore skip frequency cannot
be reported as eight instruction fetches or a fixed number of cycles saved.
The new comparison/reduction overhead can still outweigh any dependency or
execution saving; only later uninstrumented paired timing can answer that.

The common PV has the same BF16/FP32 types, MMA geometry and K-fragment matrix
accumulation structure as control, with only its input carry selected. Both
producer PTX files contain22 total MMAs:12 FP16 QK, eight historical BF16 PV
and two tail BF16 PV. Neither uses WGMMA or native FP8 QK matrix instructions.
This structural check does not assert universal bitwise behavior for exceptional
values or replace native original-oracle testing.

## Original K and V buffering survives

Candidate TTGIR227–228 allocates separate
`2×64×64×FP8` K and V shared buffers, with slots0 and1 initialized in the
prologue. Both K and V copies are async and have their own committed tokens.
The loop at314 carries eleven values, including two K and two V async tokens.
Its wait at323 consumes current K/V tokens with`num=2`; indexed K and V loads
at324–325 and355–356 use the same rotating slot.

The next K copy/commit at423–424 and next V copy/commit at428–429 return their
tokens through the loop yield at430. Final wait/deallocation at433–435 covers
both buffers. Both producer PTX files have six commit groups and wait-group
values2 and0. This preserves the original two-slot K/V async structure, unlike
the rejected e66 variant's direct V load and single K-token pipeline.

No additional V representation materialized on both sides of a conditional
PV consumer is present here: V is loaded from the buffered tile and converted
once to BF16 before the single common PV. Equal metadata alone was not used as
proof of buffering preservation; the token/buffer/dataflow structure was read.
This does not prove identical cycle scheduling or achieved memory overlap.

## Denominator, tail and merge

Candidate TTGIR348–354 retains old_l×alpha plus the same P row reduction,
outside the numerator condition. PTX756–757 keeps `fma.rn.f32` in the common
block. SASS`0x3870/0x3880` updates R78/R77 by FFMA with the unchanged alpha
and P sums. Max, alpha and P remain their original computations; there is no
new deferred/threshold rescale or changed denominator live array.

The exact source new-token-tail AST equals control, SHA256
`a9154c3b7f8f04a9c4dab9fa3c53a21d80f8a6fe39eff8303bc701bb2c9eb9b7`.
Candidate TTGIR437–546 keeps the last-split tail, including unconditional
denominator/numerator rescale and tail PV. PTX2204–2214 retains its two FFMAs
and eight unpredicated carry multiplies, followed by tail BF16 PV at2301/2306.
It does not use the historical exact-one predicate.

The separate merge SASS is byte-identical across fresh control/candidate:
SHA256`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.
Producer tail instruction bytes are not claimed identical inside separately
allocated whole producer functions.

## Resource gate and interpretation

| Fresh unit | Registers | Shared bytes | Raw driver spill field |
|---|---:|---:|---:|
| Control producer |96|20,480|0|
| Common-PV producer |96|20,480|0|
| Both merges |27|1,024|0|

Independent static searches find no PTX local allocation/load/store and no
SASS `LDL/STL` in either producer or merge. These are static/driver observations,
not measured dynamic spill traffic or achieved occupancy. The fixed gates were
not relaxed after seeing these values.

Static PTX barriers stay18 and shuffle count rises38→41. PTX branch count rises
4→7, but final SASS has five BRA instructions in both producers because this
small new conditional was if-converted to predication. Both disassemblies have
1336 address-labelled instruction slots including NOP/padding, ending at0x5370;
that equality is not an instruction-execution or latency measurement. Total
shared bytes and register allocation are restored relative to e66, but the
predicate overhead and real alpha-one frequency remain unmeasured.

This compile covers only L1/G8/HQ16/HKV2/D64, M16/N64, SPLITS11, chunk8192,
PAGE_LOOKAHEAD=true, four warps/three stages on H200. Environment reports
Torch2.14.1+cu130, Triton3.8.0, CUDA13.0 and driver610.57.04. It is not a compile
or correctness proof for every generic DP/page/split specialization, and not
an assertion of equality with the official runner environment.

The compiled-specialization gates pass. Any next phase still needs its own
reviewed harness and bounded admission. Frequency instrumentation must remain
separate from timings, and original-oracle/cache/replay/boundary checks must
precede fresh-fixture whole-call timing. No onward run was started by this
review, no performance benefit is established, and no working selection changes.

## Fresh artifact hashes

| Artifact | SHA256 |
|---|---|
| Control producer TTGIR | `573720bb90b00b4b1f82e640373d19bbd12fbf528450295da87f1063933db894` |
| Control producer PTX | `692b99394b0f534bbdf1938f39460f7da37b9474ba17d019d4b8461e21967764` |
| Control producer SASS | `44dc14bbc9a1d2ab246146c7c76d76c9cb7f13b929ad4c900f18c3f3f021e8de` |
| Common-PV producer TTGIR | `3610b912316aadef1bbcdcbdeb871f239bbe6afc49cfda728086212f468035d4` |
| Common-PV producer PTX | `ef38dc8abeda9797234fc2dff7245b106518719a6c179b0e3867ee9fb61fcd44` |
| Common-PV producer SASS | `4a61b188fc06b893cb2a56842b4ddb7916bd2c645d8ce12fa02b3565b241dbf9` |
