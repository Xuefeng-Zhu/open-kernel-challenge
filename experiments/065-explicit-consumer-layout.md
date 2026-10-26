# P065 — explicit consumer-layout decoder and matched raw computation

## Mechanism and eligibility

Parent mechanism: P043 lossless codec; P064's tested K×N integer expression.
Selected P046 remains 1.1841399409×. This is a bounded explicit-layout prototype,
not another logical orientation or tile sweep.

P064 still materializes decoded BF16 into shared memory. P065 instead creates
byte-load addresses, metadata broadcasts, integer reconstruction and BF16 values
in an explicit `DotOperandLayout(1, mma, 2)`. A uses its own operand-0 coordinates.
Both raw and compressed specializations then call `mma_v2` with FP32 accumulation.
Tile geometry BM16/BN64/BK64, S8, masks, exact decode and output reduction remain
unchanged. Raw weights use the same B-layout coordinates and direct-load loop.
The final accumulator-to-blocked conversion is output handling, not decoded-weight
staging, and is accounted separately.

The pinned official PyPI Triton 3.8.0 wheel contains Gluon's distributed/slice/dot
layouts and MMA-v2 API. Source review confirms the frontend accepts these generic
layouts for address generation and loading; backend feasibility remains an actual
compile gate. [TASK.md](../TASK.md) permits lossless static weight compression and
in-kernel decoding, with original BF16 precision and fresh activations. This uses
Triton APIs already present in the pinned dependency; no new dependency, compiler
patch, subprocess in the submission, evaluator access or credential is needed.
Exact official runner package identity still needs native validation if admitted.

**Pipeline distinction:** Gluon's compiler does not run ordinary Triton's automatic
coalescing/layout-selection/software-pipeline passes. `num_stages=2` does not
restore the old async pipeline. Thus this is a matched Gluon raw/encoded screen,
not a pure timing ablation versus P043/P064. A smaller shared request can include
removed buffering as well as removed operand conversion. Record both effects.

P046 already packs static pooler weights into two column groups and uses a full-K
vendor BMM for M8/M16/M32 with no separate reduction. P065 adds control over
reconstruction's consumer lane layout, not new decode fusion, static preparation
or launch reduction. Saved exact P046 vendor PTX/SASS is unavailable; we cannot
claim its underlying scheduling lacks any particular hardware feature.

## Predeclared gates

- CPU actual-source checks must preserve every BF16 representation, normal/escape
  blocks, K256 boundaries, nibble parity, rectangular axes, tails and empty splits.
  Both raw/encoded full partition models must satisfy the unchanged oracle on
  fresh finite inputs, sparse power-of-two rows and exponent outliers.
- Compile exact M16/N4096/K8192 with the existing pinned CPU-only SM90 recipe,
  BF16/FP32 matrix instructions, four warps, S8/BM16/BN64/BK64 and both specializations.
  Use `GluonASTSource` explicitly; ordinary `ASTSource` would select the wrong path.
- Require decoded BF16 shared roundtrip removed, no decoded global buffer, no
  spills and no comparable replacement conversion. Record raw/encoded register,
  shared, load-vectorization, scalar-load and scheduling costs. Distinguish output
  conversion from weight conversion. Direct register loads alone do not prove gain.
- Reject if unsupported or if no distinct consumer capability is established.
  If the gate passes with a defensible candidate, use a bounded official off-board
  check for original correctness and whole-call performance. Promotion requires
  support against P046, never merely improvement over the weaker custom raw arm.
  A missing same-GPU raw timing remains an attribution limit, not an invented result.

## Score relevance

M8/M16/M32 pooler is 14.456% of mean load time in the saved P046 receipt. A
hypothetical uniform 20% target reduction models +3.024% whole score; zero target
latency models 1.40515×. These are conditional sensitivity/ceilings, not achieved
or predicted performance. Broader coverage requires separate evidence. P062's
raw geometry was 28.5–41% behind P046, so a decoder repair alone cannot establish
competitiveness. The new port changes that consumer too, hence its raw control.

## Source and local checks

Owned kernel: `projection_p065_explicit_operand.py`, SHA256
`79fc79207f097ae016c907699c828820fc91c27c9d19c1fe5142b44e8188fd0e`.
The offline helper extracts only this reviewed kernel and required imports.

```sh
python -B -m unittest tests.test_projection_p065_explicit -v
python -B projection_p065_offline_compile.py --prepare local-notes/p065-offline
```

Five CPU tests pass. The source interpreter strips static layout descriptors,
implements loads/stores with bounds assertions, and computes FP32 dot on CPU.
It is not Triton's interpreter, tensor-core execution or validation of GPU lane
layout. All 65,536 BF16 patterns are checked in normal and escape representations;
finite full raw/encoded models are oracle-checked across M2/7/17/32/33 and K
1/65/257/513, with partial N tiles and empty splits. Native correctness remains
unverified. Independent review caught the wrong offline AST wrapper before any
execution; the helper now selects Gluon explicitly and records language/extension.
