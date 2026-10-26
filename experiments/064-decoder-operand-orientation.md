# P064 — decode in the dot operand's logical orientation

## Frozen experiment card

Parent: P043 encoded source, SHA256
`9598c7d1ebe6373b56f9ced2c0405b855f3a6895628f771c2eb8af314003b837`.
Current selected P046 is unchanged at 1.1841399409×. P064 is a mechanism probe,
not a selected optimization or a claim that the raw P043 backend is competitive.

Evidence: P063 exact-source offline SM90 codegen materializes the decoded BF16
operand in shared memory, transposes its descriptor and reloads it. Encoded
uses 120 registers and 18,688 bytes of dynamic shared memory; raw uses 38 and
10,240. Both have zero spills and BF16 HMMA. These are static facts, not measured
stall/traffic/occupancy attribution or judge-identical codegen.

Hypothesis: constructing the decoded tile as `[BK,BN]` from the same byte
addresses lets the compiler place reconstruction in the dot consumer's layout,
removing the extra shared transpose/reload. Only that logical orientation is
changed. Preparation, encoded format, scalar offsets, masks, int32 decode,
BF16 bits, FP32 accumulation, launch geometry, S8 partition, reducer and all
other paths remain unchanged. Raw still uses the original transposed load.
Candidate: `candidates/projection_pooler_codec_kn.py`, SHA256
`c9bff299644c9796af99dfe1e98fbe96cd409e558fc4ca5ac85b21abe4fe3149`.

Affected eventual paths: compressed, profitable pooler 2 <= M <= 32; scored
M8/M16/M32. There is no new numerical relaxation or activation-derived state.
The expression changes compiler layout opportunities, not logical traffic.

## Predeclared gate

1. CPU execution of the actual integer-decode AST must reproduce all BF16 bits,
   normal and escape blocks, mixed blocks, rectangular axes, tails and empty
   partitions. This is a CPU load shim, not the Triton interpreter or GPU.
2. Offline compile both raw and encoded at P063's exact M16/N4096/K8192,
   S8/BM16/BN64/BK64, four warps/two stages, explicit SM90 and 16-byte pointers.
3. Require removal of the decoded-operand shared materialization and related
   stores/barrier/reloads without replacing it with another comparable conversion.
   Preserve BF16 HMMA, full K work and zero spills. Registers/shared/instruction
   expansion should fall, without a compensating scalar-load/shuffle expansion.
4. Raw specialization must remain equivalent after accounting for names/debug
   locations. A moved/renamed conversion or a resource decrease alone is not a pass.
5. If this gate fails, do not issue an official check. If it passes, native
   original-oracle correctness and whole-call timing against P046 remain required.
   Never promote on speed versus only the weaker custom parent.

No achieved speed or physical bound is predicted. The earlier generous
storage-only scenario yielded 1.18060×, below P046, so repairing only this decoder
has no established competitive margin. A hypothetical 20% improvement in all
three selected target paths would yield only about +3.024% whole score; this is
an illustrative score sensitivity, not a forecast for P064.

## Local evidence before compilation

```sh
python -B -m unittest tests.test_projection_p064_layout tests.test_projection_weight_codec tests.test_projection_pooler_codec -v
python -B projection_p064_offline_compile.py --prepare local-notes/p064-offline
```

20 tests passed in 3.979 seconds: four new candidate AST/address checks and
16 existing codec/parent regression checks. Exact signed-zero/NaN payload bits
are covered as representation cases; this does not authorize NaN/Inf numerical
outputs. The actual integer decoder is executed through a CPU load shim with
bounds assertions; BF16 tensor-core multiplication is not executed. NumPy is
absent locally; the Torch warning did not prevent these tests.

Independent read-only review verified the candidate hash and unchanged executable
AST outside the single native kernel. At source freeze, native raw/encoded
compilation was pending; the completed offline result follows below.

## Offline result: gate failed; no official check

Frozen source `7579660` completed in the CPU-only `okc-p064-offline` container,
exit 0, with the pinned P063 image/wheel/toolchain and unchanged compile options.
Both variants produced hashed IR/PTX/cubin/SASS and assembler/resource reports.
Generated text is retained in `evidence/p064`; cubins remain in ignored local
artifacts with their hashes preserved. No kernel was loaded or executed.

| Static evidence | P063 encoded | P064 encoded |
|---|---:|---:|
| Registers/thread from ptxas | 120 | 80 |
| Stack/spill store/spill load bytes | 0/0/0 | 0/0/0 |
| Dynamic shared request, bytes | 18,688 | 18,688 |
| Static SASS instruction sites | 944 | 888 |
| PRMT sites | 162 | 33 |
| Scalar LDG.E.U8 sites | 18 | 33 |
| BF16 HMMA sites | 8 | 8 |

The raw control retains 38 registers, 10,240 bytes shared, zero spills and an
identical normalized SASS instruction sequence, including registers, predicates,
order and branch operands. Only headers/encoding comments are excluded from
that comparison. Instruction sequence SHA256:
`2bac291adaf10b752c1b27c9a06c94699dad3da4e33a3f1dca584efeea73d5c7`.

P064 encoded TTGIR lines 269–271 still allocate the reconstructed BF16 tensor
into shared memory and reload it into a kWidth 4 dot operand. The explicit
`memdesc_trans` disappeared, but its underlying materialization did not. SASS
still contains the corresponding shared stores/reloads. Treating the removed
view operation as removal of data movement would be incorrect. Other conversions
and scalar load sites also changed; lower registers/PRMT counts do not prove
lower runtime. Both variants retain the same BF16 HMMA/FP32 accumulation family.

**Reject P064 for official admission.** It fails the predicted shared-staging
removal and shared-request gates. The register/instruction reductions are useful
compiler evidence but do not waive those gates. No GPU correctness or timing
result exists. P046 remains selected; no check quota was spent.

```sh
python -B assess_p064.py
python -B -m unittest tests.test_p064_assessment -v
```

The three artifact/gate tests passed in 0.012 seconds. They cover the saved
rejection, source-valued shared allocation detection without a transpose, and
instruction comparison that preserves predicates/register/order. The assessment
verifies the generated file hashes, including the assembler resource logs.

## Source audit and next boundary

The pinned [Triton MMA pass](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L328-L395)
walks through pure elementwise/view operations and uses the smallest upstream
load width when choosing dot encodings. That is consistent with both decoded
variants retaining kWidth 4 from byte inputs. The [inline-assembly operation](https://github.com/triton-lang/triton/blob/v3.8.0/include/triton/Dialect/Triton/IR/TritonOps.td#L791-L805)
is itself elementwise; a proposed pure packed identity assembly operation
therefore lacks the claimed basis for stopping that traversal. It was not added
or benchmarked. This source reading is not proof of runtime cost.

A different follow-up would need explicit consumer-layout control, including
coordinates/loads and integer reconstruction, with a matched raw implementation.
Applying a layout conversion only after decoding can simply recreate the same
roundtrip. Such a port must expose changes to pipeline/load scheduling instead
of presenting them as an otherwise identical decoder. No broader port or new
candidate is claimed complete here, and a win over P043 alone remains insufficient.

## Cleanup and coordination

The exact container exited and was removed. Inventory showed no other containers
or Linux machines; OrbStack was restored to its initial Stopped state. No cloud,
credential, paid resource, official check or GPU operation occurred. Cached image
and wheel remain local; temporary installation/cache vanished with the container.
`docs/OFFLINE_SM90_COMPILATION.md` gives the reproducible setup and per-track
isolation rules so Attention can reuse it without touching Projection's files.
