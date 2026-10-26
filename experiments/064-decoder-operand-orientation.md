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
AST outside the single native kernel. Native raw/encoded compilation is pending.
