# Independent source review: integer-only alpha frequency diagnostic

**Local source review passes.** The diagnostic is an integer-output
instrumentation of the exact common-PV candidate, with no added floating-point
arithmetic or feedback from counters into attention. This is not a review or
admission of the complete runtime harness, and no diagnostic kernel, service,
GPU, network or credential operation was performed. Only this report was
written by the reviewer.

| Reviewed file | SHA256 |
|---|---|
| Uninstrumented candidate | `14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5` |
| `diagnostic.py` | `ee5488bd8c6b8ee1e4bb9b6fccc1184215949136008dd15df11859b876f76c60` |
| `test_diagnostic.py` | `70cb800eb8cf08833170fc157d741a56822a10b4d91e05d7eca7b7f08416cdb0` |

## Exact integer-only delta

The five declared patches reverse to the **entire original candidate byte
stream**, not just selected floating expressions. The only changes are:

1. Add `stats_ptr` immediately after `out_ptr` in `_fused` and pass the explicit
   diagnostic tensor from the existing launch.
2. Initialize three scalar `tl.int32` counters at0 per CTA.
3. After the unchanged `all_alpha_one` predicate, increment total by1, fast by
   its int32 boolean value and slow by1 minus that value.
4. After the historical loop and before the unchanged new-token tail, store
   those counters into three explicitly owned output slots.

Counter values do not feed `acc`, `l_i`, P, row maxima, alpha, memory addresses
for attention, split decisions, lookahead or merge. The source does not add
atomics, persistent/module-global state, floating reductions or approximations.
Existing generic counter/ticket code elsewhere is retained verbatim; this
diagnostic introduces no new use of it.

The predicate is reused exactly rather than recomputed with a different
threshold or invalid-row rule. The counts describe historical iterations only.
The last-split new-token update is deliberately outside their scope. For an
empty historical chunk, initialized zeros are still written before the tail.
PAGE_LOOKAHEAD-false executions do not increment these counters; the planned
frequency cases must assert that the intended B32/L1/full/no-RoPE route is active.

## Ownership, bounds and independent count model

CTA `(b,kvh,split)` owns row`(b*HKV+kvh)*SPLITS+split`, columns0/1/2. For the
declared full launch grid this mapping is injective and exactly covers
`[B*HKV*SPLITS,3]`, without inter-CTA counter aggregation or an atomic operation.
The harness must supply a contiguous CUDA int32 tensor of exactly that shape;
the source's pointer stores rely on that explicit output contract.

For the fixed full-attention8192-token chunk and64-token tile, a CTA's expected
total is

`ceil(max(0, min(seq,(split+1)*8192) - split*8192) / 64)`.

Totals are0..128 and each fast/slow increment is0 or1, so int32 counter
overflow is excluded in these cases. Require nonnegative counters,
`fast+slow==total`, and the exact CPU expected total for **every** CTA. Aggregate
frequency should use summed fast/summed total, not an unweighted average of
CTA ratios that treats empty chunks as observations.

All five supplied offline tests passed independently. Additional boundary
enumeration checked0/1/63/64/65,8191/8192/8193,16383/16384/16385 and
90111/90112/90113 with enough shape-derived chunks; the sum of per-CTA counts
equals the independent historical range length and every count stays≤128.
These are CPU/source checks, not native execution evidence.

## Required complete-harness gates

Source identity does not guarantee identical generated floating-point scheduling
after adding integer loop state and output stores. Before using any frequency
record, the complete harness must therefore:

* Verify exact source/support hashes and the diagnostic tensor's device, dtype,
  shape, stride and separate storage. Retain fixed-address replay semantics.
* Reset/restore the counter tensor outside captured work as appropriate and
  validate every output row, including zeros for empty chunks, so no stale or
  uninitialized result can masquerade as a count.
* Run the uninstrumented14a candidate and diagnostic on the same fresh fixture
  using independent writable state. Compare their collected attention/new-KV
  outputs **exactly**, compare whole K/V cache bytes, and retain the unchanged
  original oracle, immutable-input and untouched-row gates.
* Exclude only the designated `alpha_stats` diagnostic output from that runner's
  immutable-input view; validate it separately. Do not weaken immutability for
  any original input, cache region or uninstrumented arm.
* Keep diagnostic capture, replays, synchronization, counter collection and
  exact-output comparisons outside **all** performance timing regions. Measure
  only the original uninstrumented control/candidate in the paired timing phase.

The diagnostic's source counters measure its computed predicate frequency.
They do not measure executed machine instructions, cycles, CTA residency or
hardware stall causes. The accepted uninstrumented common-PV compilation uses
predicated FMULs; a high fast fraction cannot be converted into eight fetched
instructions or fixed cycles removed. Instrumentation can perturb codegen and
timing even if outputs compare exactly. No instrumented timing or extrapolated
whole-score benefit should be reported.

No source-level blocker was found under the stated output contract. Native
comparison and the full harness's timing/immutability separation remain pending
independent gates, not properties established by these five local tests.
