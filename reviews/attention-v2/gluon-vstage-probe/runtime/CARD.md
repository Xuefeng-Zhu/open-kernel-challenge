# Prospective bounded V-stage whole-call diagnostic

Preparation is local only. No runtime job, controller upload, GPU launch,
official check, production dispatch edit or promotion is authorized by this
file. The parent must first review the current native artifacts and admit one
serialized run. A compile pass alone is insufficient.

## Question and immutable sources

Does the actual remaining V gather/conversion/publication placement difference
between the two `V_BEFORE_WAIT` specializations improve complete nearby calls,
and does either prototype compete with the actual selected FA3 call?

* Raw Gluon core `../kernel.py`: SHA256
  `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`.
* Fixed geometry `../config.py`: SHA256
  `686bee98df915e75ea938ba3c0cb691718527497fa5e5401ead36e9ef8b59fc0`.
* Actual working baseline `../../fusion-lookahead-probe/candidate.py`: SHA256
  `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
  Its B32/L8/full route is unchanged FA3 with fused preparation. No selected
  source is copied over or modified.
* Worker pins the original helper `modal_worker.py` (`9480d32a…68745e`),
  original `official/reference.py` (`386f7f67…1a37b`), original
  `official/checks.py` (`9a046c3c…bb8055`) and root licenses by complete SHA256.
  It also pins the three previously inventoried installed Gluon source files
  and exact Triton3.8.0 before importing the kernel.

The only native geometry is TP8 B32/L8, D64, HQ16/HKV2, GQA8, page1,
grid(32,2), M=N=D64, four warps, one stage, no RoPE/window. Both metadata
pointers are **int32**, matching the revised compile probe. Page-table stride
is fixed86008. The kernel itself and its resource/lifetime mechanisms are not
edited for runtime.

## Three timed call boundaries

`v_after_wait` and `v_before_wait` use the same adapter:

1. Allocate FP8 Q and invoke exact working `_fa3_prepare_fused` with
   `ROPE=False`, BM4 and the original scale flag. This performs Q RMS and
   BF16→E4M3 rounding plus all eight new K/V writes inside the call.
2. Convert that FP8 Q to BF16 inside the call. The core's existing
   BF16→E4M3→BF16 round trip preserves these E4M3 values. QK scaling remains
   once after the dot, including the live K scale.
3. Invoke the unmodified core with the live GPU page table, history lengths,
   K/V scales and caches. BF16 WGMMA QK/PV, FP32 online state and BF16 P remain
   the core's reviewed precision. The raw output is separate BF16 storage.
4. Invoke the exact working `rms_norm_gate` and return BF16 `[32,8,1024]`.

`fa3` calls exact working `kernel_fn`, including its own fused preparation,
automatic FA3 scheduler/splits, attention/combine and output gate. It does not
receive an artificial BF16 cast or altered scheduler. All preparation, cache
append, cast, core and gate device work remains in each captured call.

The adapters use different raw attention math from FA3. Original oracle
tolerances apply separately to every arm. No byte equality is required between
FA3 and prototype attention outputs. New K/V and whole cache bytes must match
all arms; the two identical-math prototype schedules must also match complete
output/newKV bytes.

## Fixed fixtures and legality

Fresh seeds are135751/135752 (A/B). Three cases only:

| Case | Nominal context | Distribution | KV scale | History lengths |
|---|---:|---|---:|---|
| Short tail, unit |129|fixed/soft|1|`[0,1,55,56,63,64,65,129]*4`, reversed for B|
| Short tail, nonunit |129|fixed/soft|1/64|same A/B vectors|
| Nearby performance fixture |4096|agentic/coherent|1|original generator's two fresh length vectors|

The first case covers total-key counts8/9/63/64/71/72/73/137, causal within-block
visibility and masked final tiles. The zero-history request's first new token
owns physical row0. Both boundary cases regenerate QKV, scattered mappings and
entire K/V cache for B; all fields are restored into stable graph addresses for
A1/B/A2. Positions, cache lengths and `cache_loc` are recomputed from the live
mapping. No metadata or output is cached by value.

Outside capture, each original int32 page table is padded to `[32,86008]` with
−1. Existing live page entries and physical pool capacity are preserved; padding
does not allocate historical K/V. Every entry through `seq+8` must be in the
physical pool and every unused suffix must be−1. `cache_loc` must equal the
original reference's `new_slots`. Length tensors remain int32; verify scheduler
metadata remains None. All arms receive this same legal static table shape.
The timed case is therefore **padded nearby4096 geometry**, not a scored80K
case or a result for an unpadded table. No80K fixture is allocated.

## Pre-timing admission and correctness gates

Before launching this worker, the parent must establish on actual current
compiled artifacts: a real remaining V-stage placement contrast, matched
logical work/common raw-K prefetch, safe current/next shared-buffer ownership,
strict precision and acceptable resources. Identical schedules, unsafe
lifetimes, spills/local-memory traffic or resource rejection stop this branch.
The worker cannot infer these facts merely from successful execution.

Within the one native run, **all** following checks precede any timing:

* Three arms × three cases × A1/B/A2 =27 calls to unchanged original
  `check_implementation`. Every result must be finite, retain immutable inputs
  and untouched cache rows, and retain tensor/output addresses. All nine A2
  returns must reproduce their own previously oracle-checked raw bytes.
* Across each of nine ABA groups: FA3 versus each prototype must have identical
  whole K and V bytes (36 comparisons total), plus exact newKV payload
  (18 comparisons). The prototype pair must have exact complete output/newKV
  bytes (nine comparisons). FA3 output is tested by the original oracle only.
* On the unit boundary case, a second independent `v_before_wait` capture owns
  its own original-oracle A expectation before separate-stream A/B submission.
  The two submissions call the original oracle (three additional calls,
  **30 total**). Both graphs have distinct outputs and writable caches; private
  pool handles are compared when exposed by Torch.
* Replay first→B while checking second's retainedB, then second→A while
  checking first's retainedB: two additional replays/four exact own checks.
  All expectations belong to the graph that established them. Actual device
  overlap and internal scratch pointer ownership are not observed; report
  these limits rather than claim a race certificate.

Any failed oracle, byte, immutability, address or isolation gate produces a
bounded partial report and forbids timing. No tolerance fallback, mask rescue,
new fixture or changed algorithm is permitted within this experiment.

## Prospective timing and separate conclusions

Only the already oracle-checked third case's A/B inputs are timed. There are
four total warmup rounds plus twelve total measured rounds: two warmup and six
measured rounds per seed, three arms per round (**48 checked replays**). Pair
members use the same fixture and reverse the complete three-arm order; cyclic
rotation balances position across three pairs per seed/six pairs total.

Restore and256MiB cold-L2 eviction occur outside CUDA event timing. Each event
spans the complete uninstrumented graph replay. No empty-graph subtraction,
instrumented kernel, profiler or counter session is used. Every warmup and
measured replay must reproduce its own original-oracle output/newKV bytes.
The prototype pair is checked16 times; FA3/prototype newKV is checked32 times.

Two predeclared comparisons are reported independently:

* **Mechanism signal:** `v_before_wait` versus `v_after_wait` must reduce median
  paired latency≥2% in **each** seed and win all six reverse-order pairs.
* **Actual baseline signal:** `v_before_wait` versus actual working FA3 must
  independently meet the same≥2% per-seed/all-six-pair gate. A prototype-pair
  gain while losing to FA3 is a mechanism-only result and rejects a replacement
  claim. An FA3 win without the mechanism gate does not establish this
  scheduling hypothesis.

No outcome automatically promotes a candidate or admits an official check.
Even both signals require later representative80K, other shapes/precision
contracts, original checks and control validation before a production claim.
There is no official-score or award-level extrapolation from this diagnostic.

## Envelope, stopping and ready state

The prospective envelope is one existing-image H200 job, CPU2/memory8GiB,
worker160s, child170s/function185s/startup20s/watchdog210s, zero retry. Preserve
partial atomic JSON throughout, bounded500000 bytes. Parent verifies fresh
free credit and actual paid cap0 before any launch; conservative free-credit
reserve is$0.58 using the earlier documented envelope, not an invoice or live
price quote. Stop app and verify tasks0 before releasing the serial slot.

There is no new controller in this directory. Default worker CLI performs
only source-bound CPU preflight; `--execute --output … --deadline160` enters
native work after external admission. Current-source compiled-artifact review,
final runtime source/card freeze, independent code review and parent/controller
serialization remain prerequisites. Local CPU tests prove structural guards,
not native numerical correctness or performance.
