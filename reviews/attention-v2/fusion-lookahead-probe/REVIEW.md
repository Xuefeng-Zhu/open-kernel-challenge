# Independent composition review

Source and local harness review passed. No blocking source, ownership,
correctness-protocol, timing or launcher issue was found. This supports an
immutable local freeze and committed-payload preflight for one bounded native
test after current serialized-slot/free-credit/paid-zero admission. It does not
establish native composition correctness, performance, official-check readiness
or selected-version promotion. This reviewer performed local reads, offline
tests and this report write only; no service, browser, credential or compute
action was performed.

## Reviewed source isolation

The control is the validated fusion source, SHA256
`5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0`.
The candidate is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.
Independent whole-module AST comparison confirms exactly `_fused` and
`_e2_kernel_fn` change. Both new definitions match the measured A2-15 donor
`fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c`,
including decorators; restoring these definitions reconstructs the entire
fusion-control AST. Every other function, import, global, module docstring and
fusion entrypoint remains unchanged. There is no `num_splits` override.

The shape/config-only flag enables lookahead only for full B32/L1 without
RoPE, retaining the donor's supported head/page/scale scope. The new DP smoke
addresses the generic-head true guard. Pruning false constexpr branches
recovers the control's historical operations; this is a semantic/source proof,
not a claim of identical generated binaries on all unchanged specializations.

Q/K encoding, both explicit `max_num_imprecise_acc=0` QK sites, BF16 PV,
FP32 scalar softmax/partials, masks, splits, warps4/stages3 and the separate L1
merge remain the measured variant. Lookahead loads historical page IDs only,
with masked tail/bootstrap reads; it introduces no K/V payload prefetch or new
token pre-read. The original last split still owns new K/V even with empty
history. Per-call outputs and partial allocations remain private; the L1
ticket branch stays inactive under separate merge. No data-derived cache,
cross-call plan or new scratch lifetime was introduced. Original A2-11's
selected source hash is preserved by the source tests.

## Local verification

All **26 test methods** passed independently:

| Suite | Methods | Scope |
|---|---:|---|
| `test_source.py` | 8 | Exact source/donor binding, complete false-branch reconstruction, unchanged fusion entrypoint, address coverage, live zero/negative-padding parity, 64-bit offsets, fresh mappings and boundaries |
| `test_model.py` | 4 | Current score reconstruction, frozen evidence, signed/monotonic transfer arithmetic and threshold inversion |
| `runtime/test_worker.py` | 7 | Five declared cases/fresh seeds, boundary capacities, balanced sample order, failure-closed source/timing and 4% replication/control gates |
| `runtime/test_controller.py` | 7 | Payload/remote-layout, exact commit binding, transport integrity, slot/budget/stop guards, reserve model and no credential closure |

The source/address tests ran in 1.209 seconds. CPU integer/address models and
AST tests are not Triton compilation, native correctness, graph or performance
evidence. The runtime/controller tests import standard-library protocol code
only; their mocked-commit checks do not replace the later actual committed
payload preflight.

## Native harness and timing review

The worker verifies source/support hashes before GPU imports, then invokes the
two complete entrypoints without module patching. The five cases match the
plan: scored full B32/L1, target page64/scaled chunk boundaries, B4 unchanged
custom fallback, DP one-chunk smoke, and unchanged full B16/L8 fusion control.
Seeds are the predeclared fresh135711/135712 pair, distinct from the historical
A2-15 fixtures. Boundary generation rebuilds lengths, positions, cache lengths,
physical new-token locations and valid serving scheduler metadata at fixed
shapes; physical page0 is made live and unused suffixes are negative. The
custom full-decode path may ignore supplied scheduler metadata as permitted.

Both sources receive A→B→A oracle checks for every case:30 calls and10 exact
return-A checks. Entire K/V caches must match across arms after each replay
label, with immutable and untouched-row checks. Attention outputs use the
original adaptive oracle and their own deterministic repeat checks, rather
than requiring cross-source bit identity. Two additional target-boundary
candidate graphs use disjoint writable caches and private default graph pools;
separate-stream submission adds two oracle calls. This establishes proposed
independent-call coverage, not measured hardware overlap. All32 calls precede
timing. No fallback/DP concurrency coverage is claimed.

The target and unchanged control each retain four warmup and12 measured
two-arm samples. Adjacent reverse-order pairs share one fixture version, and
versions alternate by pair. All64 replays are checked against their own
oracle-validated bytes. Full inputs/caches are restored and256MiB L2 eviction
runs outside CUDA-event timing. The measured graph includes complete device
work, private partials and merge; replay reuses captured memory and excludes
host allocation. Empty-graph diagnostics remain separate and unclipped.
No profiler or hardware counter operation enters this test.

The benefit calculation correctly uses the median of six geometric pair ratios:
`reduction = 1 - 1/median_ratio`. Official-check consideration requires at
least4% target reduction,6/6 favorable target pairs and at most5% median paired
control slowdown. The new4% criterion is a declared composition/generalization
replication question. It does not relabel the old failed10% architectural
goal or isolate page-ID causality; the donor also changed compiler-generated
K/V buffering. One session, two fixtures and six pairs do not yield an
independent-session interval or an achieved weighted score. The conditional
whole-score model is modest, not a prediction.

## Launcher review

The eight-file upload allowlist contains only worker, two candidate/control
sources, unchanged buffer helpers/reference/checks and legal files. Repo-relative
mounting preserves the worker's `PROBE`/`V2` layout. Controller bytes are also
checked against the exact reviewed HEAD, though the controller is not uploaded.
The default CLI is offline preflight. The launch path requires a fresh parent
serialized-slot record for the exact commit, fresh free-credit/$0 paid-cap
evidence, live supported CLI paid-zero/app-stopped/rate checks, and the existing
owned mode600 non-symlink config through supported `MODAL_CONFIG_PATH`.
Credentials, receipts and local configuration are not serialized into the
remote function or payload. No renewal, installation or expanded credential
environment is introduced.

Bounds remain worker160s, child170s, function185s, startup20s, watchdog210s,
one H200/2CPU/8GiB invocation and no retries. The fixed$0.58 free-credit reserve
must cover twice the live full-watchdog rate plus$0.02; actual paid authority
remains$0. A unique local request record prevents duplicate dispatch. Results
are byte/hash checked under a500KB transport bound. Finally-stop and independent
state/billing reads preserve an uncertain or stopping outcome for reconciliation;
such an outcome must not release the slot or trigger a retry. Native correctness
completion and the separate benefit-admission field remain distinct.

## Exact reviewed files

| File | SHA256 |
|---|---|
| `runtime/worker.py` | `03c1261d8cb18b9c2ea266556463d087ee83d2da8cc121e1be80ff04fdc3cb15` |
| `runtime/test_worker.py` | `c07f07dc5b8efce19a794f0f4dc5b4582b168069d847c4ece6f296be1b570409` |
| `runtime/PROTOCOL.md` | `540e6946bdb92cc87c32cd38e0b833f2e937ee299bf0bda0958a19cb60eb5c31` |
| `runtime/controller.py` | `0b78f5a5147030f67701dcaf1298ba5077b947e133c6485478931e938014cbf1` |
| `runtime/test_controller.py` | `184c7a812f722ede16d9e6e18cbe3b86ab7feb77e7f16536f916a66715a5141d` |
| `test_source.py` | `a9d01af00f3c007b1fc1870da9d7db4a343a522a9eceec70effd243e9d6e0fa1` |
| `model.py` | `a175a4e09ed12285967d439ce2606a5736b13dec46cfd99c2fde4979c81bf66d` |
| `PLAN.md` | `251829926e4c81b9e03be19e2381167ac16aaf3f6a99b0dc43bcd87e0f2c630a` |

No remaining local review blocker was identified. Actual native correctness,
resource/compiler behavior, fresh guardrails, terminal resource closure and
measured benefit remain execution gates owned by the parent. No official
check or promotion follows from this review alone.
