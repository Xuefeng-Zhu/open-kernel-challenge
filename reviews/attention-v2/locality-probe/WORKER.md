# Physical-page-locality runtime worker: local preparation only

No native execution, service, GPU request, authentication, upload or official
check has been performed by this worker task. Default execution is a CPU-only
source preflight. The future native phase remains pending the parent's existing
access, free-credit/$0-paid and serialized admission checks. Source/test success
is not a native result or permission to bypass unavailable access.

Fixed selected source is3da22a…; reference386f7f… and checks9a046c… are unchanged.
`worker.py` pins full SHA256 values and the pure-CPU fixture203240b0… before
vendor imports. It never modifies candidate code, native scheduling policy or
the original oracle. It does not import Modal, run a subprocess, install tools
or use network/counter APIs. Torch/Triton imports occur only behind `--execute`.

## Scope and exact input contrast

The three declared80K/page1/TP8/D64/unit-scale cases are B1/L1/sliding/fixed/soft,
B16/L8/full/agentic/coherent and B32/L1/full/agentic/coherent. Each uses two fresh
seeds135831/135832. One selected whole-call graph is captured per case and reused
for both layouts and both seeds. If a second seed changes any allocated tensor
shape/dtype/stride or config/None field, the worker stops rather than recapturing
or padding to force compatibility.

`fixture.py` constructs full-capacity inverse bijections. The runtime copies
opaque UINT8 cache pages to form canonical B, preserving logical FP8 bytes and
the original page-table shape/stride/capacity. Both calls retain a present page
table. Only physical cache rows, page-table values and corresponding cache_loc
values differ. QKV, logical lengths, positions, scales and provided scheduler
metadata remain byte-identical across layouts. Restores copy into the same
captured input allocations; graph source/arguments and output allocation stay
fixed. Actual input and output address/shape/stride/dtype states are checked.

The worker also proves the contrast was realized on attended historical pages:
movement must be nonzero, canonical attended adjacency must be fully contiguous
and original adjacency must be lower. Identity maps, movement only outside the
sliding window and already-contiguous request-block permutations are rejected.
Attended/moved-page and adjacency counts and complete int32 page-table hashes
are reported. These are address-map facts, not measured memory transactions.

## Correctness, replay and timing boundary

For each case/seed, both layouts first replay the captured source, pass the
unchanged original oracle, and pass immutable-input/non-new-cache ownership
checks. Full post-call caches must remain byte-exact after inverse permutation.
This includes unused pages and encoded new-KV updates. Each arm's own checked
output and new KV bytes become its exact replay reference. Cross-layout output
byte equality is recorded separately; it is not asserted merely from oracle
success. All later warmup/timing/activity replays must match their own checked
output/new-KV state and unchanged input/cache-ownership rules.

There are four warmup rounds and six reversed-order pairs per fixture, two
layout calls per round. Each of six case/seed fixtures has8 warmup and24 measured
replays. Totals are12 original-oracle calls,48 checked warmup replays and144
checked measured replays:204 successful checked graph replays. Attempted oracle,
attempted graph and completed graph counts are also tracked separately so early
failure cannot masquerade as no attempted operation. Partial reports retain
case, seed, phase, arm and round context.

The256MiB eviction read, full fixture restoration and all validation remain
outside CUDA event intervals. Events wrap only the unchanged whole-call graph
replay; no cost floor is subtracted. Six reversed pairs yield raw median times,
pair ratios and per-fixture summaries. No threshold, sweep, score extrapolation,
production promotion or hardware-bottleneck conclusion is built into the worker.
Nine warmup function calls and three capture calls are explicitly outside the
graph-replay counts. Every case uses the default private capture pool; observed
pool handles must be distinct. This is not proof of simultaneous hardware
overlap or disclosure of every internal scratch allocation.

## Optional one-session activity evidence

`--activity` adds exactly one Torch CPU/CUDA activity session with six fenced
replays: both layouts of each case on the second seed. Each replay is validated,
bringing the declared successful graph-replay total to210. Full kernel symbols,
graph/node IDs, grids, blocks, registers and shared bytes are retained without
symbol truncation. Native identity is compared as a multiset of nodes/symbols
and launch fields, not arrival order. Missing or invalid launch metadata is
UNKNOWN; two missing values never prove parity. Observed identity differences
stop the diagnostic. A fixed captured graph is used throughout; no forced
PackGQA/split override or different instruction geometry is introduced.

Raw Chrome trace is at most8MiB. A deterministic gzip/base64 artifact carries
raw/compressed hashes, and the complete JSON result is at most2MiB. Bounds
failure is explicit; the worker never clips symbols and claims complete
evidence. Instrumented durations and static launch fields are separate from
unprofiled timings, not achieved residency, HBM traffic, utilization or stalls.

## Local commands and future invocation

```
python -B -m unittest discover -s reviews/attention-v2/locality-probe -p 'test_*.py' -q
python -B reviews/attention-v2/locality-probe/worker.py
```

The second command imports no vendor package and reports zero launches/services.
Only a separately reviewed admitted H200 environment may use:

```
python -B reviews/attention-v2/locality-probe/worker.py \
  --execute --activity --deadline 160 --output /tmp/locality-result.json
```

The worker has a160-second alarm/deadline. External child/function/app watchdog,
resource budgeting, secret scanning, payload freezing and stopped/tasks0 checks
belong to the parent's future controller; they are not implemented as service
actions here. No remote controller has been reviewed by this task.

Fourteen worker CPU tests plus eleven fixture tests currently pass. Worker tests
cover source binding/mutation/symlink refusal, protocol/order/counts, raw pair
math, incomplete/nonfinite data, scalar byte comparisons, realized locality
contrast, exact timed interval, fenced full symbols, event-order independence,
missing/invalid launch metadata, empty names and exact bounded trace transport.
Synthetic arithmetic inputs exist only inside tests; no mock speed results are
exported as an experiment. Native API, graph/copy compatibility, oracle success,
runtime duration and any locality sensitivity remain untested.
