# One bounded BF16-custom versus FA3 follow-through

This protocol is prepared locally. No execution or performance result is implied.
The earlier compile gate found BF16 QK WGMMA with strict FP32 accumulation, zero
reported spills and lower register/shared requirements. It did not establish
asynchronous overlap or a speedup against FA3.

The unchanged exact A2-13 `candidate.py` is the FA3 control (`kernel_fn`). Its
source SHA256 is
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
The experiment calls a separate `candidatecustomhelper.bf16_custom_fn` copied
from its dormant `_e2_kernel_fn`. A source test reverses global qualification,
function renaming and the one `QK_DOT=tl.bfloat16` substitution, then requires
AST identity. The original module and formal dispatch are never patched.
Per-call allocation and private zeroed ticket scratch remain inside capture.
The original producer owns new KV on the last split and uses its integrated
counter merge for L8; BF16 PV, strict QK accumulation and FP32 softmax remain.

Three cases, both arms, two independently generated inputs each:

1. Full verify B16/L8, page64, KV scale1/64, lengths
   0/1/63/64/65/8191/8192/8193 repeated twice and reversed for B.
   Metadata covers all eight new tokens; page-table row zero is explicitly
   exercised and unused page entries become -1.
2. Full verify B16/L8, 80K agentic contexts, page1, scale1, coherent distribution.
3. Full verify B32/L8 with the same scored-family configuration.

Seeds are 135701 and 135702. Each of six independently captured graphs replays
A→B→A at fixed tensor addresses. All 18 replays call the unchanged official
oracle, check immutable inputs and all untouched cache rows, and require the
second A's output/new-KV bytes to match the first A exactly. Every case must
pass before any timing starts. A passing result covers these representative
cases; it does not cover the whole official prerequisite suite or concurrency.

Then both 80K families receive four warmup two-arm samples and twelve measured
two-arm samples (six reverse-order pairs). Each adjacent reverse-order pair
uses the same oracle-validated input version. All samples are retained. Before
each measured whole-call graph replay, inputs/caches are restored and a 256MiB
buffer is read to evict L2 outside the event interval. Every timing replay is
checked against its arm's own oracle-validated exact output/new-KV bytes and
immutable/untouched-row requirements. The record retains raw times and separate
empty-graph calibration without clipping, per-sample ratios and reverse-order
geometric ratios. No official score or confidence interval is claimed.

A 110s worker alarm plus caller-provided 120s child kill, 135s function timeout
and 170s app watchdog bound this experiment. A timeout retains partial evidence
and forbids claiming completion. No tolerance is relaxed to fit the deadline.
If more than 15s remains after all validation/timing, one unchanged FA3 B32/L8
graph replay is profiled outside timing to collect complete CUDA kernel names.
Those names require source/template matching before identifying the scheduler.
Profiler failure is optional evidence and cannot repair a failed correctness gate.

The launcher must freeze every allowlisted byte, use one existing H200 image,
one container and one explicit invocation, verify free-credit headroom and paid
cap $0, then stop and independently verify zero tasks. There is no official check
or selected-candidate change in this protocol.
