# Common PV: one compile falsifier, local preparation

This is a separate lowering hypothesis grounded in the rejected branch variant's
actual IR. It preserves that negative source and result. The only target
conditional contains the old-accumulator multiplication; PV is one common
unconditional consumer. The question is whether that permits the compiler's
original K/V pipeline and register allocation while still bypassing identity
MUL. The unique cause of prior register allocation is not proven.

Control SHA256 is
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
prepared source SHA256 is
`14a8fa2e3d7b64fa299fe69c8da6569501c069425a62e07cc5f11b1e9b15f3e5`.
Denominator, rowmax, exp/P, precision, lookahead, last-split new-KV, merge and
dispatch are unchanged. No approximate threshold, register cap or geometry
change. Seven source tests and independent semantics review are source evidence;
native equivalence and speed remain untested.

Compile only the same four main units: producer and merge for both sources,
full B32/L1 TP8, D64, M16/N64,8192-token chunks,11 splits, four warps/three
stages. All24 TTIR/TTGIR/LLIR/PTX/CUBIN/SASS files and exact source/options/
transport hashes are mandatory. No candidate or merge invocation, oracle,
frequency instrumentation, performance timing or official check in this phase.

The predeclared gates are unchanged:

- At most the same-job control's96 producer registers, no new spill/local
  accesses, original two-slot K **and V** async buffers/token pipeline. Equal
  total shared bytes alone do not establish preserved buffering.
- A real computed-alpha==1 bypass over all valid query rows, ignoring padding;
  no unconditional MUL followed by a select pretending to skip work.
- Fast PV C is old carry, slow C its unchanged alpha product, with the same
  PV accumulation order and one common logical consumer. Reject PV-from-zero
  plus final scalar addition, extra executed PV or changed precision.
- Unchanged denominator/tail/ownership and byte-identical merge SASS. Record
  branch/shuffle/barrier/layout/code-size changes without calling them measured
  stall, occupancy or latency evidence.

If any gate fails, stop this reformulation. Do not relax register bounds or
tune thresholds, warps, stages, splits or precision to rescue it. A pass would
only justify separately reviewing a fast/slow frequency diagnostic, followed
by original-oracle/replay/cache checks and fresh uninstrumented paired timing.
The current working source and official1.233633059× remain unchanged.

The reviewed envelope remains one existing-image H200/2 CPU/8GiB-host app,
zero retries,90s worker/100s child/115s function/20s startup/140s watchdog,
at most$0.40 free-credit reservation and actual paid$0. Source/worker/controller
must match a reviewed Git freeze. Use existing protected credentials through
supported config only, with fresh live free-credit/paid-cap/rate/app checks.
No install, renewal, permission, billing or visibility change.

The controller/worker clone changes only exact source hashes, experiment/app/
module/path labels and documentation. Five payload files exclude credentials
and receipts. Previously reviewed2MiB report/8MiB raw gzip, exact24 flat names,
per-file hashes, framing/bomb rejection, exclusive writes and10s installed-
cuobjdump limit remain. Stop and independently verify stopped/tasks0 before
slot release, preserve initial observations, and record a closed cost cutoff.

Projection owns the serialized measurement slot. This preparation has no grant,
request, upload, app or reservation. The previous alpha-one slot receipt does
not authorize this new experiment; only a new parent-coordinated slot can admit
the frozen compile falsifier. No formal entry or unchanged-source quota use.
