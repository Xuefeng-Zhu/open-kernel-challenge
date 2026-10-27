# Independent preparation-fusion review

Source review passes for the dispatch-only candidate. Native correctness and
whole-call speed remain unmeasured; this review does not admit a GPU launch.

The reviewed worktree starts at `ebb80b91c58032641bf5341927261eb64929f14e`.
The complete candidate differs from its frozen A2-13 parent in exactly one
condition: the first full-attention fused-preparation guard now admits
`(batch, q_len)` pairs `(16, 1)`, `(16, 8)`, and `(32, 8)` in addition to the
existing `(1, 1)` and `(1, 8)`. Zero window and zero RoPE remain required. Full
decode B32 retains its existing custom path. No helper, FA3 argument, scheduler,
split setting, precision, cache format, output gate, import, or license changed.

| Reviewed source | SHA256 |
|---|---|
| `baseline.py`, byte-identical to `../mixed_fa3_fp8_qk.py` | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| `candidate.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Preserved selected A2-11 `../mixed_fa3_decode_parallel_merge.py` | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |
| `test_source.py` after support-binding update | `91c0d8f630ed44521a0e48afe80f920729ecdfbfa1885ae75f2923f973c03832` |
| `cpu-support-hashes.json` | `e38d7644d7d35207424b8cf73a790596a9597e519ad31472a022fdb53b7a0cd1` |
| `runtime/worker.py` | `b4b47a4729fb06b85995c115cb91492bc6d552f5a809653359e38a67cd35f08f` |
| `runtime/test_worker.py` | `9c1b08931c69252e2c9854dd549262fa943ea9ab5f306c58a1b987ade58971f5` |
| `runtime/PROTOCOL.md` | `e9c26c4979f9aa5a008953948fb845a4e3caa51094628336cb72d98aee17ea65` |
| `runtime/controller.py` | `5fa940cd1df79fabe0374b03149f18f404063f46fd1132dfa61799e9d96a0808` |
| `runtime/test_controller.py` | `acd8f658ae36ebdf6d1cbc5bccf94a6895f2daca522879c729ca9fc85b5dd6c9` |

Independent inspection of the unified diff and whole-module AST normalization
confirms the change boundary. Three `SourceGates` tests pass, including a
270-configuration nearby-route comparison, byte hashes, unchanged preparation
kernel AST, unchanged FA3 keyword expressions except the existing prepared-Q
expression, and unchanged final postprocessing statements.

The existing fused helper preserves FP32 RMS reduction, its BF16 normalization
boundary, optional BF16 scale-division boundary, and direct E4M3 conversion.
It writes cache rows from live physical `cache_loc`, including zero. Its
`ROPE=False` branch uses no RoPE table. All FA3 lengths, page tables, decode
scheduler metadata and non-unit descales are still forwarded unchanged.
These are source-level properties; native cast and reduction results still
require the planned comparison with ordinary preparation.

The CPU host-model test meaningfully checks the three new dispatch routes,
independent ordinary-adapter output, the unchanged official oracle, exact new
cache bytes, immutable inputs, untouched cache rows, live slot zero and
fixed-address A→B→A. It correctly labels its synthetic decode schedule as an
identity sentinel. Its launch model does not execute Triton indexing,
reductions or native FA3 scheduling. The support-binding update verifies eight
transitive CPU-model/reference/helper files before model imports; those exact
dependencies should also remain in the experiment export manifest.

The prepared runtime and protocol also pass local review. Six stdlib runtime
tests pass. The worker verifies candidate, baseline and official helper/oracle
hashes before GPU imports, then compares native preparation Q and complete
cache bytes for both fixtures of all six cases. A mismatch blocks timing.
It independently captures both complete entrypoints, replays A→B→A against
the unchanged official oracle, and submits independent-cache captures on
separate streams. The decode fixtures regenerate valid serving-supplied
scheduler metadata after live-length changes. No loaded candidate is patched,
and no FA3 setting is overridden.

The timing plan uses four warmup and twelve measured two-arm samples per
scored target. Adjacent reverse-order samples share one input version;
versions alternate between pairs. Input/cache restore and 256MiB L2 eviction
are outside the event interval; the entire captured kernel call remains
inside. Empty-graph calibration is retained separately. One session is not
an independent repeated experiment and does not establish an official score.

The final controller also passes independent local review and seven offline
guard/transport tests. Its eight-file allowlisted upload preserves the relative
directory layout used by the worker. Every payload file and the controller
must match an exact committed HEAD. Default preflight only reads Git/files and
parses code. The protected existing Modal config stays outside the upload;
the remote entry is closure-free, and no credential expansion is admitted.

Launch requires an explicit fresh parent slot for the same source commit,
sufficient expiry for the watchdog, live stopped/tasks-zero apps, live rates
and CLI paid spend zero, plus a fresh read-only billing receipt proving the
zero paid cap and sufficient free credits. Slot and receipt freshness are
rechecked immediately before reservation. There is one H200 invocation,
zero retries and no automatic relaunch after an uncertain outcome. Reviewed
bounds match the protocol: worker160s, subprocess170s, function185s,
startup20s, watchdog210s. The fixed free-credit reserve is $0.58 and rejects a
live doubled-rate-plus-slack bound above it. An explicit stop and independent
terminal reconciliation remain necessary after any real execution.

No source, runtime or controller blocker was found. Native equivalence,
correctness and performance remain pending. Exact committed freeze, fresh
guardrails and explicit serialized-slot admission are required before compute.
Counter collection and formal submissions are outside this experiment.
