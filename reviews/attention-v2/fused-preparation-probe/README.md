# Three-path preparation fusion: prepared, awaiting serialized compute

This experiment changes one dispatch condition in a standalone copy of A2-13.
Full decode B16 and full verify B16/B32 reuse the existing fused preparation
before the unchanged FA3 call. The fusion is absent from those routes in the
authoritative private baseline `ebb80b91c58032641bf5341927261eb64929f14e`.
Selected A2-11 is unchanged. This is a small incremental hypothesis; assuming
5–7 microseconds saved in each complete call models approximately 0.84–1.19%
overall improvement, not a measured result or a forecast.

The isolated worktree is `attention-fused-preparation`, on the branch of the
same name. No unrelated report or kernel work was merged. The candidate is a
byte-for-byte A2-13 copy except for the first dispatch guard:

```diff
-    if (cfg["batch"] == 1 and cfg["q_len"] in (1, 8)
+    if ((cfg["batch"], cfg["q_len"]) in ((1, 1), (1, 8), (16, 1), (16, 8), (32, 8))
             and cfg["window"] == 0 and cfg["rope_theta"] == 0):
```

| Source | SHA256 |
|---|---|
| Frozen A2-13 `baseline.py` | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| Standalone `candidate.py` | `5c3d1c5b00f0e759f1c499b8f567a0f47c77ce0fa38566f3028a571fccc614a0` |
| Preserved selected A2-11 | `c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0` |

All helper bodies, FA3 split/scheduler/precision arguments, scale descales,
BF16-before-FP8 boundaries and output gate remain unchanged. Full decode B32
retains the frozen parent's existing custom implementation. The source audit
explains cache ownership, live slot zero and exact source formulas. Native
byte equivalence remains a gate because compiler/backend arithmetic can differ
even when the source formulas agree.

Four local source/CPU model tests passed. They check the exact source hashes,
the complete AST except the first guard, nearby dispatch, unchanged FA3 and
gate expressions, and all three target shapes under page1/unit scale and
page64/1⁄64 scale. CPU models compare output and new-cache bytes against the
ordinary preparation with the unchanged official oracle, untouched-cache
checks, live slot zero and fixed-address A→B→A input changes. These tests
execute neither Triton nor FA3 and establish no native performance result.

```sh
/Users/frank/Documents/Codex/2026-10-23/task/.venv/bin/python -B \
  reviews/attention-v2/fused-preparation-probe/test_source.py
```

`runtime/` prepares the native correctness/replay/cache/concurrency and paired
whole-call protocol. Compute admission requires the final source/harness
review, exact committed payload, a parent-granted serialized slot, current
free-credit coverage and actual paid limit zero. Projection currently owns
the slot. This stage has launched no Modal app, profiler or official check and
has incurred no compute charge. No external Pro message was sent.

The intended order is strict: compare native query and entire post-store cache
bytes; pass unchanged official output/cache checks for the three scored
families and boundary variants; verify changing live metadata in A→B→A and
independent disjoint-cache calls; only then measure balanced cold-L2 whole-call
graph replays. Stop on semantic mismatch. Preserve all timing samples and
distinguish one-session evidence from a repeatable gain. Promotion still needs
the complete official prerequisites and scored shapes.

The official-check queue in `OFFICIAL_QUEUE.md` is conditional on evidence and
verified remaining quota. Unused quota is preferable to an uninterpretable or
duplicate experiment.
