# Conditional official-check queue

Prepared at the parent's request to use today's remaining quota for meaningful
tests. The parent is verifying the live quota and reset time and coordinates
all measurement slots with Projection. No official job has been started here.
Every official action must be `check`, with unchanged tolerances, no formal
entry and no duplicate/no-op variants. Readiness below is source readiness;
native correctness and performance gates remain pending.

1. **Three-path preparation fusion.** `candidate.py` extends only full B16
   decode and full B16/B32 verify to the existing fusion. First run the bounded
   native preparation/oracle/replay/cache/concurrency and paired whole-call
   protocol. If admitted, a complete official check answers whether the change
   remains correct over all prerequisite/scored shapes and whether the local
   improvement survives the official calibration. This candidate is implemented
   and independently source-reviewed. Expected official wall time is bounded
   by the problem's 15-minute job limit; actual queue time is unknown.

2. **Isolate a mixed result, only if observed.** If the native or first official
   result shows a repeatable gain in decode B16 but a regression in verify, or
   the reverse, prepare a dispatch-only restricted candidate for the winning
   family. Its question is whether the composition loses a useful gain; this
   is conditional causal localization, not a preselected extra check. Repeat
   the affected native gates and review its actual source before submission.
   Do not consume a check on this if the three-path result is uniformly useful,
   uniformly unhelpful or statistically unresolved.

3. **Sliding B16/B32 verify split-combine elimination.** This is a separate
   hypothesis from the completed Pro/source review: force supported FA3
   `num_splits=1` only on those two sliding paths, preserving preparation,
   precision and causal/window arguments. First verify that the *installed*
   wrapper supports the argument, then native correctness/replay and paired
   whole-call timing. The saved upstream signature is a lead, not proof of the
   installed API. Removing combine can lose parallelism (approximately 32/64
   useful packed tiles across 132 SMs); no speedup is assumed. This candidate
   is not implemented or admitted. Exclude B1. Use a complete official check
   only after useful local evidence, not merely because quota remains.

4. **A composition check only if two independent changes win.** If both fusion
   and the separately tested sliding split change pass their own gates and
   improve their target families, a reviewed combined source would answer the
   integration/correctness question over the full official set. Without two
   independent positive results there is no fourth meaningful candidate now.

Do not repeat the rejected BF16 custom verify route, dynamic reorder/planner,
NCU timeout attempts, arbitrary warp/stage sweeps or tolerance changes. This
queue does not authorize parallel jobs or a GPU launch before slot release.
