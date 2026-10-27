# P060 — baseline-versus-proposal matrix and methods correction

Local evidence audit only, 2026-10-26. No candidate, GPU launch, quota use or
new performance result. This report supports the user's requested research
handoff about **experimental method**, rather than adding another speculative
architecture branch. Selected P046 is unchanged; P039 is retained as rollback.

## Scope and scoring

Here **baseline means our selected P046**, not the official SGLang comparison.
Source `candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
`001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
Official job `41ba778763a5433b` passed 9 prerequisites and 30 scored shapes:
**1.1841399409×**, reported interval **[1.166377842, 1.201902040]**.
P048 independently confirmed the three pooler target improvements with 480
oracle comparisons; unchanged control median movements stayed below 0.7%.
This supports the working selection, not a new official score or confidence claim.

Each of six loads sums five family times, each multiplied by 62. The score is
the geometric mean of SGLang/P046 load ratios. It is not an unweighted mean of
30 shape speedups and does not use Attention's 46/16 mixture. All calculations
below hold the other rows fixed and use this exact official receipt.

| Family | Mean share of each load | Score gain if **every** family path cuts latency 10% / 20% |
|---|---:|---:|
| Pooler | 31.05% | +3.21% / +6.63% |
| QKV | 23.46% | +2.40% / +4.92% |
| Unpooler | 21.30% | +2.18% / +4.46% |
| O | 11.10% | +1.12% / +2.27% |
| Router | 13.10% | +1.33% / +2.69% |

These are conditional sensitivity calculations, not achieved/forecast gains.
Coverage matters: improving only QKV M128 by 20% models just +0.780% overall.
No fresh leaderboard lookup is made here; historical leader gaps must retain
their observation timestamp, provisional status and cross-run qualification.

## What the selected kernels already do

Source line numbers refer to the frozen file above. Instrumented durations are
single profile samples unless explicitly stated; they are not calibrated official
latencies. The complete selected official profile is
`candidates/profiles/projection_pooler_column_batch2_hybrid.41ba778763a5433b.txt`.

| Family / scored scope | Existing selected mechanism | Actual cost evidence | Proposed missing capability and cheap falsifier |
|---|---|---|---|
| QKV M1; M8/16/32/128; M256 | M1 uses one output-column Triton GEMV program, full K and FP32 reduction. M>1 calls BF16 `x @ W.T`. Vendor split-K plus reducer already exists at M8/16/32/128; M256 is one main kernel. Source 226–236, 283–285, 345–352. | Official M128 main/reducer 13.5/3.1 us. P059 same-run gross P046 21.840 us versus native FP32 split 21.856 us, paired 0.324% slower. Identical visible main/reducer names. | **FP32 split-K is not missing.** Any replacement must reduce main-path cost or reduction cost without losing useful parallelism. First reject if it only beats a restricted no-split control or repeats P046 symbols with flat whole-call timing. Exact vendor instruction/scheduling evidence remains absent. |
| Pooler M1; M8/16/32; M128/256 | M1 custom GEMV. M8/16/32 statically pack output-column halves and use one BMM writing disjoint output views; full K and a zero-stride shared X view. Other M>1 already use a contiguous prepared W transpose. Source 275–282, 312–350. | P046 M8/16/32 already use one vendor main kernel. M128 main/reducer 20.9/3.0 us; M256 one main 30.4 us. P053 M8 custom partial alone was 14.60% slower than complete P046. | **Static transpose and removal of the three small-M reducers are not missing.** A new high-coverage mainloop must beat actual P046 including its packing/dispatch. P053 already falsifies reducer-only rescue of that fixed custom route; do not restart it under a new instruction label. |
| Unpooler M1/M8 (N1024); M16/32/128/256 (N8192) | M1 custom GEMV; remaining paths BF16 `x @ W[:N].T`. N changes at M>8, so nearby M is not always the same workload. Source 283–285, 345–352 and official reference shape contract. | M8 has main 4.0 + reducer 1.6 us; M16/32/128/256 one vendor kernel. P057 unpooler16 native no-split near parity was 6/12 wins; zero-bias P049 regressed 3.59%. | **Generic launch fusion cannot remove a reducer from the large paths because none is listed.** A genuine main-computation or data-movement change needs a same-shape/same-output control. Test both N regimes; reject any extrapolation from QKV or an M16-only result. |
| O, all six M | Ordinary BF16 `x @ W.T`, including M1; no selected custom M1 GEMV. Source 351–352. | One vendor kernel on all six shapes, 4.5–8.2 us instrumented. P040 grouped-eight-output M1 improved its target only 6.51%, below its predeclared 15% gate, and was not selected. | A specialized small-M route is actually absent, but its broad score leverage is limited and one version already missed its gate. Require a different source-supported cost reduction and multi-load coverage; do not present removal of a nonexistent launch or a higher aggregate point as evidence. |
| Router M1; M8; M16/32/128/256 | M1 uses original FP32 weights in GEMV. M8 uses 32-way split-K, two BF16 hi/lo dot products with FP32 accumulators, then FP32 reduction. M>8 already combines hi/lo into **one wide GEMM**, then adds halves. Source 239–269, 286–291, 353–388. | M1 GEMV 3.7 us; M8 partial/reduce 3.3/1.4 us. Larger paths show vendor main + reducer + final half-add, whose activity is 1.4–1.6 us. | **Combining two GEMMs is already done.** Fusing the paired-column fold is a real missing epilogue capability, but legacy measured subtraction models only about 1.37% overall at best and mixes profile/official times. First prove the exact hi/lo arithmetic and no extra X duplication/mainloop loss; do not make this a gap-closing strategy by itself. |

## Resource evidence and what is still unknown

- Exact P046 pooler M8 in P053: 384 threads, 168 registers/thread, 164,308
  shared bytes, grid [4,16,1]. Nearby M7 uses 164,292 shared bytes and [2,64,1].
  These are launch/resource metadata, not measured occupancy or stall attribution.
- P044 QKV128 and unpooler16 were captured on **P039**. Their selected code
  expressions are unchanged in P046, but the metadata remains historical:
  QKV 168 registers, 221,492 shared bytes, 384 threads, grid [8,12,1]; unpooler
  168/180,692/384, grid [2,64,1]. Do not relabel them as a new P046 capture.
- P051/P052's `mma.sync` versus WGMMA code and 42→36 registers / 10,240→20,480
  shared bytes belong to custom candidates, not the vendor baseline. Outstanding
  copy/matrix scheduling in those candidates does not establish a capability
  missing from P046. The saved loop uses `wgmma.wait_group 1`, leaving a group
  outstanding while later copies issue; only the exit drains to zero. Calling
  this tested mainloop fully serialized would be incorrect.
- No saved exact P046 vendor PTX/SASS, measured HBM traffic, transaction
  efficiency, stall counters or achieved occupancy proves the proposed hardware
  bottlenecks. Two bounded NCU attempts timed out without metric values; absence
  of a permission-denial error does not make the counters available.
- Warm/cold-cache experiments changed queue/cache history together and sometimes
  made warm runs slower. They do not establish that HBM bandwidth is the limiter.

## Factual corrections for the methods handoff

1. **Rules were over-restricted.** Our earlier precompiled-only assumption
   excluded the global rules' explicit native CUDA `load_inline`/nvcc exception.
   P055 corrected our assumption narrowly; arbitrary subprocesses, downloads,
   harness access and altered timing remain prohibited. This was a methodology
   mistake, distinct from actual tool approval denials. Preserve the provenance
   and wording in P055 rather than claiming the entire rule changed.
2. **P057 proved an interface, not a vendor ceiling.** It passed 444 comparisons
   but only timed two of seven eligible QKV no-split algorithms under a
   self-imposed mask. QKV was roughly 194.7 versus 22.1 us. The assumption that
   these represented the useful vendor family was unsupported.
3. **P059 closes that mistake without a win.** It admitted an actual FP32 split
   (split2, ID66/tile27/custom2, 2,359,312-byte workspace), passed 298 comparisons,
   and matched P046 rather than beating it. The 51.804% cut versus no-split ID66
   co-varies tile20→27/custom0→2; it is not a clean split-count-only effect.
   P046/native main and reducer names match, but exact tactic/geometry identity
   is not proven. The 51.616-us native tail remains in all statistics.
4. **Comparator strength is non-negotiable.** P052's small win over its custom
   orientation control still lost to P046. P053 partial-only results rejected
   a reducer-only rescue. P043 compression lacked a GPU-tested same-geometry
   raw-weight control, so decode overhead and slow custom matmul were confounded.
5. **Information gain differs from optimization gain.** P056's compiler failure
   established a toolchain limitation; P057 fixed it; P059 recovered a needed
   policy. These are useful capability results, but none raises the selected
   score. Neither instruction names nor passes against a weakened control count
   as progress toward a leaderboard target.
6. **Selection needs confirmation.** P045's large-M loss was retained; P046
   narrowed dispatch, then P048 confirmed gains against P039 on one GPU. A higher
   point score alone did not select it. Tails, control shifts and failed gates
   must remain in the report rather than be filtered away.

## Minimum next decision, before another branch

No architectural proposal in this audit has yet demonstrated a missing broad
capability in the strong selected vendor paths. The method research should
identify the **minimum additional observation that distinguishes competing
causes**, before prescribing a broad CUDA rewrite or another tactic search.
For example, a native epilogue proposal must show a reducible existing phase and
preserve the competitive mainloop; an overlap/reuse proposal must establish what
the baseline already overlaps/reuses and where the candidate changes it.
Unknown vendor instruction behavior is an explicit missing fact, not evidence
that our proposed feature is absent.

Require: exact parent/source; one causal change; primary actual-P046 comparison;
matched secondary control; scored and nearby shape coverage; original oracle;
same-address fresh graph replay; raw paired samples with predefined warmup;
phase/instruction evidence separately from acceptance timing; score leverage
calculated before a run; fixed rejection criteria; and fresh confirmation after
selection. A single-shape diagnostic may establish a reusable mechanism, but
must not be sold as a whole-score gain. This is a gate for research quality,
not a request for another routine user approval.

Reproduce frozen scoring arithmetic without GPU/network:

```bash
python -B projection_baseline_matrix.py
python -B projection_p059_assess.py
python -B -m unittest tests.test_projection_gap_model -v
```

Both analysis commands completed locally; six existing score-model tests passed
in 0.010 s. Independent source/profile and arithmetic review found no factual
blocker. None of these checks executed native CUDA or made an external call.

Sources: P046 official JSON/CSV/profile and source; reports 039/040/044/046/048/
049/051/052/053/055/057/059; source-bound P057/P059 raw receipts. No new competitor
claims or research request were sent. P059 app is stopped/tasks0, paid cost $0;
its provisional workspace gross increment is $0.05740071, not a final invoice.

方法结论：先证明“现有最佳实现没有什么”，再设计“我们要加什么”。
当前五个家族的基线已经包含多种分流、静态布局、split-K 和归并；
重做这些机制、只胜过较弱自制对照，不能算整体竞争力进步。
