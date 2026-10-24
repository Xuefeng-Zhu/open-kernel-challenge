# R10 — final controlled composition, tenth and last allowance

Source `candidates/router_pooler_hybrid.py`, commit
`5f491a34d52449f5d0c5c7985b4a0d4ecd17217d`, SHA256
`891b2063edde9ef9c508c2983e6cf10804bf5066fc036d63912c47ddeb8fc9ee`.
Configuration/hypothesis/provenance and local gates: experiment014. Relative to R07,
compose only independently measured split32 low-M router and m1 SIMT pooler; wide
large-M router stays R02. All native kernel bodies match their checked controls.
No new tuning, lower precision or regressing poolerSplit8/QKV path is included.

```sh
python3 remote_check.py send R10-final-composition candidates/router_pooler_hybrid.py
python3 remote_check.py poll R10-final-composition
python3 summarize_check.py R10-final-composition
```

Sole accepted check job `02e099c3724e4a05` at2026-10-24T16:28:45Z. Admission10/10;
**no remaining new checks**. Existing nine admitted jobs terminal/passing. No uncertain
request, paid resource, scored/public/final entry. Monitor this exact ID to terminal;
never launch a replacement. Full composition GPU correctness/performance pending.
Local39/39 full-size CPU cases +57/57 combined tests pass, which is not native GPU proof.

## Terminal actual result — entire budget complete

Sole job `02e099c3724e4a05` is **done/pass**,9/9 prerequisites +30/30 scored-shape
correctness checks, integrity checker ok with no flags/rejections. Official speedup
**1.1207320490× SGLang**,95%interval **[1.1039210683,1.1375430298]**. Original starter
comparison **1.2348217013×** is reference-normalized across separate runs, not paired.
Normalized score_us4310.3969447 is not a directly measured single-load latency.

All10 newly admitted checks are now terminal and passing; no uncertain reservation,
extra POST, paid resource, scored/public/final entry. Service quota11/20checks used
(includes baseline),9left;0/10scored used. **User allowance10/10,0left** remains the
strict admission ceiling regardless of service quota.

Actual targeted router m1/8:7.1789/7.4584µs versus R07's9.2852/9.1417µs. Router
m16/32/128/256:9.1754/9.4573/10.2562/11.0143µs, retaining the wide algorithm.
Pooler m1:18.1286µs versus R07's20.6142µs; all other pooler paths remain original.
Within-run SGLang-normalized ratios versus R07: router m1/8 improves1.2894/1.2453×,
pooler m1 improves1.1441×. See COMPARISON_R07_R10_ROUTER.md and _POOLER.md.

## Confidence and regressions

This is the highest **observed passing point score**, not a statistically proven
optimum. Its interval overlaps R07's[1.0827090,1.1156850]; independent repeated
validation/held-out/nearby GPU shapes were not performed, and no leaderboard entry
was made. All six weighted loads exceed SGLang in this trial, but decode32's paired
ratio is1.0705 versus R07's1.0833; verify32 is1.0697 versus1.0700. Do not attribute
all variation in unchanged paths to our changes.

R10's m8 router7.4584µs is slower than isolated R08's6.8717µs despite identical
native kernel bodies; the SGLang-normalized comparison also favors R08 there.
Preserve that evidence (COMPARISON_R08_R10.md), not just the best whole-workload
score. Shared image/evaluator hashes, full timing samples, profiles and GPU metadata
are in results/remote-R10-final-composition.json and CSV. A future repeat could
separate composition/layout effects from run-to-run variation; none is authorized
inside the exhausted budget. R07/R08/R09 sources remain unchanged for comparison.
