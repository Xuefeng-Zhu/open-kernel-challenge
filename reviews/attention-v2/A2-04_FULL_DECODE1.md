# A2-04 — add full single-request decode to unchanged FA3 adapter

Parent selected `mixed_fa3_decode16.py`, sourcec94ec71/SHA8e1db32f,
jobbdd11e743365439a,1.0585245184× with CIabove1.0. Candidate
`mixed_fa3_decode_small.py` adds onlybatch1 to the full-q_len1 batch tuple.
Full decode32 remains E2; all eleven other scored dispatches/helpers unchanged.

Hypothesis: full decode1 C41.447µs vs same-receipt S31.776µs can improve at least
20%in reference-normalized latency through the already-validated full FA3 API.
Forward supplied scheduler metadata/lengths/offsets/page table unchanged. Full
has no RoPE, causal window=(-1,-1), all cached/new keys/values participate.
Official BF16norm→FP8query/cache (with scales), live row0 cache write, and BF16
attention output before RMSNorm/gate are unchanged helper behavior. No separate
precision alteration or third-party competitor source execution.

Exact frozen-other-times20%-cut/matching-reference models are in
A2-04-cpu-preflight.json; these are estimates, not achieved speedups. This target
has modest total weight; CIoverlap may leave whole-workload gain inconclusive.
Reject target goal if correctness fails or normalized cut<20%; investigate>5%
control regressions without treating cross-run shifts as automatic code effects.

Four new local tests pass:

```
.venv/bin/python reviews/attention-v2/test_mixed_fa3_decode_small.py
```

ASTproves all helpers and onlyone batch literal change;180nearby dispatch cases.
CPUstand-ins cover ctx0/64/3000 with coherent attention, TP8/DPheads, page1/64,
unit/nonunit scales, live row0/new cache/untouched rows, exact scheduler identity,
and same-addressA→B→A changing lengths/content/opaque schedule values. Scheduler
sentinel tests only forwarding, not actual native scheduling. No native execution
on this Mac; explicit target GPUABA/page64/nonunit/untouched-row/concurrency
coverage remains unverified unless receipt supplies it. Scored native path has
full coherent80Kcontext, page1, unit scales and actual supplied scheduler.

After exact source commit and free shared slot:

```
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-04 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-04 --cycle2
```

Attention7checks left before request; projection5remain unspent after bounded
P040review. No Modal/paid/newcredentials/terms/ranked entry. Reviewed private
auto-push is allowed. Never duplicate an accepted/uncertain request; preserve
source/result commits and all samples including inconclusive outcomes.


## Actual native result — retain target win, qualify aggregate delta

Job66b44d7586184708 accepted once2026-10-25T03:28:43.890482Z, source
28991e79b6062602705bc25463139eec8c154903/SHA5ef5c9e9, DONE16+12pass.
Score1.0770658900×,CI[1.0609099017,1.0932218784], parent1.0585245184×,
CI[1.0426466506,1.0744023861]. Relative+1.7516%is not statistically resolved
with overlapping intervals; both whole-workload intervals exceedSGLang1.0.
Target41.446812→31.669972µs, same-run S31.263998µs; normalizedlatencycut22.3375%
supports>=20%goal. All11controls source-preserved, no>5%latency regression.
Retain new shape-local benefit as nextparent, preserve oldsource/measurements.

20%-cut and matching-S models1.0774250/1.0808146 were conditional. Frozen-parent
controls with observed targetratio model1.0797943; actual1.0770659 also includes
reference/control variation. No extra run or assumption of causal totalattribution.
All12raw/adjusted/paired values and six46×sliding+16×full loads are retained in
A2-04-RESULT.md, -assessment.json, -per-shape.csv and -official-result.json.
Reproduce `.venv/bin/python reviews/attention-v2/assess_cycle2.py A2-04`.

Native scored full-decode1 output/newcache/fresh-input graph path with supplied
scheduler passes; explicit customGPUABA/nonunit/page64/untouched-oldrow/concurrent
stress not separately run. CPUstandins are not GPUevidence. No new prerequisite
entersB1/L1/full. No compiler register/shared/spill/hardwarecounter data captured.
Attention new4/10used,6left; platform14/20checksleft,0scored. Projection5left,
boundedpause. No Modal/paid/newcredential/ranked action; reviewed automatic
private push retains source/result commits and accurate inconclusive status.
