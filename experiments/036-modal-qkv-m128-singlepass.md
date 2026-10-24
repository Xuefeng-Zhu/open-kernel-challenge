# 036 — full correctness and paired H200 screen, QKV m128

Exact035source, control officialbest fullcomposition c34c497b and suppliedSGLang.
One hypothesis: fullK output-owner removes split-K scratch/reduction. BM64/BN32/
BK64/GM8/warps4/stages3, no explicitTMA/broadrewrite. Nativepath only m128QKV;
other shapes/families/source unchanged. No inputvaluebranch or precision reduction.

Predefined screen-v1: all39CUDA cases each of3runners, including prerequisite
m512QKV. Nearby m127/128/129/256, N2304/2301 including nativeNtail;2freshgraphreplays
each, plus m128 stride1=2fallbackcase =>36extraoraclechecks. Collect nativecompile
resources/PTX after correctnesstest, no profiler.6warmups retained/excluded by
protocol then30steady samples per30shapes, six balancedordersx5, reversepairs,
all samples retained/no overhead subtraction, seed3000000.256MiB coldL2 before
CUDA-eventinterval. Fullfresh/probe mutability tests. CPU models are not CUDA.

Promotion criteria before dispatch: target m128 must consistently beat best,
whole-workload pairedgross aggregate meaningfully above1 rather than unchanged-
path noise, no material regression. Require at least13/15targetpairs faster;
review all distributions/perload weightedratios, not onlytarget. No officialcheck
if parity/regression. If warranted, onlyone officialoffboardcheck from remaining8;
commit exactsource/plan/CPU+Modal evidence and freshquota first, no scoredentry.

Existing cachedimage/12hauth,18allowlistedsource/licensefiles+manifest, no repo/
credentials. OneH200 CPU2/RAM8GiB max/min0/buffer0,scaledown2,retries0,function300s,
startup90s,internal240s/wrapper255s. Conservativeledger1.034495+0.60reserve<$2;
never decrease ledger because delayedbilling readsfall. MacUI cap0/credits29.33,
existingcredentialexpires06:26:47UTC Oct25; no newcredentials/payment/cap/push.

```sh
.venv/bin/python modal_preflight.py --candidate candidates/projection_qkv_m128_singlepass.py --build local-notes/modal-m36-bundle
.venv/bin/python modal_diagnostics.py --experiment M36-qkv-m128-singlepass --bundle local-notes/modal-m36-bundle
.venv/bin/python finalize_modal_diagnostic.py M36-qkv-m128-singlepass
```

Full112CPUtests+39contractCPU+static/syntax pass; GPUpending at preparationcommit.
All prior10Modal apps stopped; no active/uncertainofficialjob, secondcycle2/10used.
