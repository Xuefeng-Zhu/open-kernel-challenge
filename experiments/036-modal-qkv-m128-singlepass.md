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

Terminal result: app ap-N9JWk0BvaWxhSdDYLxUJmL stopped/tasks0,worker56.403s,
app66s,exit0. Full39CUDA cases perrunner,36nearby/layoutoraclechecks andall30shape
6warmup+30steadyreplays pass. Frozenbundle40772dc; source035commit26189baa01e72d6e629f130a85f776347ca15a3f,
SHA84e86a5e6fff111220cc6ddc77d3b8885a00f7383b068f360699956de2d39132.
NativeactualPTX wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16,
46regs,0reportedspill,36864Bshared,144CTAs,noTMA. No explicitpipeline rewrite.

REJECTED: targetm128 candidate55.104usmedian,p9056.515,max57.248;
best21.888usmedian,p9022.499,max23.072. Paired best/candidate0.3954008,
0/15winningpairs. All30candidate samples slower than all30control samples.
AggregateGMofloadmedians0.9560322; verify16load0.7534077. Post-hoc descriptive
matched reversepairGM acrosssixloads median0.9551637,0/15wins; not a CI or a
separate promotion rule. Otherunchangedcases range0.9781–1.0342 due variation,
not attributable gains. Removed reduction fails to offset slower full-K geometry;
actual causal DRAM/occupancy/stall breakdown remains unmeasured.

No officialcheck admitted;2/10used,8remaining,platform13/20used7left/scored0.
No nextcandidate generated or broadrewrite. Keep negative/source/alltimings.
All11apps stopped. Latestbilling0.89fullycredited/net0; conservative stageguard
1.146411930, never reduced because asynchronous previousbilling was lower. Below
same2USDceiling, no credential/cap/payment/terms/push. Exact JSON/CSV evidence:
results/remote-modal-M36-qkv-m128-singlepass.json/.csv. Summary:
QKV_M128_SINGLEPASS_RESULTS.md. No pending/uncertain/deniedoperation.
