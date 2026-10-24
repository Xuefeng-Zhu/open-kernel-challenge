# QKV m128 single-pass experiment — rejected

Completed 2026-10-24T20:29:48.540681+00:00. One isolated hypothesis implemented and fully tested; no new
broad Hopper/TMA pipeline rewrite. Preserve official best full composition c34c497b,
job48b1357016134e52, **1.1486499523x SGLang**,95%CI
[1.1314202031,1.1658797016]. Original starter0.9076064x; reference-normalized
best/original1.26558 is an independent-run comparison, not paired. No rank claimed.

## Change and actual execution

Source [projection_qkv_m128_singlepass.py](candidates/projection_qkv_m128_singlepass.py),
commit26189baa01e72d6e629f130a85f776347ca15a3f, SHA256
84e86a5e6fff111220cc6ddc77d3b8885a00f7383b068f360699956de2d39132.
Frozen harness/bundle40772dc015eccdeae0f9c46ac1c5d61acf18598e.

Only QKV m128 changes to one full-K GEMM: BM64/BN32/BK64,GM8,warps4,stages3,
144CTAs at N2304/K8192. Original BF16 operands, FP32accumulation, BF16output;
no split-K scratch/reduction. Noncontiguous x falls back to original library path.
All other dispatch and prepare functions verified AST-identical to best. TASK
explicitly permits shape/config dispatch; other canonical/prerequisite/nearby m
use the preserved generic paths. No activation caching, precision reduction,
reference access or timing shortcut in the candidate.

Modal **ap-N9JWk0BvaWxhSdDYLxUJmL**,worker56.403s/app66s,stopped/tasks0.
Full39CUDA cases for each candidate/best/SGLang,36additional fresh nearby/layout
oracle checks (m127/128/129/256,Ntail2301,nativeN2304,stride1=2fallback) all pass.
Every30scored shapes timed with6predefined warmups retained/excluded then30steady
fresh-input replays; six balanced runner orders x5, reverse-order pairing,
256MiB cold-L2 read before event intervals. All outputs and samples retained,
no outlier deletion or guessed overhead subtraction.

Actual generated instruction is
`wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16`;
46registers/thread,0reportedspill,36864Bshared,144CTAs,noTMA. This is automatic
compiler lowering of ordinary tl.dot, not an additional explicit pipeline rewrite.

## Result: correctness pass, substantial performance regression

| QKV m128 gross graph time | Candidate | Preserved best |
|---|---:|---:|
| median |55.104us|21.888us|
| p90 |56.515us|22.499us|
| maximum |57.248us|23.072us|

Paired best/candidate **0.3954008**, **0/15winning pairs**; all30candidate samples
are slower than every control sample (candidate minimum54.304us >control
maximum23.072us). The predeclared13/15target-win gate fails decisively.

Whole-workload gross diagnostic: geometric mean of per-load median ratios
**0.9560322** best/candidate, verify16load **0.7534077**. Descriptive matched
reverse-pair geometric aggregation across six loads has median0.9551637 and0/15
winning pairs; this is post-hoc description, not a CI or changed admission rule.
No Modal number is an official contest score.

Other QKV shapes were unchanged; their paired best/candidate ratios were:

| m | paired ratio |
|---|---:|
|1|0.996139|
|8|0.999200|
|16|1.012921|
|32|1.005894|
|256|1.002886|

Across all29unchanged shapes ratios range0.9781–1.0342, reflecting measurement
variation rather than newly implemented gains. All samples remain in the full
JSON/CSV; no untouched-path changes were hidden or tuned away.

The tested single-output-owner geometry does not benefit from removing reduction:
its full-K kernel is much slower. It already emits WGMMA and reports0spill, so a
claim that adding WGMMA alone or fixing spills would cure this result is unsupported.
The causal share of memory traffic, effective occupancy, synchronization and long
K dependency was not measured. This result rejects this candidate; it does not
prove all single-pass or TMA implementations must lose. Do not generalize to an
unimplemented broad pipeline or infer competitors from these measurements.

**No official check admitted**, because the candidate fails the performance gate.
No replacement tuning/GPU app launched after the negative result.

## Exact checks and evidence

Local MacM2/Python3.12.13/Torch2.14.1:39CPUcontract+3focused+1nearby/stridedCPU
checks,static/syntax pass; complete local regression **112/112pass**. CPU numerical
models do not execute Triton. Actual CUDA evidence is separately recorded above.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python check.py --submission candidates/projection_qkv_m128_singlepass.py --device cpu --suite all --output results/router-cycle2-qkv-m128-singlepass-cpu.json
.venv/bin/python static_check.py --submission candidates/projection_qkv_m128_singlepass.py
.venv/bin/python assess_qkv_m128.py
```

GPU records: results/remote-modal-M36-qkv-m128-singlepass.json/.csv, compiledPTX
inside nearby-correctness record, and results/modal-M36-screen-assessment.json.
Exact predeclared launch/allowlist/protocol: experiments/036-modal-qkv-m128-singlepass.md;
source/CPU rationale: experiments/035-qkv-m128-singlepass-local.md. Historical
accepted launch must not be rerun; new independent experiment needs its own ledger.
Actual H200/132SM,public pinned image,Torch2.14.1+cu130/Triton3.8.0; separate
from private official evaluator. All three runners share this same app/GPU.

## Budget and closure

Official second-cycle **2/10used,8remaining**,0active/uncertain; daily freshquota
2026-10-24T20:26:39.185676+00:00:13/20checksused,7left,scored0/10. No rankedentry.
Original10projection and separate1attention budgets remain separate/exhausted.

All **11Modal apps verified stopped/tasks0**. Latest cumulative gross snapshot
**$0.89**,fullycredited,**netpaid$0**; observed highwater$0.89. Counters may settle
asynchronously. **Conservative stageguard$1.146411930/$2**, never reduced due to
lower delayed billing; remaining conservative headroom$0.853588070. Paid cap0
reverified in user's Mac browser before launch. Existing12hcredential unchanged,
expires2026-10-25T06:26:47Z; no newcredentials/payment/cap/terms/push.

Source/preparation/result commits separate, including failed attempts. Best and
original sources preserved. No operation denied or pending/uncertain in this
experiment. Technical limitation is that this concrete hypothesis is slower;
further architecture-aware work remains separate/unimplemented.
