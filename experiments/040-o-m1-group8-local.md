# P040: grouped-eight-output O m1 SIMT, local prototype

Standalone candidate candidates/projection_o_m1_group8.py. All existing candidates
and historical results are preserved. No check/ledger admission, commit or GPU run.

Parent: verified P039 candidates/projection_router_m1_fp32_gemv.py,
source commitcdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da, SHA256
0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc.
Official jobd5636ac8d1404abd passes9+30; point score1.173238998969285,
CI[1.1556404139847458,1.1908375839538243]. Its aggregate comparison to prior best
remains unresolved by overlapping CIs/shifted unchanged controls. Router m1's
normalized30.1%latency cut is separately documented and preserved here.

## One controlled hypothesis and exact source change

O m1 is N8192/K1024. Direct reuse of one-row _gemv would create8192small CTAs.
Grouping eight adjacent outputs per CTA can amortize that ownership/reduction
layout while keeping at most8192tile elements per CTA, the established long-K
SIMT work budget. Fixed eight is not an autotune/sweep. Performance is unmeasured.

Add one _gemv_group8: rows=pid*8+[0..7], K block=next_power_of_2(K); load current
BF16 X tile and eight BF16 weight rows; multiply/reduce K in FP32; explicitly
round outputs to BF16. Masks cover K and N tails. Each output has one owner,
no split buffer/reducer, atomics, input-dependent branch or activation/output cache.
Four warps, no other new configuration. Existing _gemv and ALL P039 helper/JIT
bodies remain unchanged, including the verified original-FP32 router path.

Untimed prepare adds only o_rows=weights['o'].contiguous() when O weights exist,
preserving original BF16 values. It may alias an already contiguous tensor.
The only new dispatch is layer='o',m=1. Gate requires0<K<=1024,positive N,
BF16 X/W on the same device, matching K/sufficient N rows and dense row-major
strides. Thus8*next_power_of_2(K)<=8192. K>1024,strided/unsupported metadata or
other device paths use the original timed x @ state['o'][:n].T library fallback.
All other dispatch is AST-identical to P039 after removing the one new branch;
prepare is AST-identical after removing its one O-state conditional.

No new imports/dependencies, static allowlist change, global state mutation or
precision reduction. Existing full Apache2 license comments remain embedded for
standalone upload. CPU target path is an FP32 product/sum model with BF16 output;
it never executes native tile/index/layout code.

## Source traffic rationale, not hardware evidence

Canonical one-row-per-CTA source counterfactual:8192CTAs,2KiB weights/CTA,
16MiB total weight requests,16MiB logical X requests. Group8 source prototype:
1024CTAs,16KiB weights/CTA,16MiB total weight requests,2MiB intended logical X
requests. Register/layout conversions may replicate actual X loads. Cache reuse
can satisfy repeated requests. Neither model reports measured HBM/cache traffic,
occupancy, bandwidth, spills or actual executed instructions. The counterfactual
one-row O kernel has NOT been benchmarked; it is not the actual library baseline.

Actual library O m1 already has ONE GEMM activity (~4.9us separate instrumented
profile in P039), so launch removal is NOT this mechanism. Latest adjusted O m1
latency6.65584163us comes from results/remote-P2-04-router-m1-fp32-gemv.csv.
Successful single-token SIMT pooler/QKV/unpooler/router provides bounded motivation,
not validation of this new two-dimensional reduction layout. Larger-M native
pooler/QKV and P038 TMA failures remain rejected; no failed geometry is reused.

## Bounded score scenarios and stopping policy

Freeze every P039 adjusted timing except O m1, with decode1 total44.95301165us.
O target5.5/5.0/4.0us gives aggregate+0.4351/+0.6275/+1.0201%. Even the published
optimistic3.5us floor gives only+1.2205%;1.5%aggregate would need2.81425582us.
These are arithmetic scenarios, not predictions or native measurements. A useful
shape-local gain is permitted by the user, but this change alone cannot reasonably
establish a statistically resolved overall improvement under ordinary+/-1.5%CI.

After coordinator review and immutable source commit, a separately admitted native
check may compare this exact candidate against P039. Preserve complete raw failure/
correctness/profile/timing output, all30shapes and CIs. Reject correctness failure,
target regression or mere small variation; retain untouched-control shifts rather
than crediting them to this change. No automatic tile/group/warp-count sweeps,
nonce repeat or further projection work without new measured mechanism.

## Local validation commands and limitations

Working directory /tmp/okc-projection-pipeline-preflight, existing primary Python:

```sh
PY=<primary-workspace>/.venv/bin/python
$PY -m unittest discover -s tests -p test_o_m1_group8.py -v
$PY check.py --submission candidates/projection_o_m1_group8.py --device cpu --suite all --output results/router-p040-o-m1-group8-cpu.json
$PY static_check.py --submission candidates/projection_o_m1_group8.py
$PY -m py_compile candidates/projection_o_m1_group8.py tests/test_o_m1_group8.py
$PY -m unittest discover -s tests -v
```

Environment MacM2/Python3.12.13/Torch2.14.1,noCUDA/noTriton. Known missing-NumPy
warning retained in local logs; no installation attempted. Focused6methods cover
canonical N8192/8189 dense/outlier/zero and exact probes at K0/63/64/511/512/1023;
264fresh/probe/zero/outlier combinations over K1/3/63/64/65/127/511/512/513/1023/1024,
N1/7/8/9/34/37; lossless noncontiguous O weight preparation; K1025/2048 and strided
library fallbacks; P039 router/all other dispatch equality and AST preservation.
CPU tests do NOT validate native masking, lowering, register use or CUDA graphs.
Native compilation/correctness/capture/performance remains UNVERIFIED. Main risks
are FP32 reduction layout/shuffles, register pressure, lower useful parallelism,
compiler replication and an already efficient existing library GEMM.

Six projection checks remain; no request is admitted by this local record. Primary
coordinator exclusively owns GPU/check scheduling and authorized private pushes.
No ledger edit, commit, GPU/OKC/Modal/install/auth/credential/push occurred here.
No operation was denied or uncertain. Durable actual local outcomes will be saved
in results/router-p040-preflight-cpu.json, with the39case CPU oracle separately.

Final candidate SHA256: 8ca6417ad5a01c96ae1a137e43e547fc804c43b2e657edfd00515bcb131d2604

## Actual completed local outcomes

39/39CPU official-oracle cases PASS; original weights/inputs unchanged.6focused
methods PASS,131/131full regression tests PASS; static/syntax/diff checks PASS.
CPU oracle source SHA equals the final source above. No source changed after
these checks. Full summaries are now saved in results/router-p040-preflight-cpu.json
and results/router-p040-o-m1-group8-cpu.json. There is no native GPU measurement.


## Coordinator bounded-test decision

Independent6focused methods/static checks pass; exact source8ca6417a verified.
Fixed grouping is an alternative to the measured one-library-kernel O m1 path,
NOT elimination of a launch or measured reduction of baseline traffic. P039's O
profile is one nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT sample4.9µs; its name does
not establish full internal tile configuration. Eight-row source requests are
compared to an UNMEASURED one-row SIMT counterfactual, not actual library traffic.

Target P039 official6.6558µs. Predeclare >=15%reference-normalized target latency
cut for a meaningful shape-local benefit; otherwise retain P039 and stop this
mechanism without a tile sweep. Even published3.5µs floor gives<=1.22%aggregate,
so no claim that total-vs-parent statistical superiority will be resolved.
Other29shapes must preserve source; investigate>5%control shifts and do not
assign cross-run noise to this kernel automatically.

Attention A2-02 is currently active; P040 may not dispatch until its terminal
receipt releases the shared GPU slot. Projection6checks remain before any P040
request. New explicit standing authorization allows reviewed private auto-push,
never raw local logs or credentials; GPU remains unverified at source commit.
