# Serving integration audit

Research date: 2026-10-25. This is an artifact/interface audit, not a serving
patch, installation, model run or performance result. No serving submission exists.

## Current official artifact and platform status

Primary live sources: [public serving API](https://openkernelchallenge.dev/api/v1/serving)
and [official agent guide](https://openkernelchallenge.dev/llms.txt).
The API names the following contract:

- Artifact: a unified `git diff` against SGLang
  `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, normally `serving/patch.diff`.
  Standalone kernel Python submissions are components, not this final artifact.
- Only `python/sglang/`; exclude `python/sglang/srt/model_loader/`.
  Allowed suffixes: `.py`, `.cu`, `.cuh`, `.h`, `.hpp`, `.c`, `.cc`, `.cpp`, `.json`.
  No renames, mode changes or binary patches; maximum737280bytes (720KiB),
  added lines at most400characters. Prebuilt `sgl-kernel` binaries are not rebuilt.
- Base repository: https://github.com/sgl-project/sglang . Base image:
  `lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`.
- Serving is currently **closed**, opening2026-11-05 at12:00Pacific and closing
 2026-11-20 at00:00Pacific (Nov19 at11:59pm wording). API phase is `kernels`.
  Closed-track requests are refused. No attempt was made.
- Service caps:1scored serving run/day and3checks/day, reset00:00UTC. Full runs use
  an8xH200 node for up to90minutes. A check is concurrency16,reps1,two conversations
  per slot,rescore8conversations,35minute deadline. Full rungs are concurrency
  1/4/16/32,reps3/3/3/1,conversations per slot2/2/2/1. These are service limits,
  separate from current user authorization and the removed personal kernel ceiling.

Official workflow is `okc serve init`, edit the pinned checkout, `okc serve diff`,
then an explicitly in-scope serving check/submission. None of these commands was run.
The official [rules](https://openkernelchallenge.dev/rules) and guide require the
same model/norms/parameter names and shapes, top8experts,2048window, BF16 activations
where stock uses them, FP8 cache, released DFlash weights and exactly220output
 tokens. TTFT must not exceed2xstock at a load for that load to raise the score.
End-to-end serving correctness/throughput is a separate gate from kernel oracles.

## Launch and concrete integration boundaries

API launch uses target `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`, draft
`RadixArk/Muse-Spark-Open-DFlash2`,TP8/EP8,`modelopt_fp4`,Marlin,DFlash8,
hierarchical cache,16384chunked prefill,131072context,max32requests and decode
CUDA graphs through32. It does not explicitly set attention-backend or KV dtype.
Actual defaults, pools and active dispatch must be established from the pinned
runtime; the microbenchmark's FA3/FP8 layout cannot simply be assumed globally.

Projection's `prepare(weights)`/`kernel_fn(data,state)` and attention's
`kernel_fn(data)` are benchmark adapters. They need engine adapters, stock
fallbacks and graph-safe state ownership. Do not globally replace parallel-linear
modules and accidentally remove their biases, shape restoration or collectives.

Pinned primary source anchors:

- [BF16 dispatcher and optional addend](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/quantization/unquant.py#L397-L438).
  Add narrowly guarded implementations in allowed source files. Preserve bias,
  addend/aliasing, output dtype and unsupported-shape/device behavior. Weight-only
  layouts can be prepared after loading outside the forbidden loader directory;
  invalidate them when original weights change and warm them before graph capture.
- [Attention entry/boundaries](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L418-L459)
  and [backend metadata](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/attention/flashattention_backend.py#L70-L135).
  Map positions, query lengths/offsets, FA3 schedules, cache lengths and page tables
  from actual ForwardBatch/backend objects. Verify SWA translation, FP8 scales/
  strides, reserved padding slots and ragged verification. Preserve cache writes,
  norm/RoPE/gate semantics and prefill/unsupported fallbacks. Reuse model RoPE state.
- [O projection retains reduction](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L333-L342).
- [Rank-specific unpooler slice and gather](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/gemm_ag.py#L108-L130).
  `gemm_local` selects each rank's own weight rows; it is not `[:1024]` on all
  ranks. Keep all-gather/deferred-gather and boundary norms. The downloaded
  `official/baseline.py` explicitly omits this collective in the kernel benchmark.

Serving has no benchmark `prepare` API or transferable300second untimed allowance.
Account for startup, graph warmup, side-stream lifetimes and added weight storage.
For example, P039's pooler transpose retains64MiB per participating layer;
naively keeping62copies is3.875GiB per GPU, before router layouts. This is a
source-size scenario, not measured serving memory, and could displace KV capacity.

## Selected projection component and recommendation

Re-read local source/receipt and recomputed source SHA with `shasum -a256`:

- Selected P039: `candidates/projection_router_m1_fp32_gemv.py`, source commit
  `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`, SHA256
  `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
  Off-board job`d5636ac8d1404abd`,9+30pass,1.173238998969285x,
  CI[1.1556404139847458,1.1908375839538243]. Routerm1normalized latency cut30.1%.
  Its aggregate advantage over prior best remains unresolved by shifted controls
  and overlapping intervals. Evidence: `results/router-p039-official-assessment.json`.
- P040's higher point1.1758713266400707x does not change selection. It passes
  correctness but its6.5077%normalized O m1cut misses the declared15%goal;
  intervals overlap. Retain its rejected result without group-size sweeps.
  Evidence: `results/router-p040-official-assessment.json`.

One evidence-backed next integration task is P039's **complete router dispatch**,
retaining BF16 hidden states before stock's eager FP32 conversion. See
[router call and side-stream synchronization](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/models/muse_spark.py#L774-L845).
P039 expects BF16 activations; intercepting only after the cast would miss its
native gate. Preserve top8routing/correction logic, stream joins and lifetimes.
P039's m8router is7.49us versus same-run stock26.34us; m128is10.15versus16.24us.
These verify-related paths are more relevant to DFlash8 than incremental m1 alone.
Router/pooler overlap can hide savings: no serving throughput gain is established.
No further speculative projection GPU optimization is justified by this audit.

## Research method, attribution and remaining uncertainty

The web search tool returned inaccessible/empty content for the live API/guide.
Read those exact public URLs in the **user's Mac Chrome** using a temporary
agent-created research tab, then closed it. No login, terms acceptance, account
page, credential action, upload, file download or private browser-history read.
Pinned public SGLang source was read through the web tool, not downloaded or
executed. Local receipts/source were inspected read-only. No candidate edits,
experimental code execution, installs, GPU/check requests or pushes occurred.

No SGLang/model was loaded, no8GPU integration was tested, and no new patch was
created. Actual backend selection, real checkpoint layouts, graph padding,
ragged/DFlash cache behavior, collectives, memory footprint and stock-output
agreement remain integration gates. This audit does not reuse hardware resource
metrics from another kernel or convert microbenchmark gains into serving scores.
Code subsequently adapted from SGLang must retain its Apache2 attribution; existing
candidate license notices remain intact. This documentation update itself adds no
code from third-party implementations.
