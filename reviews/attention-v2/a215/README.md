# A2-15: page-address lookahead, local diagnostic candidate

2026-10-25. The prototype and ten local checks are complete. The subsequent
native diagnostic stopped before upload because existing Modal authentication
expired. A read-only official capability/quota audit completed; no new GPU run
or OKC check was sent. The source, tests and sanitized findings are prepared
for reviewed private delivery, whose success is verified separately. Selected
A2-11, promising A2-13, rejected A2-14 and projection remain preserved. See
[NATIVE_ATTEMPT.md](NATIVE_ATTEMPT.md) and
[OFFICIAL_CAPABILITY_AUDIT.md](OFFICIAL_CAPABILITY_AUDIT.md).

## What the A2-14 evidence actually establishes

The same-evaluator official receipts report producer activity of 311.1 µs for
A2-13 and 422.0 µs for A2-14; merge is 2.6 versus 2.7 µs. Full decode B32
whole-call adjusted latency regressed 33.9263% after reference normalization.
The profile locates the loss in the producer. It does not identify spilling,
occupancy, bandwidth or a particular native instruction as its cause.

Scoped inspection of the two receipts, log tails and saved attention/local-note
artifact filenames found no candidate-specific PTX, SASS, TTIR, TTGIR, cubin,
register, spill or shared-memory report. Historical M33/M34 resources belong to
projection. A09 failed to return its compiler probe and cannot fill this gap.
See [diagnosis.json](diagnosis.json) for source/receipt hashes and search scope.

The source does permit an exact *expanded scalar* work model. For a full
8,192-token chunk (`BLOCK_M=16`, `BLOCK_N=64`, 128 historical tiles):

| Denominator work/state | A2-13 | A2-14 |
|---|---:|---:|
| Persistent FP32 elements | 16 | 1,024 |
| Multiplies over the chunk | 2,048 | 131,072 |
| Additions including final reduction | 131,072 | 132,080 |
| Horizontal denominator reductions | 128 | 1 |
| Per-tile maximum reductions | 128 | 128 |

A2-14 adds 129,024 multiplies and 1,008 additions in this model. Its nominal
persistent denominator representation grows by 4,032 bytes per CTA. It retains
the maximum dependency, exponentials and accumulator rescaling. It therefore
did not eliminate the online-softmax dependency chain. FMA contraction,
first-iteration simplification, padded query rows and layouts can change native
work; these are neither instruction counts nor a throughput model. Logical
state size is not allocated registers, spill traffic or occupancy.

## One next mechanism, chosen from the weighted gap

Full attention occupies 84.45% of A2-13's decode-B32 weighted load. That path
remains 316.412 µs, so it warrants a focused producer experiment. Large full
FA3 paths also dominate other loads, but changing their algorithm would add
several new factors at once. The saved six-load analysis is in
[../strategy/weighted_analysis.json](../strategy/weighted_analysis.json).

The new candidate starts from A2-13, **not the rejected denominator change**.
The parent performs a dependent `page table → physical row → K load` sequence
at each historical tile. A2-15 exposes one next-tile page lookup early:

1. Bootstrap the current tile's masked int64 physical rows.
2. Load current K using those rows.
3. Issue the next historical tile's page-ID lookup as int32.
4. Perform the unchanged current QK, softmax, V load and PV.
5. Only then consume the next page IDs, cast/clamp in int64 and compute the
   next row vector to carry into the following iteration.

This is a scheduling experiment, not a bandwidth reduction. With page1,
each head/tile logically requests 256 bytes of page IDs and 8,192 bytes of FP8
K/V. No valid requests are removed. The carried row vector is 512 logical
bytes; prefetched raw page IDs are another 256 logical bytes while live.
Temporaries, `next_t`, rematerialization and layout replication prevent deriving
the register delta from these numbers. Bootstrap/final masked work may add cost.

Only `B=32, q_len=1, window=0, no RoPE` enables the change. The other 11 scored
paths retain the parent's source operations. All masks, QK/PV precision,
scalar denominator, softmax ordering, chunk8,192, tile widths, warps/stages,
new-token ownership, cache writes, merge and FA3 routes remain unchanged.
No new K/V payload prefetch, persistent state or extra global scratch is added.

E3 merely added `tl.range(num_stages=3)` and showed no meaningful target gain;
its native schedule was not retrieved. A2-15 changes explicit loop-carried
dataflow. It may still compile to equivalent or worse scheduling. Ordinary
`tl.load` is not a promise of asynchronous overlap; a source-order change is
not evidence of a hardware win. The current [Triton range documentation](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
distinguishes compiler loop attributes from launch stages, and its [load
contract](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
specifies that false-mask elements do not access memory. Those documents do not
establish the version or emitted schedule of the official worker.

A vector L1 engine was considered but deferred: it could avoid eight padded
query rows, yet changes reduction, data sharing, tensor-core use and rounding
together. There is no current native evidence favoring that broader rewrite.
No wide-tile, dual-head, query-hoist, materialized QK/PV or split sweep is repeated.

## Exact candidate and local checks

- Source: [`../mixed_fa3_page_lookahead.py`](../mixed_fa3_page_lookahead.py).
- SHA256: `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c`.
- Parent: `mixed_fa3_fp8_qk.py`, source commit
  `21fefb946524a25d8a06562dd28b631d49cd5175`, SHA256
  `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
- Parent official job: `c2fb7b23433f4888`, 1.2109908035×, provisional.
- Selected attention remains A2-11, 1.1876787561×.

**Seven source/address tests and three diagnosis tests pass.** Full AST
reconstruction verifies the precise declared difference from A2-13. Tests
assert the sole shape guard, no use of `next_pg` until after current PV, scalar
denominator preservation and unchanged entrypoint/math/cache code. Integer
models cover scattered page1/page64, physical page zero, padded/masked entries,
empty and partial chunks, 64-bit offsets, padded row strides, both KV-head
configurations, and lengths 0/1/63/64/65/8,191/8,192/8,193/80,000/81,920 plus
40 randomized configurations. Same-object A→B→A changes actual lengths and page
mappings. It is a CPU model, not captured CUDA-graph execution.

The first test run had a test-only AST selector error on tuple assignments.
The selector was fixed, and all tests reran successfully. Independent review
also prompted moving the next page result's first consumption after PV;
the final exact hash above passed the complete suite and independent review.
No native result was fabricated for either local draft.

This Mac environment is arm64/Python3.12.13 and has no installed Triton package.
No candidate was imported or executed. Native compilation, output/cache
correctness, source-exact ABA/concurrency, register allocation and performance
remain unverified. The 16 official prerequisite shapes do not exercise the new
B32 condition; a future ordinary check cannot substitute for target-specific
page64/non-unit-scale/nearby-length/untouched-row tests.

Run locally from the project root:

```sh
.venv/bin/python -B reviews/attention-v2/a215/test_page_lookahead.py
.venv/bin/python -B reviews/attention-v2/a215/diagnose_a214.py --self-test
.venv/bin/python -B reviews/attention-v2/a215/diagnose_a214.py
```

## What would falsify the next experiment

First obtain native compiled-code/resource evidence under a separate explicit
authorization; see [NATIVE_DIAGNOSTIC_PLAN.md](NATIVE_DIAGNOSTIC_PLAN.md). If the
compiler sinks the next page load or produces an equivalent dependency schedule,
the intended mechanism is not established and an official timing check is not
justified merely by different source text. If a new schedule exists but worsens
resource pressure or whole-call latency, reject this implementation.

For a subsequent authorized performance test, the predeclared useful target is
at least 10% normalized full-D32 latency reduction, all correctness gates and
no unexplained >5% control regression. Holding everything else fixed, −10%
models 1.228929× overall (+1.48%); −20% models 1.248910× (+3.13%). Neither is
achieved or predicted. A modest aggregate gain inside the reported intervals
would still need paired/repeated evidence before selection. No check is
authorized or reserved by this local plan.

This bounded phase ends with one concrete candidate and tested diagnosis.
No new algorithm or parameter sweep follows automatically. The remaining
prerequisite is approved native execution access and exact-source transfer,
not another local arithmetic claim. Existing credentials must not be renewed
or replaced without separate approval. No privileged counters are required.

## 中文实验卡

**问题：** A2-14 虽减少横向归约，却大幅增加分母更新和存活状态，目标延迟退化
33.93%；现有资料不能证明发生了 spill。**假设：** 提前读取下一块页表，把其结果
延后到当前 QK/PV 之后消费，可能隐藏地址依赖。**改动：** 只动 full decode B32，
保留 A2-13 的全部数值计算、分块和缓存写入。**结果：** 7 项源码/地址模型测试和
3 项诊断测试通过；没有 GPU 编译或速度结果。**下一关：** 先看原生代码是否真的
提前发出页表加载，再决定是否值得测时；不能把源码顺序当作流水重叠的证据。
