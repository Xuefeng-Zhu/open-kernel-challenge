# A2-15 native diagnostic: schedule changed; runtime benefit untested

2026-10-25. The single approved Modal diagnostic completed on an NVIDIA H200.
App `ap-TQMFJknnfH0UZbG8lOBcBe` was created at 15:52:13 UTC, explicitly stopped,
and reached `stopped` at 15:52:43 UTC. A separate read at 15:53:12 UTC verified
zero tasks and zero active apps. No new OKC check was issued.

**Decision:** the native scheduling-change gate passes, with a narrower mechanism
than the source suggested. A2-15 is worth a correctness/paired-timing experiment;
it is not a validated replacement. Selected A2-11 and projection are unchanged.

## Frozen inputs and actual coverage

The three-file payload was committed before launch in
`31010ce6b69d54767dc322a53b0cf6894b03d955`:

| Payload | SHA256 |
|---|---|
| A2-13 `../mixed_fa3_fp8_qk.py` | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| A2-15 `../mixed_fa3_page_lookahead.py` | `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c` |
| Executed `native_diagnostic.py` | `443241f373998e8fae85b2941c014b599e95be6abf344aef028c7316d17f741b` |

A2-15's prototype source commit is `a2c9a91b1dd347d11cb333b2f96b8973951cfa03`;
its experimental parent is A2-13, source commit
`21fefb946524a25d8a06562dd28b631d49cd5175`. Candidate bytes did not change.

Four specializations were compiled, loaded into the CUDA driver and disassembled:
producer and merge for each version. The synthetic specialization matches full
attention B32/L1, TP8, agentic context 80,000: page 1, page-table stride 86,001,
QKV stride 2,304, HQ16/HKV2/D64, BM16/BN64/BL16, eleven 8,192-token chunks,
four warps and three launch stages. Producer grid is `(32,2,11)`; merge grid
`(32,16)`. QK_DOT is E4M3, DOT is BF16, separate merge is enabled. A2-15 alone
has PAGE_LOOKAHEAD=True.

One-element aligned CUDA tensors supplied pointer types and alignment for JIT
specialization. They are deliberately unsuitable for kernel execution. The
program used `.warmup(...)` and `_init_handles()`, never a candidate launch,
`kernel_fn`, or a CUDA graph. **Candidate kernel launch count is zero.**

Consequently there is no new numerical correctness, cache-write, unchanged-row,
same-address A→B→A, concurrency, non-unit-scale, page 64, nearby-shape, whole-call
timing or score evidence. Successful compilation does not establish those.
The 16 ordinary official prerequisites do not exercise A2-15's new B32 guard.

## Observed environment and resources

Image `im-cPTbGpNiX3RarKResPgX7h`; NVIDIA H200, capability 9.0; driver 610.57.04;
Python 3.12.3; Torch 2.14.1+cu130; CUDA 13.0; Triton 3.8.0; SGLang 0.0.0.dev0.
The `sgl-kernel` package metadata lookup returned no distribution version;
candidate imports nevertheless succeeded. Do not assume this is identical to
the current official runner environment.

| Compiled kernel | Registers | Driver spill field (raw) | Shared memory |
|---|---:|---:|---:|
| A2-13 producer | 96 | 0 | 12,544 bytes |
| A2-15 producer | 96 | 0 | 20,480 bytes |
| A2-13 merge | 27 | 0 | 1,024 bytes |
| A2-15 merge | 27 | 0 | 1,024 bytes |

The candidate adds 7,936 shared-memory bytes (+63.27%). This does not establish
an occupancy loss or memory bottleneck. The raw driver spill field is not a
measurement of dynamic traffic. Both merge SASS files are byte-identical:
`69985cf3d726014b1e4242c61d8c40738cace85a71b831c6383e53c304f56bb4`.

## What the compiler actually changed

The source places the next page-ID load before QK, then consumes it after PV
(`mixed_fa3_page_lookahead.py`, lines158/160/171/175). This literal order does
not survive compilation.

- Parent TTGIR uses one shared page-ID buffer and one K/V tile each
  (`native_artifacts/a213.producer.ttgir`, lines 205–207), and waits for all async
  groups at line 281. Candidate uses two K and two V tiles (lines 223–224), direct
  page-ID loads in two layouts (lines 371–372), and a wait allowing two groups
  outstanding (line 319). Its prologue prepares two tiles; the steady loop
  computes tile i while preparing tile i+2. Shared growth is 8,192−256=7,936 bytes.
- Candidate TTGIR puts QK at 325, PV at 360, page loads at 371–372. PTX maps the
  page loads at 1613–1643 to source line 158, after PV at 1539–1574.
- SASS schedules those page loads **within part of PV**, not before current
  QK. Current QK is at `0x1d90–0x20c0`; two PV HMMAs are at `0x2c30/0x2c40`;
  page loads at `0x2db0–0x2f60`; remaining PV HMMAs at `0x3000–0x35e0`.
  First dependent clamps occur at `0x35b0–0x3660`.

Verified SASS load-to-first-use distances:

| Register | Load address | First dependent use | Static address distance /16 |
|---|---:|---:|---:|
| R49 | 0x2db0 | 0x35b0 | 128 slots |
| R48 | 0x2de0 | 0x35d0 | 127 |
| R47 | 0x2e30 | 0x35f0 | 124 |
| R46 | 0x2e50 | 0x3630 | 126 |
| R89 | 0x2f10 | 0x3600 | 111 |
| R62 | 0x2f60 | 0x3660 | 112 |

See `a215.producer.sass` lines 1467–1521 and 1723–1745. Intervening work includes
PV HMMAs and FP8 unpack/conversion. These static positions do not measure
elapsed cycles, dynamic overlap or hidden latency. Source staging is unchanged,
but the compiler-created pipeline is a coupled consequence of the dataflow edit;
a future timing difference cannot be attributed to page-load distance alone.

## FP8 operands do not mean native FP8 matrix instructions here

Both producers quantize Q to E4M3 and then promote Q/K to FP16 before QK MMA.
Candidate PTX 205–253 quantizes Q, 383–404 unpacks it, 786–906 unpacks cached K,
and 925–963 issues `.f16.f16` MMA. New-token K is unpacked at 1971–1985 with
FP16 QK MMA at 2014–2029. Parent has the same pattern for both cached and new
tokens. TTGIR QK operands are f16 at candidate 323–325 and parent 284–286.

Neither producer contains native FP8 MMA or WGMMA in this specialization.
SASS confirms FP16 `HMMA.16816.F32` for QK and BF16 HMMA for PV. Thus A2-13's
“FP8 QK” label describes quantized values, not observed native FP8 tensor-core
execution. This result does not determine why Triton chose that lowering or
what another specialization/compiler would emit. It does not prove the current
FP8 rounding is numerically wrong or explain the historical score by itself.

## Evidence, commands and transport

All 24 compiler/disassembly artifacts and summary.json were retrieved as bounded
chunks, and every SHA256 was verified. Raw evidence is 1,894,074 bytes; the
compressed transfer was 386,434 bytes in 24 chunks. The completion marker was
received. This avoided the historical oversized-result retrieval failure.

`native-modal-run.json` contains app state, exact uploaded hashes, the full
transport manifest and shutdown observations. `native_artifacts/summary.json`
contains compiler metadata. `native-analysis.json` contains reproducible offline
hash, instruction and resource checks. Two independent reviews agreed on the
narrow scheduling interpretation.

Local verification commands, all passed:

```sh
.venv/bin/python -B reviews/attention-v2/a215/test_page_lookahead.py
.venv/bin/python -B reviews/attention-v2/a215/diagnose_a214.py --self-test
.venv/bin/python -B reviews/attention-v2/a215/native_diagnostic.py self-test
.venv/bin/python -B reviews/attention-v2/a215/analyze_native.py
git diff --cached --check -- . ':!reviews/attention-v2/a215/native_artifacts/**'
```

The full staged whitespace check reports 476 compiler-emitted whitespace
diagnostics, all inside raw artifacts. Those bytes are intentionally preserved
for their exact hashes. The scoped source/report check above passes.

The first two are seven source/address-model and three diagnosis tests. The
executed diagnostic had six local guards. Post-run commit `aaa801a` adds a narrow
SDK warning filter and a seventh privacy guard: an expiry warning had printed a
credential identifier, not the secret. No identifier or credential is saved in
these reports. Other warnings remain visible. This local hardening was not
uploaded or GPU-tested; the executed harness hash above remains authoritative.

The launch entrypoint was `native_diagnostic.py launch`, with existing
`MODAL_CONFIG_PATH` pointing to the ignored owner-readable local configuration
and `MODAL_PROFILE=xuefeng-zhu`. Its persistent run ledger prevents a duplicate
launch. Reproduction of native compilation requires a newly authorized run;
the commands above only analyze existing evidence and local models.

## Cost and shutdown

Actual app lifetime was 30 s; client wall 30.4423 s; remote wrapper wall 20.4618 s;
worker compile/disassembly work 18.7851 s. These are setup/diagnostic times, not
candidate kernel latency. The app had one H200, CPU cap 2, RAM cap 8 GiB, max 1
container, no idle capacity, no detached deployment and no application retry
policy. A 60 s worker limit, 75 s function timeout, 20 s startup timeout and 110 s app
watchdog bounded the run. Only one remote invocation was made.

Before launch, the Mac browser showed $28.80 free credits, paid spend cap $0 and
net charges $0. The stopped/tasks0 result is independently saved. Accounting
uses the stricter remaining $0.212180198 and retains the prior $1.787819802.
The reserved $0.196609280 was released only after shutdown verification.

The conservative incremental model is
`ceil(max(30,30.442335458)+5) × 0.00130496 × 1.1 + 0.01 = $0.061676416`.
This gives cumulative $1.849496218 and remaining $0.150503782 under the $2 cap.
This is a cost allowance, not a settled provider invoice. Billing initially
remained at $1.20 gross / $0 paid due to reporting delay. At 16:00:27 UTC the meter updated to $1.23 gross / $0 paid: +$0.03 rounded
($0.02871331 in the ephemeral breakdown). Keep the larger model allowance;
this snapshot is not a final invoice. One 20-second read timed out before the
successful bounded follow-up; no credential change or GPU relaunch occurred.
The latest snapshot is in `native-modal-run.json`; never reduce
the historical high-water figure because a later snapshot is lower.

## Next justified test, not executed here

One official A2-15 off-board check can now test whether the realized pipeline
offsets its larger shared buffers and prologue. Preserve A2-13 as experimental
parent and A2-11 as selected best. The predeclared target remains at least 10%
normalized full decode B32 latency reduction, all correctness gates, and no
unexplained >5% control regression across all 12 shapes and six weighted loads.
Near-noise aggregate movement is inconclusive. Target-specific page 64,
non-unit scales, varied lengths, cache untouched-row and graph replay coverage
still need proper full-sized buffers and execution; an ordinary check is not
proof of those dedicated tests. No check or additional Modal run was started.

## 中文实验卡

**问题：** 提前读页表能否改变真正的 GPU 指令，而不只是源码顺序？
**改动：** 仅比较 A2-13 与 full decode B32 的 A2-15，四个内核只编译、不执行。
**结果：** 页表加载被排进部分 PV 计算，距离首次使用111–128个静态指令槽；
K/V 变成双缓冲，共享内存增加7,936字节，寄存器仍为96。原先“早于 QK”的
顺序没有保留。两版的 FP8 值都转成 FP16 做 QK 矩阵运算。
**结论：** 有了值得测速的真实排程变化，但还没有正确性或速度收益；继续保留
A2-11。应用已停止、任务数为0，费用按保守上界计入账本。
