# A2-15: official capability audit and bounded decision

## Actual observations

At **2026-10-25 08:36:21 UTC**, supported read-only OKC commands succeeded
with existing authentication. The sandbox's initial DNS attempt had failed
before a request; the permitted network read then succeeded. This was not a
candidate submission. There are no new accepted or uncertain checks.

| Problem | Checks used / limit | Checks left | Scored used / limit |
|---|---:|---:|---:|
| Attention | 16 / 20 | 4 | 0 / 10 |
| Projection | 3 / 20 | 17 | 0 / 10 |

All 37 jobs returned by `okc jobs --json` were terminal. This is an observed
recent-job list, not a claim about unlisted history. The daily UTC reset is
2026-10-26 00:00; these counts remain a timestamped snapshot.

## What the supported interface provides

Inspected `okc 0.7.1` help, `check --help`, `status --help`, `okc api`, local
runner tools, and saved exact A2-13/A2-14 receipts/profile JSON. The documented
API lists job results and the CLI offers JSON/wait/no-wait. Existing receipts
contain correctness, paired whole-call timings, GPU kernel-duration profiles
and log tails. None of the inspected interfaces or saved fields exposes
PTX/SASS/TTIR/TTGIR, register allocation, spill counts or shared memory for
these exact candidates. This is absence from the inspected interface/data,
not proof that the service could never support such artifacts.

The [official CLI/API guide](https://openkernelchallenge.dev/llms.txt) could
not be retrieved by the web reader; a direct public read returned HTTP 403.
No alternate route was attempted after that response. The installed official
CLI documentation and saved receipts support the capability conclusion.

The official task permits diagnostic logs, but automatic logs are not native
artifact capture. Adding instrumentation would change the checked file and
require its own review. No instrumented submission, worker-cache access or
undocumented endpoint was attempted. An ordinary timing profile cannot show
whether the next page load moved before QK/PV, or whether a later instruction
waits on its result.

## Decision and next justified experiment

**Do not spend an official check in this bounded native-diagnostic phase.**
Preserve selected A2-11 and A2-13 as this candidate's exact parent. A timing-only
check could usefully falsify A2-15's net benefit; independent review agrees.
It would still leave the specific scheduling hypothesis unresolved. This
phase retains its predeclared native-evidence gate rather than treating a new
score as proof of overlap. No check was reserved or sent.

The target remains full decode B32: A2-13 is 316.411810 µs with within-run
SGLang/candidate paired ratio 1.216237805. A future >=10% normalized latency
reduction requires a ratio >=1.351375339. The equivalent 284.770629 µs is only
illustrative if the reference timing stayed identical across runs. Holding
other paths fixed, this models +1.48% aggregate, close to the minimum official
uncertainty interval; it is not a prediction or achieved result. All official
correctness cases must pass and any >5% regression in the eleven unchanged
controls requires investigation. Exact target page64/scales/ABA coverage is
also not supplied by the ordinary prerequisite set.

The next justified experiment is the **two-source native compile comparison**
already specified in NATIVE_DIAGNOSTIC_PLAN.md: frozen A2-13/A2-15, identical
full B32/L1 geometry and 8192-token chunks, generated producer/merge artifacts
with hashes and actual compiler/runtime versions, then compare next-page
load placement and its first dependent instruction. PTX differences alone
are not final machine scheduling evidence. Report missing SASS explicitly.
Register/spill metadata may qualify the result but cannot establish causality
alone. Only after the native result warrants it should correctness and paired
whole-call timing test performance.

This requires valid supported GPU access. Existing Modal authentication has
expired; this phase neither renews it nor claims a fresh account/billing/app
inspection. A narrow diagnostic program still needs implementation because
the old wrappers upload extra files and omit current compile parameters.
There are no new compiler artifacts or GPU samples. The reviewed source,
tests and sanitized blocker/audit are prepared for the existing private
export branch; the final task receipt records actual delivery SHA.

中文实验卡：官方检查仍可用，attention 今天剩 4 次。但现有接口只给正确性、
计时和 kernel 耗时，不能证明页号预取改变了原生排程。因此本阶段不花检查
额度追一个无法解释机制的新分数。下一步仍是相同配置下比较 A2-13/A2-15
原生指令和资源；当前缺少有效 Modal 认证，没有擅自续建。
