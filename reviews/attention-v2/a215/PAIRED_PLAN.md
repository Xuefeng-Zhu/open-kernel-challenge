# Completed focused follow-through

App ap-DjbFQfnZXQjpr1Zgt3rtFK completed and is stopped/tasks0. See
[PAIRED_RESULT.md](PAIRED_RESULT.md). The frozen prelaunch plan follows;
do not relaunch its existing ledger.

---

# A2-15 focused execution follow-through

Parent: frozen A2-13 `mixed_fa3_fp8_qk.py`; candidate: frozen A2-15
`mixed_fa3_page_lookahead.py`. Their hashes are unchanged from NATIVE_RESULT.md.
Selected A2-11 stays selected. No official check is part of this run.

The native gate found a real pipeline change: two K/V shared buffers, later
page-ID loads interleaved with PV, 7,936 additional shared-memory bytes and
unchanged register count. It did not prove a speedup. This test asks whether
that change preserves outputs/cache/graph behavior and reduces target latency.

## Coverage fixed before launch

1. B32/L1/full, TP8, context 80,000, agentic lengths, coherent values, page 1,
   scale 1. This is the changed scored shape family, with two fresh seeds.
2. B32/L1/full, TP8, page 64, scale 1/64, scattered pages, lengths
   0/1/63/64/65/8,191/8,192/8,193 repeated and reversed between A and B.
   A zero-length request writes cache row zero. Unused page-table tails are
   -1. The official fixture's positions, cache locations, lengths and FA3
   scheduler metadata are updated consistently. Zero historical length is
   an additional robustness case, not a scored fixture.

Each source captures one graph per case. It replays A→B→A at unchanged tensor
addresses. A/B outputs and new cache entries use the unmodified official
reference/checks. Returning A must exactly reproduce its previously validated
output/new-cache bytes. Every replay checks immutable inputs and all untouched
cache rows. All required checks must pass before timing begins. This does not
test overlapping calls, all 16 prerequisites, all 12 scored shapes or controls.

Target timing uses four predefined warmup pairs and 12 measured pairs, alternating
runner order. Adjacent opposite-order pairs use the same input version; versions
alternate between independent A/B fixtures. Every timing replay restores the
inputs/cache outside the event interval and checks the result against its same-
source, same-input oracle-validated bytes. A 256 MiB read precedes each replay
to evict L2. Retain all samples, including slow ones. Raw whole-graph timing and
independently empty-graph-subtracted values are separate; neither is an official
score. Six balanced pair ratios are limited evidence, not a confidence interval.

The predeclared useful target remains at least 10% latency reduction. Passing
this small test would justify full official validation, not automatic selection
or a claim about six-load overall performance. A failure, partial coverage or
timeout yields no improvement claim. No extra launch follows automatically.

## Feasibility and resource ceiling

Historical A07 completed two large full-attention cases, two implementations and
A→B→A in 26.09 s worker / 35 s app. Its B32/L8 page-64 non-unit-scale case was
larger than this L1 probe. The current native run took 18.79 s worker / 30 s app.
These observations support attempting a narrow run; they do not guarantee time.

Conservative prior accounting is $1.849496218, leaving $0.150503782 under $2.
At the existing H200 + 2 CPU + 8 GiB rate, the 90-second model reserves
`95 × $0.00130496 × 1.1 + $0.01 = $0.146368320`. A 93-second model would exceed
the remaining ceiling. Use one container, no idle capacity, no retries, a
52-second internal deadline, 55-second worker kill, 60-second function timeout,
15-second startup timeout and 75-second app watchdog plus 15 seconds cleanup.
Stop immediately on return; independently verify stopped/tasks0. Keep the
reservation until actual shutdown/cost reconciliation. Never lower prior cost.

The existing Mac browser showed $28.77 credits, paid cap $0 and paid charges $0.
Read-only existing-auth CLI calls showed no active apps and $0 billed charges.
Those checks must still be fresh at launch. No credential creation, payment or
settings change is implemented. The repeated-source no-duplicate ledger is
`paired-modal-run.json`.

Allowlist: two candidates, `paired_worker.py`, existing `modal_worker.py` helpers,
official `reference.py` and `checks.py`, LICENSE and THIRD_PARTY_NOTICES.md.
The controller is serialized without repository automount; no credentials,
private logs or unrelated source are mounted. Results are bounded to 100 KB
plus a 4 KB failure tail. No native instruction dumps are repeated.

Local commands:

```sh
.venv/bin/python -B reviews/attention-v2/a215/test_paired_worker.py
.venv/bin/python -B reviews/attention-v2/a215/paired_modal.py self-test
```

Both must pass and final source must be committed before the one launch.

## 中文实验卡

**假设：** A2-15 的页地址读取和 K/V 双缓冲能降低 full decode B32 延迟。
**先验门槛：** 两个版本都先通过普通与边界输入、缓存写入和同地址 A→B→A。
**测量：** 只测这个目标路径，交替顺序，保留全部样本；不换算官方总分。
**预算：** 预留 $0.14637 免费额度，实际付费 $0；超时即停止并保留部分证据。
