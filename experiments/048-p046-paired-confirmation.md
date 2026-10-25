# P048 — direct P039/P046 confirmation

Status: completed and selected P046 as working candidate after direct paired confirmation.
No candidate code changed and no official check was consumed. P039 is preserved.

实验卡：P046 的三个 pooler 小批量路径已在两次官方作业中显示收益，但未改动
的 O/unpooler M1 也发生漂移。本次让 P039/P046 在同一 H200、同一输入地址上
交替执行，验证收益能否保留，以及控制路径是否还会变慢。不重新调整参数。

Evidence: P046 job `41ba778763a5433b` passed 9 prerequisites and 30 scored shapes.
The three target normalized latency cuts were 11.52%, 8.32%, 8.72%; the original
three-target modeled whole-score change was +1.5533%. Cross-job raw paired
control movement was +3.45% (O M1) and +3.76% (unpooler M1), while calibrated
movements were larger. Source ASTs of those paths are unchanged; this does not
exclude allocator/library/environment effects. Direct paired evidence is needed.

## Frozen inputs and protocol

- Parent `candidates/projection_router_m1_fp32_gemv.py`, SHA256
  `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`.
- Candidate `candidates/projection_pooler_column_batch2_hybrid.py`, SHA256
  `001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e`.
- Six cases: pooler M8/16/32 targets; O M1, unpooler M1 and restored pooler M128 controls.
- Existing image `im-cPTbGpNiX3RarKResPgX7h`, H200 SM90 only. This is a native
  diagnostic environment, not proof of exact official judge overlay equivalence.
- Both graphs receive the same input address, same original weights and distinct
  prepared states/outputs. Original and derived weights are checked immutable.
- Before and after timing: normal A, outlier B, A again, explicit sparse signed
  power-of-two columns, and zero. Original FP64 oracle and tolerances retained.
- Six excluded warmup rounds, then 24 fresh-input rounds (12 adjacent pairs with
  runner order reversed). All raw observations retained; no post-hoc deletion.
- Each timing follows a 256 MiB eviction-buffer read. Reusable CUDA events are
  created, recorded, synchronized and queried before all samples. Graph capture,
  host setup, oracle work and warmup are outside the measured replay.
- Ratio is P039 raw microseconds / P046 raw microseconds, combined geometrically
  in each reversed pair then summarized by median. Greater than one favors P046.
- No fixed-cost calibration/subtraction and no official score are claimed.

Predeclared diagnostic gates: at least 3% geometric target raw-latency reduction,
no target regression; controls moving more than 2% in either direction require
investigation. Twelve pairs and event quantization limit small-change conclusions.
Use the official full-suite pass together with direct structural/control evidence;
this diagnostic cannot alone establish all-shape correctness or final score.

## Bounded execution

Existing protected login only; refreshed browser credits/cap and read-only CLI
app/billing checks precede launch. Allowlist is exactly 12 files: the two sources,
worker, benchmark/harness, three official contract files and four notices/licenses.
No whole repository, personal data, credentials or local notes are uploaded.
One ephemeral H200 container, CPU 2, RAM 8 GiB, no retries/detach/deploy/schedules.
Worker internal deadline 120s, subprocess hard timeout 180s, function 195s,
startup 45s, app watchdog 240s. Reserve: 270s at current H200+CPU/RAM rate,
plus 10% and $0.02; approximately $0.408 free credits, actual paid cap $0.
Explicit stop and stopped/tasks0 verification are required even on failure.

Commands (activate the existing project Python environment):

```bash
python -B -m unittest discover -s tests -p 'test_projection_p046_confirmation*.py'
python -B projection_p046_confirmation_modal.py self-test --control-root "$OKC_CONTROL_ROOT"
python -B projection_p046_confirmation_modal.py launch --control-root "$OKC_CONTROL_ROOT" --frozen-head <reviewed-40-character-commit>
```

The coordinator owns the serialized GPU slot. A durable exclusive record prevents
relaunch after an uncertain outcome. No NCU counters or new packages are involved.
Sequential A/B/A is tested; concurrency and all 30 shapes remain outside this
six-case diagnostic and must not be represented as covered here.

## Actual H200 result and selection

App `ap-lpXZhacMWtQNlM5oksAWRX` completed and stopped with zero tasks; all
workspace apps were also stopped. Frozen harness commit `d72380a95df959714f4576a55aa6ea75fde51477`.
Worker execution 11.418 s; supervised child 12.452 s; client 21.556 s; app lifetime 23 s.
All 480 original-oracle comparisons passed: 120 before/after probes, 72 warmups,
288 measured replays. Immutable original/prepared weights and same-address
A → outlier B → A checks passed. No simultaneous-call test was run.

| Path | P039 raw median µs | P046 raw median µs | Paired raw latency cut | Improved pairs |
|---|---:|---:|---:|---:|
| pooler M8 | 24.432 | 21.888 | 10.745% | 11/12 |
| pooler M16 | 25.392 | 22.544 | 10.873% | 11/12 |
| pooler M32 | 25.328 | 22.848 | 9.879% | 11/12 |
| o M1 | 11.408 | 11.360 | 0.481% | 7/12 |
| unpooler M1 | 8.592 | 8.384 | 0.676% | 6/12 |
| pooler M128 | 28.080 | 28.224 | 0.179% | 6/12 |

The paired statistic is computed from raw samples, not the ratio of table medians.
Target geometric latency reduction is 10.500%; all target gates pass. Control
median movements are -0.481%, -0.676%, -0.179%, below the 2% diagnostic flag.
Each target improves in 11/12 pairs. Keep all long tails: candidate pooler M16
maximum 82.144 µs; parent pooler M128 maximum 181.856 µs. These tails and twelve pairs
limit claims about small differences or latency predictability. No samples were
removed, no fixed overhead subtracted, and no new official score computed.

Select P046 as the working candidate. This rests on the official full correctness
pass, two official observations of the small-M mechanism, direct paired target
reproduction and structurally unchanged controls without reproduced median loss.
It does not prove the official whole-score difference statistically significant
or explain the previous cross-job drift's hardware cause. Preserve P039 intact.
The selected official score remains 1.1841399408925375×, CI
[1.1663778417791495,1.2019020400059255], job 41ba778763a5433b.

Local 29 tests passed in 0.044 s before launch; no candidate code changed.
H200 SM90, Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0, Python 3.12.3.
The Modal image is not the exact official judge overlay. Source hashes and all
raw observations are in `results/remote-P048-confirmation/timing.json`.

Cost reserve $0.407634500099 remains recorded; the 30 s conservative exposure
model is approximately $0.0631, not an invoice. Billing observations may lag;
read the ledger's latest workspace delta, not a claimed settled per-app charge.
Latest delayed workspace gross delta was $0.02179635, not a settled per-app invoice.
Actual paid usage remains $0. No official request was created; quota snapshot
remains 6/20 checks used, 14 left, reset 00:00 UTC. No credential change occurred.
