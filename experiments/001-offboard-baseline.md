# 001 — one off-board H200 check of the unchanged baseline

Status: **done; correctness passes, performance slower than SGLang**.

- GPU prerequisites: **9/9 passed**. Timed scored-shape checks: **30/30 passed**.
- Static/server checks: ok, no flagged or rejected items; evaluator return code 0.
- Official speedup: **0.9076063757813903×**; 95% interval
  **[0.8939922801446694, 0.921220471418111]**.
- Judge-normalized displayed time: **5322.571688460203 µs**, using its 4830.8 µs
  reference. This is not the directly measured latency of a single load.
- All 30 official per-shape records are preserved, including 60 replay measurements
  per shape, paired ratios and replay-overhead metadata. Aggregating their weighted
  load times reproduces the official speedup to floating-point precision.
- Host reported NVIDIA H200, driver **580.159.03**, SM clock **1980 MHz**, memory clock
  **3201 MHz**, power limit **700 W**. Exact before/after fields and evaluator/image
  digests are in the sanitized JSON; private server node identifiers were omitted.
- Queue wait: **806.39 s**; run elapsed time: **49.07 s**. These are job timestamps,
  not kernel timings. No new run was sent while waiting.


- Hypothesis: the published PyTorch starter passes the real H200 correctness/graph
  checks and establishes a measured baseline against SGLang.
- Change: **none** to `submission.py`; this is validation, not an optimized candidate.
- Source commit: `fde8a042944c742c20b76f27fcc030d767fd0058`.
- Source SHA256: `78bd5697391a335d5ba771035624bde12e36ecfa5f3058f3db08d00fbe08ae3e`.
- Server content/deduplication hash (not assumed to be a raw-file SHA256):
  `b120b039d432a4b4f5f349900811b310ebf630fa601dcafe889f57bbc177abab`.
- CLI: official checksum-verified `okc 0.7.1` on macOS arm64; existing authentication
  and rules acceptance were verified without exposing credentials.
- Target: official hosted H200 SXM judge and its supplied SGLang environment. Runtime
  package/hardware details beyond that have not yet been returned.
- Exact command issued once:

```sh
./bin/okc check submission.py --no-wait --json
```

Accepted job ID: **`e5a5b2d20da54187`**, mode **check**, off-board/unscored. The existing
monitor uses:

```sh
./bin/okc status e5a5b2d20da54187 --wait --json
```

Read-only status/quota snapshots do not create new jobs. Selected-task checks moved
from 0 used / 20 left to **1 used / 19 left**. Scored usage remains **0 used / 10 left**.
No paid resource was created, no charge was requested, and no extra check is authorized.

After the user raised a once-per-day concern, current official/account metadata was
rechecked read-only. Selected kernel checks remain 20/day and scored runs 10/day;
the **serving scored cap is 1/day**, with 3 serving checks/day. Serving is currently
closed. Rules version remains `2026-10-22.1`. The guide also documents per-minute
submission rate limits and final review/reruns, not a separate final-upload allowance.
No cancellation, duplicate or extra POST was made during this clarification.

The committed baseline was also independently extracted with `git archive` and
verified against the pinned local environment: **39/39 full-size CPU cases, 5/5 focused
tests, source hashes and static checks pass**. These are CPU results only.

Evidence: `results/h200-check.json` and `results/h200-per-shape.csv`.
Configuration is the unchanged `submission.py` in the source commit above; no manual
judge configuration override was used. The result commit is the Git commit containing
this completed record; obtain it with `git log -1 --format=%H -- experiments/001-offboard-baseline.md`.

Measured next-step evidence: BF16 projection ratios mostly remain near 1; the router
at m=16/32/128/256 takes **25.90/19.48/28.57/45.53 µs**, against SGLang's
**14.69/14.94/16.26/17.34 µs**. This supports testing the allowed hi+lo router path
as the first future one-variable experiment. It has not been implemented or run.
No extra check, scored submission or paid compute is authorized by this result.
