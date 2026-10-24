# 000 — preserve the correctness-first baseline

This is the initial repository snapshot, prepared before the first remote experiment.
Earlier local work is consolidated here; no historical commit sequence is invented.

- Hypothesis: the official PyTorch dense projection is a sound correctness baseline.
- Implementation: BF16 `x @ W[:n].T`; FP32 activation/product for the FP32 router;
  original weights retained; no caching or data-dependent host synchronization.
- `submission.py` SHA256:
  `78bd5697391a335d5ba771035624bde12e36ecfa5f3058f3db08d00fbe08ae3e`.
- Environment: macOS arm64 / Apple M2, Python 3.12.13, PyTorch 2.14.1, four CPU threads
  for the full-size correctness suite. No CUDA is available on this host.
- Commands:

```sh
.venv/bin/python check.py --device cpu --suite all --output results/cpu-correctness.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python static_check.py
.venv/bin/python -m compileall -q submission.py harness.py check.py benchmark.py static_check.py tests
.venv/bin/python benchmark.py --output results/gpu-blocked.json
```

Observed before this snapshot: 39/39 full-size CPU cases pass the unmodified official
FP64 reference and row/probe checks; inputs/weights remain unchanged. Five focused
tests pass, including deliberately wrong outputs that the oracle must reject.
Static checks and compilation pass. The local benchmark **exits 2** at its no-CUDA
guard; that prerequisite failure produces no GPU timing.

Evidence is in `results/cpu-correctness.json`, `results/unit-tests.txt`,
`results/verification.json` and `results/gpu-blocked.json`. GPU run ID: **none**.
No performance conclusion or genuine optimized candidate exists in this snapshot.
