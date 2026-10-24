# Current status

- Baseline source commit: `fde8a042944c742c20b76f27fcc030d767fd0058`.
- Baseline source SHA256: `78bd5697391a335d5ba771035624bde12e36ecfa5f3058f3db08d00fbe08ae3e`.
- CPU: 39/39 full-size cases and 5/5 focused tests; reproduced from a clean Git archive.
- Official H200 check **e5a5b2d20da54187** is **done**: 9/9 prerequisite and 30/30 scored
  shape checks passed; no rejected/flagged checks. Mode is check, off-board/unscored.
- Official speedup **0.9076063758×**, 95% interval **[0.8939922801, 0.9212204714]**.
  Starter is slower than SGLang. No optimized candidate exists yet.
- One check used; 19/20 remain in the service quota. Scored usage is 0/10. The user's
  single-check authorization is exhausted. No further check or scored entry is authorized.
- Serving's separate cap is 1 scored submission/day and 3 checks/day; track is closed.
- No paid resource, new credential or additional agreement was created by the agent.
- Private repository: https://github.com/Xuefeng-Zhu/open-kernel-challenge.
- See `experiments/001-offboard-baseline.md`, `results/h200-check.json` and
  `results/h200-per-shape.csv` for source/configuration/job/per-shape evidence.
- A separate user attention draft is under local review only; it is ignored by Git,
  has not replaced this baseline and has not been uploaded or run on the judge.
