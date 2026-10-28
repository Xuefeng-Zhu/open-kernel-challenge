# Original user v4: official check

Unchanged original `submission_v4.py`, SHA256 `dcfbca2f8e1135762b08c77df931a7f53948f69746e0314797ee3dbef4147e82`.

Job `7e9aa354da204bdf` was admitted before the user restricted future official checks to Frank. Server admission was 2026-10-28 00:30:18.597119 UTC, 12.402881 seconds before the 00:30:31 UTC restriction. Completion was 00:41:29.105613 UTC. It completed in `mode=check`; no further check or ranked submission was launched.

**PASS:16 prerequisites + 12 benchmark cases. Score 1.25282976695078×, 95% interval [1.2340373204465183,1.2716222134550415].** Historical champion score 1.2336330588627955×; point ratio +1.5561%, with overlapping intervals.

|Full path|Champion historical µs|v4 official µs|Reference-normalized change|
|---|---:|---:|---:|
|B16/L1|194.272|164.533|-15.496%|
|B32/L1|304.032|317.569|+4.335%|
|B16/L8|197.602|198.036|+0.126%|
|B32/L8|380.381|382.536|+0.055%|

Only full decode B16/B32 dispatch changed. All ten other routes retain the champion source. Their maximum absolute historical reference-normalized drift was 4.104%; sliding results are retained separately in `summary.json`.

The evaluator matches the saved champion. Each official result is paired against its own SGLang reference; this comparison to the historical champion is **not same-session A/B**. Full source and raw receipt remain preserved. The sanitized receipt omits service routing fields and local file paths.

Champion SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` remains selected. No automatic promotion. The next measurement is same-GPU graph timing and separate producer profiling of original v4 versus the actual champion; no tuning has been performed.
