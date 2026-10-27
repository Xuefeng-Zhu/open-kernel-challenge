Independent review confirms **correctness PASS, performance NO-GO** for nonranking check `3e0a7ef51d4849c6`. The working selected candidate remains `3da22a…`, officially measured at **1.2336330589×**. This review used saved files only; no runner, GPU, authentication or Git action was taken.

The terminal raw response is `done`, `duplicate_of=null`, `passed=true`, return code 0, error null, and clean checks/flags. All 16 original prerequisites (indices 0–15) and 12 benchmark rows pass; 12 profile records are present. Both new and selected profile summaries agree with their raw responses; all 12 benchmark `median_ns`, `baseline_ns`, `paired_ratio` and status values agree exactly. The request receipt binds this job to committed candidate bytes `96e682…` at commit `04447c1cc8f64fdb489b74e1e942a64e0fae3cbb`. The service content hash is a separate identifier, not the raw source SHA256.

Using **C = candidate latency** and **S = its own same-run SGLang reference**, the descriptive historical latency ratio is `(Cnew/Snew)/(Cold/Sold)`. Independently recomputing this ratio gives:

| Changed path | Selected C, µs | New C, µs | Normalized latency change |
|---|---:|---:|---:|
| Decode B1 sliding, S16 | 11.4771 | 13.7898 | +21.0917% |
| Decode B16 sliding, S9 | 13.9211 | 17.0724 | +20.7343% |
| Decode B32 sliding, S5 | 20.1036 | 25.9469 | +29.5344% |
| Decode B32 full, S11 | 304.0319 | 546.0326 | +79.5972% |

The exact six-load geometric mean of `weighted S / weighted C`, with 46 sliding and 16 full layers per load, reconstructs **1.0995474217354162×**, matching the official score. Its reported interval is **[1.0830542104, 1.1160406331]**. The point score is 10.8692% below selected. Applying only the four normalized target ratios to selected times, holding its references and eight controls fixed, gives modeled score **1.1006272179140026×**, a **−10.7816372%** change. Thus both declared performance gates fail: modeled gain ≥3%, and no target regression >3%. Maximum absolute drift among the eight controls is **1.9017961%**, within the declared 3% control gate. `analysis.json` classification `performance_rejected` is correct.

The entire evaluator code/image/files dictionary is identical. Reported stable GPU fields match: H200, driver `580.159.03`, SM/max-SM clocks 1980 MHz, memory clock 3201 MHz and 700 W power limit, before and after each run. Nodes, temperatures and instantaneous power draw differ. These observations support an engineering rejection but do not turn two runs into paired candidate-versus-selected measurements. The official pairing is candidate versus its own reference; fresh fixture seeds also differ. Marginal score intervals do not provide a cross-run significance test.

Saved **instrumented** activity durations localize the observed difference to producer activity:

| Path | Producer selected → new, µs | Merge selected → new, µs |
|---|---:|---:|
| B1 sliding | 7.8 → 9.2 | 1.7 → 1.7 |
| B16 sliding | 9.0 → 11.7 | 1.8 → 1.8 |
| B32 sliding | 14.8 → 20.3 | 2.2 → 1.9 |
| B32 full | 297.2 → 528.6 | 2.7 → 2.8 |

These rounded profile values are separate from benchmark latency. Names `_m64n8_fused`/`_fused` associate activity with producer dispatch, but establish neither exact binary identity nor a counter-level bottleneck/cause. S9/S5 now have official original-oracle and timing evidence despite lacking completed offline compilation. Their exact executed opcode geometry, registers, shared allocation and spill counts remain unknown. The frozen candidate emits no runtime metadata marker; UNKNOWN is expected.

Verified SHA256 bindings:

| Artifact | SHA256 |
|---|---|
| New raw response | `e61d99b67ebbf917cfc4e5a5ad29848e37b92be60a08e9825728e0bd428b81c0` |
| New profile | `8fb24b19b2bcc0517baa980b974692e9e98b7862b316e5bbb7db842626bd36f1` |
| Request receipt | `df9206618a8bfdad2d17b80dfdc0e2f2132b969a8d14104ca2efc4b971737e3d` |
| Analysis | `e6f57d79a35f627f0e081fe45c7e471b40f409b166960ebe9a5e4f295fffcf71` |
| Candidate | `96e682db54aae97fd95c4e9b47b263f2322e40bfd11988b504f487dfd5c9c970` |
| Kernel source | `eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480` |
| Selected raw response | `a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb` |
| Selected profile | `3a9f9f841665ef0c678001aa62396b44557a375ce20d5e830548d257ff6f1300` |
| Analyzer | `b50f7043890f97039ad20442f6db2f1615c6df19b50c2e755f1dcea0a0e60eb3` |
| Analyzer tests | `05c7755a4776464cd82fec53fc90b788eecfde96f8f6deef4ae664960fcf8a3a` |
