# Official non-ranked check: v298

Job `9292ad030b2a4b3d`: **PASS, 2.618804×**, 95% interval [2.579522, 2.658086]. All16 prerequisites passed, with12 benchmarks and12 profiles; no flags, rejections or errors. Exactly one check was sent; no ranked entry or selection change.

Source SHA256 `0cc4a0668acf7553c0d2c6d5c3a2e7d5c85374a70ad774463e9d2fd9e5d0f883` is the pristine licensed upstream v298 already validated locally.

Finished 2026-10-28T02:49:06.265150+00:00. Free check quota:19 before,18 after. Scored quota unchanged. No Modal app was launched in this stage.

Historical champion control: check `5090b83f1d63407b`, 1.233633×. Evaluator code, all evaluator file hashes and judge image match exactly; node/cluster and all12 fixture seeds differ. The table is a descriptive cross-run comparison; the prior same-H200 paired native test supplies separate evidence.

| Shape | Historical champion µs | v298 official µs | Reference-normalized cut |
|---|---:|---:|---:|
| decode 1 sliding | 11.477 | 4.913 | 57.13% |
| decode 1 full | 27.031 | 11.035 | 59.65% |
| decode 16 sliding | 13.921 | 8.372 | 41.49% |
| decode 16 full | 194.272 | 86.940 | 55.32% |
| decode 32 sliding | 20.104 | 9.998 | 50.72% |
| decode 32 full | 304.032 | 165.843 | 45.56% |
| verify 1 sliding | 16.988 | 6.288 | 61.98% |
| verify 1 full | 29.569 | 14.334 | 52.86% |
| verify 16 sliding | 21.740 | 11.574 | 48.35% |
| verify 16 full | 197.602 | 100.770 | 49.33% |
| verify 32 sliding | 27.468 | 14.677 | 46.94% |
| verify 32 full | 380.381 | 184.317 | 51.77% |

Official profile records show one native `attn_kernel` call for each scored shape. Activity profiles are not NCU counters or a causal decomposition of the improvement. Shared scratch concurrency remains unproven; the selected champion source and rollback baseline remain unchanged.

Raw receipt SHA256 `dc42f36a9a68b28a6c53b7b2c350061f1c86d4d6f4d8b43dc67a1128914e7b07`; complete timings, tests and environment identity are retained in official-result.json and analysis.json.
