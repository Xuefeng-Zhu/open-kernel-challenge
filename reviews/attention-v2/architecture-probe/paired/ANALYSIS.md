# BF16 full-verify follow-through: complete representative run

Frozen execution commit `7c544c918996739863f126771f4ebe8fcdc7f122`. Result SHA256 `b55b87101004742d43d33272a632250d094a2345f1a3e37fce64589f76d931e9`.
Published launch record SHA256 `640de3af133ef350ded5c812b064fbb03f51b88513f67dfaca4a60670a739cda`; default analysis reads `launch-receipt.json` and requires no ignored local record.
Recorded status `complete`; official-oracle calls 18, passed replays 18/18, exact returning-A checks 6/6.

The FA3 arm is unchanged exact A2-13 `candidate.kernel_fn`; the experimental arm uses a separate BF16-QK helper. Formal routing and selected A2-11 remain unchanged.

Both families completed all planned samples after all three cases passed the unchanged official oracle. All measured and warmup replays checked exact same-arm output/new-KV bytes, stable addresses, immutable inputs and untouched cache rows.

| Family | FA3 raw median us | BF16 custom raw median us | Median FA3/custom pair ratio | Custom latency change |
|---|---:|---:|---:|---:|
| full_verify16_scored_family | 211.584002 | 342.480004 | 0.618493336 | +61.683% |
| full_verify32_scored_family | 396.080002 | 588.400006 | 0.671632037 | +48.891% |

Positive latency change means slower than FA3. Each ratio summarizes six adjacent reverse-order pairs; these are within-session observations from two input versions, without confidence intervals or an official score.

full_verify16_scored_family reverse-order pair ratios: 0.615323911, 0.618778583, 0.621208884, 0.617155241, 0.619048641, 0.618208089.
full_verify32_scored_family reverse-order pair ratios: 0.673425107, 0.667056140, 0.672140291, 0.668209680, 0.676553236, 0.671123782.

**Decision: reject this complete custom BF16 route as a performance candidate.** Both families are slower than unchanged FA3; an official check is not justified by these results. WGMMA feasibility passed compilation, but that is insufficient to overcome the complete-path deficit. This run did not time the old custom FP8 route, so it cannot isolate BF16 widening's own runtime effect.

The optional profiler is outside paired timing.

The actual B32/L8/D64/page1 FA3 event name contains the complete scheduler type:

`flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true>`
`flash::VarlenDynamicPersistentTileScheduler<64, 128, 128, 128, true, true, true, true, true, true>`

Thus the existing B32 verify specialization already names a dynamic persistent scheduler. Exact build/source identity, boolean meanings, B16 scheduling and native instructions remain unverified. See `FA3_PROFILE.md`; no scheduler novelty is claimed.

Scope: page64/scale1/64 boundary lengths plus full B16/B32 L8 80K page1 coherent families; same-address A→B→A replay. This is not all official shapes, the full prerequisite suite, concurrent-call testing or independent-session confirmation. No official check or selected-candidate promotion follows from this report.

Exact payload hashes, all recomputed values and failures are preserved in `analysis.json`; deliverable hashes are in `analysis-manifest.json`.
