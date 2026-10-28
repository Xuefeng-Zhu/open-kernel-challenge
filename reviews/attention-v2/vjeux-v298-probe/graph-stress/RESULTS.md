# v298 serial graph correctness stress

**PASS:** 86 original-oracle checks, 12 retained independent captures across six kernel families, 264 main graph replays plus2 growth-test replays, and198 checks that another graph's output/new-cache entries remained unchanged. No performance measurement or official check/submission was performed. Candidate `0cc4a066…` and selected champion `3da22a16…` remain unchanged.

Coverage: same-stream16-replay bursts; A/B/A and cross-shape switching; fresh QKV; live lengths including0/1,64-token and2048-window boundaries; bijective page tables with consistent positions/cache_loc/length metadata; GPU scales1 and1/64. Full-L1 lengths/table remain fixed to preserve its provided scheduler metadata. Synthetic correctness fixtures apply a fixed0.25 factor to historicalV values for both scales, preventing FP8 overflow; K normalization is unchanged. These inputs were never used for scoring or speed claims.

All observed active FP32 numerator/denominator cells reset to zero. Nonzero parity epochs/completed counter banks are intentional and were not treated as stale data. No harness scratch clearing or candidate patching occurred.

The separately labelled unscored B33/DP/L8/page64/ctx64 case grew scratch capacity32768→33792 rows. It and an older retained graph both passed the original oracle. All three old Python scratch objects were no longer alive. No address recycling was demonstrated: this one passing old-graph replay does **not** establish universal allocation-lifetime safety. The source's device-only workspace cache still warrants caution for future growth/reuse.

The available TASK describes candidate/SGLang graphs taking turns and fresh-input replay; no simultaneous-stream requirement was found. Concurrent execution was therefore not tested and is not claimed safe. Current scored plans stay below the initial32768-row workspace capacity.

App `ap-w5uTMhngOSKIwYBZ2z8XG0` completed in73.99s worker wall time and was explicitly stopped, then independently observed stopped/tasks0. Existing free-credit reserve$2; actual charge not queried; paid cap$0 unchanged. No new credentials, security settings, billing query or official action.

Raw results, source hashes and receipt are retained here. Worker/test files are frozen snapshots from the isolated stress directory; `helpers.py` is the unchanged parent probe worker, hash7fa92eba152f715c995b312b475cf54195aad5ca5a94ddc825727c3bfcd2262f.
