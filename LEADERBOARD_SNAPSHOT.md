# Informational leaderboard comparison — 2026-10-24 16:06:00 UTC

Official problem board: https://openkernelchallenge.dev/api/v1/leaderboard/muse_bf16_proj
Official metric/mode guide: https://openkernelchallenge.dev/llms.txt

The read-only official CLI response reported **749 ranked participants**, 749 distinct
participant rows; rows aggregate 3012 attempts, so this is not every historical run.
The guide says withdrawing a run restores the participant's next-best run, consistent
with one best row per participant. SGLang's unranked reference row is excluded.
Metric: speedup over SGLang, higher is better, same muse_bf16_proj workload/metric as
an off-board check. Checks are expressly the same run kept off the board.

R05's exact point score 1.0509508061× (95% interval[1.0351865,1.0667151]) has **719
entries above it and 0 exact/display ties**, so hypothetical insertion is about
**720th** (among750 after adding a new participant). Adjacent existing rows:
719th1.0517×, 720th1.0507×. Topthree:1.5608×,1.5307×,1.5186×.
The response's `you` was null: no actual ranked entry was made. Counts use the more
precise score_us values (4830.8/value), not only rounded speedup labels.
This is a point-score snapshot, not a guaranteed rerun rank, overall rank, serving
standing, prize qualification or final leaderboard placement. The board is provisional.
No scored/public/final submission was created by this informational read.
