# A2-14 complete: reject performance regression

Job `6d6019056cbc4f81` is terminal DONE; do not resubmit. Official H200 passed
16 prerequisites + 12 timed shapes. Score 1.1687086418×, CI
[1.1511780122, 1.1862392714]. Target full decode B32 latency 316.412→422.966 µs,
normalized +33.9263%, versus a 20% reduction goal. Aggregate −3.4915% vs A2-13;
no unchanged control regressed >5%. Independent analysis confirms rejection.

Frozen source: reviews/attention-v2/mixed_fa3_deferred_denom.py, commit
2092957b1c8312b1fc90b43f64c3ed5eed8f4658, SHA256
bba54e6e5cd779074ad0c372ca60556048085251edd6317463312e4a74cee43d.
Full receipt, all 12 timings, six weighted loads, numerical coverage and
commands are in reviews/attention-v2/A2-14-RESULT.md and associated JSON/CSV.
Source bytes have not changed after dispatch. Exact-source ABA/concurrency,
untouched rows and B32 page64/non-unit-scale stress remain unverified. No
current-candidate register/spill metadata; do not invent a hardware cause.

Selected attention remains A2-11, mixed_fa3_decode_parallel_merge.py,
job cfa8a8670e394f80, 1.1876787561×. Promising A2-13 FP8 QK remains separately
preserved, job c2fb7b23433f4888, 1.2109908035×. Projection P039 is unchanged
in its separate worktree, job d5636ac8d1404abd, 1.1732389990×.

The one exact A2-14 check permission is consumed; no new candidate upload,
Modal, credentials, paid resource or ranked entry is authorized by it. The
reviewed A2-14 source/tests/related analysis/sanitized evidence private push
completed at 07:55:09 UTC, verified first-delivery SHA
16a8c48296884950e308d8217918444ccefedb99 on attention-fa3-sanitized.
A confirmation-only commit may advance that branch; the final task/ignored
receipt records its exact latest SHA. No additional candidate transfer is granted.
Push verification belongs to the task and ignored local push receipt; inspect
actual remote state after any interruption, never blindly retry. Never push
original attention-fa3-local history. No unrelated projection push.

Actual platform snapshot 2026-10-25 07:50:53 UTC: attention16/20 used4left,
projection3/20 used17left, scored0/10 each. Reset October26 00:00 UTC.
No active jobs among37returned; zero uncertain requests. These counts are a
timestamped observation, not a perpetual grant. Historical Modal apps were
stopped; no Modal action here. Conservative cost remains $1.787819802.
