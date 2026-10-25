# A2-15 local diagnosis and prototype complete; no external operation pending

Read CURRENT_AUTHORIZATION.md. No new OKC/GPU/Modal/push/credential operation
was attempted or reserved in this phase. Native compilation and execution are
not available in this Mac environment (no Triton installation). Do not upload
or launch without a separate exact approval.

Candidate: reviews/attention-v2/mixed_fa3_page_lookahead.py, SHA256
fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c.
Parent is preserved A2-13, not rejected A2-14. Only full B32/L1 prefetches next
page IDs before current QK/PV, consumes them after PV and carries int64 rows.
Math, scalar denominator, K/V payload loads, merge/cache ownership, geometry and
all other paths remain the parent. Seven AST/address and three diagnosis tests
pass. Independent review found no concrete semantic defect; native scheduling,
resources, correctness and performance remain unverified.

A2-14 diagnosis: denominator state16→1024 FP32 values and full-chunk logical
multiplies2048→131072, with max reductions unchanged. These are source counts,
not measured instructions/registers. Producer profile311.1→422.0us; no saved
candidate-specific compiler artifacts were found. Do not claim proven spilling.

See reviews/attention-v2/a215/README.md, diagnosis.json, preflight.json and
NATIVE_DIAGNOSTIC_PLAN.md for evidence, commands and the falsification gate.
The bounded local phase has one candidate; do not start a broader rewrite or
parameter sweep. Selected A2-11 and projection remain unchanged. Latest actual
verified private remote tip from the prior phase:
73d9d1acfa21722415bb0957f3ec9197c75c5855 (attention-fa3-sanitized).
New A2-15 work is local and unpushed.

The prior completed A2-14 checkpoint follows; its check must never be repeated.

---

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
