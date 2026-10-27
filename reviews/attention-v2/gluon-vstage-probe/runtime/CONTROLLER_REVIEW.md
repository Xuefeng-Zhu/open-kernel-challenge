# Independent final runtime-controller integration review

**PASS; no concrete blocker found.** The old benefit.admitted schema mismatch
was corrected before a launch. This local review verifies the actual worker/
controller contract, not a copied old completion fixture. It does not establish
GPU correctness or benefit, and does not itself launch or admit a job.

All11 final controller CPU tests passed independently. The earlier12 worker
tests/default preflight and semantic review remain passing with unchanged
workerbc93f1. No vendor/GPU imports, services, network, credentials or Git were
used. Only this report was written; no source/card was edited.

## Actual helper integration and counts

The review called the **real worker.benefit_gate** on12 numeric sample rows
generated with its actual order/fixture helper. All four mechanism/actual-FA3
true/false combinations passed unchanged through JSON, controller transport
hash validation and validate_success. Complete negative outcomes remain valid
experiments, not execution failures. Legacy admitted-only objects, integer
instead of boolean fields, inconsistent conjunctions and official-admitting
objects were rejected.

The actual fields are mechanism_passed, actual_fa3_passed and
prospective_followup_supported; the third must equal the conjunction of the
first two. Both nested benefit admission flags remain false. Source maps,
actual worker hash, correctness/timing completion and all counts are checked.
The ledger records those three fields and actual timing presence, without
creating a performance, replacement or official admission.

For an additional independent count check, the review executed the worker's
actual count-building AST and timing-counter increment statements on a CPU
contract event ledger. This reproduced all12 expected fields:
30 original oracles,9 return-A,36 cache comparisons,2 submission oracles,
1 setup oracle,4 isolation checks/2 extra replays,9 prototype ABA checks,
18 cross-arm new-KV ABA checks,48 checked timing replays,16 prototype-pair and
32 cross-arm new-KV timing checks. This is schema/count integration, not a
claim those GPU checks have run.

## Admission and closure guards

The source-bound native receipt must explicitly pass mechanism, resource,
lifetime and matched work; current raw compile SHA258c907f…41240 and hash-bound
source/native review files are rechecked before reservation. A compile/resource
pass alone cannot create this receipt. Parent review of its actual verdict and
the real frozen-head preflight remain separate requirements.

Nine allowlisted payload files and the local controller are committed-hash
checked. Credentials and receipts are absent from the closure-free remote
payload. Fresh source-bound serialized slot and billing receipts, live paid0/
free-credit/rate checks, existing protected credentials and no active apps are
required. The worker receives --execute/160s. One H200/2CPU/8GiB, zero retries,
170/185/20/210 child/function/startup/watchdog and$.58 reserve are retained.

Exclusive ledger/result creation prevents automatic duplicates. Bounded500KB
hashed raw reports preserve failed/partial evidence; non-passing workers cannot
become successful completion. Stop/final-state reconciliation is explicit;
stopping or nonzero tasks remain uncertain closure rather than rewritten proof.
Actual raw timing presence is distinct from planned timing rows. No native
counter session, installation, source/precision rescue, production dispatch or
official submission path was added.

## Final SHA256

| Artifact | SHA256 |
|---|---|
| `controller.py` | `1c9e664635fc31637e5f5629fc6bf37a03f6121eef69952ef4a8ed6cf401d062` |
| `test_controller.py` | `96b4f7ccea9965c1096a4a99fd5c2b3b9214f673d79a07d24e5baccf389b648d` |
| `CONTROLLER.md` | `1e8bfa795e04eb73ea7dde9537897ce9de212e51f093817148a125c378b3608c` |
| `worker.py` | `bc93f136f4943e28ce3fd6be91cf67639ee4172ea9195b715bc762ac80a13d83` |
| `REVIEW.md` | `19a2a272e2b3640fe8d1847f25355612bec7cc0406cd62cee1555ce0a5b2e7be` |
