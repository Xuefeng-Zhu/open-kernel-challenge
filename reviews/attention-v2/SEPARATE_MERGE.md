# Attention A04 — full-only independent final merge

Prerequisite A03 passed first, source05922b7/SHA3a86e835, all16prerequisites and
12scoredshapes against exactofficialcache/outputoracle. No newalgorithm was
introduced before that GPUbaseline was established. Source separate_merge.py,
SHA256ddd9065c3fe444e7e5b97ed62101c7826b7745422e618aec80d910b19cbbb1a0. No third-party competitor code used.

This experimental function implements fullattention only and explicitly rejects
sliding; it is not a completecontestentry. Harness preserves originalcorrectedv2
for both slidingarms as an unchangedcontrol. This makes fullcounterelimination
and full-onlyfactorisolation explicit instead of modifying sliding's algorithm or
silently retaining a globalticket in the newcandidate. No OKCupload/check allowed.

Exact change: originalfusedattention kernel becomes _partial, publishing the same
acc/m/l scratch. The original serialmergerecurrence/norm/gate/store moves into a
second _mergekernel, ordered by CUDAstream completion. SPLITS1keepsoriginaldirect
output. _num_splits/QKnorm/RoPE/math/precision/newtokenownership unchanged. Scratch
andoutput allocated perinvocation/capture, no globaltensorcache, atomiccounter or
sharedtickets. No fixedscheduling/prefetch/FP8Q/doublehead/TMArewrite.

Four ASTgates check unchangedprologue/attention/newtokenbody, unchangedpartition/
normalization helpers, identicalmergearithmetic andabsenceoftickets/atomics. The
first localtest run had3/4pass: prefixassertion wrongly counted3statements rather
thanpbaseassignment+3stores. The test was corrected to4; algorithm source unchanged.
This is a testexpectation error, not a numerical/GPU result. Actual4/4ASTgates and5/5harnessgates pass, syntax passes beforecommit.
GPU correctness/timing pending; no numerical performance claim from ASTtests.

```sh
.venv/bin/python reviews/attention-v2/test_separate_merge.py
.venv/bin/python reviews/attention-v2/test_modal_worker.py
.venv/bin/python -m py_compile reviews/attention-v2/separate_merge.py reviews/attention-v2/modal_worker.py
```

Budget carried unchanged: conservative1.378907610of2USD afterA03,paid0. OneA05
reserve0.50fits. Existingauthentication only; no newcredentials/cap/payment/terms/
push/scoredsubmission. Source/experiment/results committed separately locally.
