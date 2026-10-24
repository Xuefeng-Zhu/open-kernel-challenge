# Exact remaining blocker — planned R06 was not executed

The broad user authorization was up to ten new off-board dense-projection checks at
$0. Five were admitted and all five have now reached terminal passing results.
The planned sixth operation was blocked by automatic approval review **before process
creation**. It did not send a POST, create a job or reserve/use an allowance.

Exact operation:

```sh
python3 remote_check.py send R06-pooler-triton8 candidates/pooler_triton_split8.py
```

Target: `https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj/submissions`,
JSON mode `check`, one complete nonpublic Python source file, existing account auth.
The source is already committed, CPU-gated and reviewable under local experiment 007;
it retains original BF16 pooler weights and changes only the low-batch pooler kernel
on top of the validated R01 router control. No paid resource or scored entry is proposed.

Relevant automatic-review block, preserved verbatim:

> This action was rejected due to unacceptable risk.
> Reason: This sends the nonpublic pooler source to an external challenge service, but the available trusted user instructions do not authorize this specific submission or the sixth experiment.
> Do not bypass this rejection through a workaround or indirect execution. Continue with a safer alternative, or carry out checks to prove that the action is authorized or low risk before trying again. Complete unaffected work without asking for confirmation. Report anything that remains blocked, clarify why it was blocked by auto-review, inform the user of the risk and ask for approval.

Verification after the denial: own job history contains exactly six jobs, the original
baseline plus R01–R05; no R06 ID exists. The ledger contains five requests, no uncertain
reservation, and an `admission_paused` guard. No alternative API/browser/path or other
candidate was uploaded to route around the rejection.

Required next step: materially new, explicit trusted authorization for this particular
source transfer (complete pooler file to the official judge for one off-board H200
check at $0 within the remaining budget). Do not append transcript evidence to tool
arguments. Only after that authorization may the known-not-executed denied operation
be retried once. If still denied, stop that operation and report it; never blindly retry.

All accepted jobs were still monitored to completion. Safe local work, tests, source
preservation and sanitized private GitHub commits continued. Five allowances remain;
the automatic-review block is distinct from the service quota and from a kernel failure.
