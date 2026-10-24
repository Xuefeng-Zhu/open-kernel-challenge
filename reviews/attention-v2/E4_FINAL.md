# E4 — complete final candidate

`final_attention.py` is byte-identical to E2 `fixed_chunks.py`, SHA256
7efbb370d2828effabdc7ed717d2e7d5785f9ee0da55380fad6e591adf7c8782.
All paths and kernel_fn entrypoint included. Only fullB16/32 uses static8192-token
history chunks; all other paths use original v2 partition. No E1 separate merge
and no inconclusive E3 loop attribute. Dtypes, precision, math, new-token owner,
and ticket protocol remain E2. Root projection submission is untouched.

Selection: E3's highest observed score0.797873× is only0.25% above E2's0.795882×,
inside overlapping CIs; modified target paths vary about±0.2%. Prefer simpler E2,
which also has recovered explicit same-address ABA results. This is a conservative
selection, not a claim that E2 is statistically faster than E3.

Final verification request is separately recorded in attention-cycle-budget.json.
Identical-source server deduplication must be honored and disclosed. Do not mutate
comments to manufacture a new run or claim a deduplicated receipt is a fresh trial.
The final matching hash links it to the full official contract check and A07.
