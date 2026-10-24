# E3 — loop-scoped pipeline attribute, E2 parent

Parent `fixed_chunks.py`, SHA7efbb370d2828effabdc7ed717d2e7d5785f9ee0da55380fad6e591adf7c8782,
passed official job7edded086c8a47f2 at0.7958818006×. Candidate `pipeline.py`.
Generate: `.venv/bin/python reviews/attention-v2/prepare_pipeline.py fixed_chunks.py`.
Two CPU/AST invariants pass (`test_pipeline.py`): pipelined body matches parent,
non-target loop matches parent, all other AST and host logic identical after
removing the single constexpr dispatch flag. Only full B16/32 targets the loop.
Chunk8192, merge/tickets, math, precision, block widths, launch stages3/warps4,
and new-token owner remain unchanged. Both q_len1 and8 are supported.

Official Triton range docs distinguish loop num_stages from launch num_stages:
https://triton-lang.org/main/python-api/generated/triton.language.range.html .
The former attempts to pipeline more loop loads, while launch stages targets
loads feeding dot. Actual emitted code/resources must be compared; adding an
attribute does not prove overlap or speed. Proposed A09 is a short correctness/
compiler probe, not a timing benchmark. No official check until it is justified.
