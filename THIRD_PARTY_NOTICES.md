# Attribution and license scope

This project studies the **Muse Spark Open Kernel Challenge**, run by Paradigm with
Meta and Andromeda. It does not claim endorsement by those organizations or GPU MODE.

- `submission.py` is derived from the plain PyTorch starter published at
  https://openkernelchallenge.dev/api/v1/problems/muse_bf16_proj. Its only algorithm is
  the documented dense matrix product; the first-line task marker and introductory
  description were added locally. No optimized kernel is claimed.
- Files under `official/` are unmodified selected-task material from that API,
  including its SGLang baseline, correctness reference, checks, template and task/agent
  documentation. The raw API response, retrieval time and file hashes are retained.
  Do not interpret this project's LICENSE as relicensing third-party material.
- The baseline identifies **SGLang**, commit
  `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`:
  https://github.com/sgl-project/sglang/tree/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
  SGLang is Apache License 2.0; its license text is preserved in `LICENSE`.
  The isolated router candidates adapt its documented BF16 hi+lo router technique;
  each file/experiment identifies its own threshold, layout or fusion modification.
- The challenge's [official credits](https://openkernelchallenge.dev/rules#credits)
  identify **GPU MODE reference-kernels** and **kernelbot** as foundations for its
  judge, problem format, harness and evaluator:
  https://github.com/gpu-mode/reference-kernels and
  https://github.com/gpu-mode/kernelbot. They are licensed under the June 9 Researcher
  Reciprocity License. That license is preserved under `licenses/`, including its
  attribution and Training Use restrictions. No GPU MODE evaluator source is vendored
  separately; the official challenge files retain any applicable upstream rights.

Independent project additions are provided under Apache License 2.0, subject to the
third-party rights above. The [challenge terms](https://openkernelchallenge.dev/terms.pdf)
also govern any code uploaded to the challenge; an off-board check is still an upload.
Official terms and technical rules must be reviewed at their original URLs before
participation. No legal PDF, unrelated task bundle, CLI binary, account configuration,
credential or private authentication record is included in this repository.


Attention testing scope: reviews/attention-v2/official contains unmodified public
muse_attention_decode files verified against the official API on2026-10-24.
The user's transcribed attention v2 source and its recorded minimal compiler fix
are preserved separately from the projection project. The Modal harness uses only
these official files, our local source and the existing authorized SGLang image.
No third-party competitor repository source is copied or executed in this cycle.

Attention E5 (`reviews/attention-v2/mixed_fa3.py`) adapts only the full-verify path
from the official attention baseline: SGLang's FusedQKNorm, cache-byte store,
FA3 and output norm/gate API sequence. The candidate identifies the upstream
commit and includes the applicable Apache-2.0 text. Its other paths retain our
preserved E2 source. No competitor source is copied or executed; the high-level
mixed-dispatch idea is independently implemented with the official contract.

Attention A2-05 (`reviews/attention-v2/fa3_prologue_component.py`, embedded in
`mixed_fa3_fused_prepare.py`) adapts RMSNorm/interleaved-RoPE equations and head
indexing from SGLang's `fused_qk_norm.py` at the same pinned commit, Apache-2.0:
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py.
Local modifications add direct FP8 query output, scaled FP8 K/V stores, the
direct FP8 conversion, and live cache-row-zero handling. The original
vendor optional fused KV path reserves zero, which differs from this task.
Applicable Apache-2.0 text remains in LICENSE and the standalone source. No
competitor whole source/module is executed or imported.

Attention A2-14 (`reviews/attention-v2/mixed_fa3_deferred_denom.py`) is an original
narrow denominator-recurrence change to A2-13. Conceptual references were the
FA3 softmax/pipeline source at commit
`060c9188beec3a8b62b33a3bfa6d5d2d44975fab` (BSD-3-Clause), and FlashInfer decode
source at `a88349f9f43df74d31d1d52ad5aa20c28824a790` (Apache-2.0).
No code from either was copied or executed. The existing SGLang adaptation and
its notices remain. Exact links and distinctions are in
`reviews/attention-v2/strategy/STRATEGY_REPORT.md`.

The strategy folder also preserves a small public historical receipt from
MadanChaollaPark/open-kernel-experiments commit
`40154c27d4b694c7d641c7959b00d7ff24a7c2e4`, with source/receipt URLs and hashes in
`public_comparison.json`. It is third-party evidence, not this project's result.
The public competitor source was inspected but not copied into or executed by
the candidate. Its original additions are Apache-2.0 under its own notices;
upstream rights remain separate. No competitor performance claim is adopted as
our measured result.

Attention A2-15 (`reviews/attention-v2/mixed_fa3_page_lookahead.py`) is an original
page-address scheduling change to the preserved A2-13 source. It retains the
existing SGLang notices. Triton's public load/range documentation informs the
API semantics; no new upstream or competitor code was copied or executed.
The local diagnosis and tests do not claim native compiler or GPU evidence.
