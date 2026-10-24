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
