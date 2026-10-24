# Framework choice in the contest environment

The [current official environment guide](https://openkernelchallenge.dev/llms.txt)
and [technical rules](https://openkernelchallenge.dev/rules#environment) explicitly
state that the supplied `lmsysorg/sglang:dev-muse-spark-open` image provides and permits
**Triton, CUTLASS/CuTe DSL, FlashInfer and DeepGEMM**, as well as PyTorch and TileLang.
Submissions have no network access to install anything else. The exact evaluator/image
digest returned by our baseline run is preserved in `results/h200-check.json`.

This verifies the organizer's declared availability and permission, **not an import
test or version/API compatibility test of every framework**. No extra GPU call,
framework installation or runtime probe was made. The judge result does not report
all framework versions, and a mutable public image tag is not a reproducibility pin.

For this dense-projection project, first test the isolated PyTorch controls based on
the exact SGLang hi+lo router path: the measured slowdown is currently understood at
that level. For a later custom fused kernel, use **Triton first** to keep one small
kernel and launch configuration easy to change and compare. Do not port the project
to four frameworks simultaneously. Consider CUTLASS/CuTe or a supplied library only
after a measured limitation justifies the extra implementation/build complexity.

FlashInfer/DeepGEMM being present does not establish that a particular API/version
matches this BF16/FP32-router workload. Any candidate must retain BF16 activations
and original BF16 weights; only the router has the explicit BF16 hi+lo exception.
Switching a BF16 projection's weights to FP8/FP4 is not an allowed optimization.
GPU validations require user authorization; the later ten-new-off-board-check budget
permits autonomous meaningful experiments within that ceiling, at $0. Package
availability itself grants no additional run, scored-entry or spending permission.

Update: remote R04 (`e3bc1481851f4a23`) has now actually compiled/executed the custom
Triton kernels in the supplied H200 judge and passed 9+30 correctness cases. This
verifies usable Triton for that run without installing anything new. It does not
constitute an import/API test of CUTLASS, FlashInfer or DeepGEMM, or report their versions.
