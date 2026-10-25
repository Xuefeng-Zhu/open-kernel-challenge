# P043 — lossless pooler weight compression fused into GEMM

Status: local preflight complete; no native result at this source commit. P039 remains selected.

**Experiment card / 实验卡:** pooler m8/16/32 is a weighted cross-load gap. Store static BF16 sign/fraction bytes plus packed exponent deltas, reconstruct every original bit inside an S8 GEMM, and test whether fewer weight bytes outweigh decoding and the existing custom-kernel deficit. 三条小批量 pooler 路径改变，其余保持 P039；字节减少不是速度证明。

Parent: `candidates/projection_router_m1_fp32_gemv.py`, commit `cdf5bc2eb1d8c066f9a5cbf1e8f40d6605e6f8da`, SHA256 `0c0c41119ef925b03a33ccab230bc64176cc22c242e3fea77ce01e317376defc`, official job `d5636ac8d1404abd`, score 1.173238998969285. It is immutable. Candidate: `candidates/projection_pooler_lossless_codec.py`; matched raw control: `candidates/projection_pooler_lossless_raw_control.py` (only the host COMPRESSED constant differs). Exact hashes are in the admission model and CPU report. The raw control is **not GPU tested** and is not uploaded in this one-check experiment.

## Permission and semantics

[Official Projection TASK](https://openkernelchallenge.dev/problems/muse_bf16_proj), saved unmodified as TASK.md, says `prepare(weights)` runs once untimed for up to 300 seconds. Its Weights rule explicitly permits lossless weight compression and decoding inside kernels. This is static weight-only preparation. Activations are never read by prepare; no outputs or input-derived state are cached, no address identity/value test selects a path, and all activation-dependent arithmetic stays in `kernel_fn`. Each call owns its output and split scratch. The state is immutable during calls. Different weights require a new prepare per the contract.

For each N-row/K256 block, store sign+seven fraction bits, four-bit exponent deltas from the minimum, base exponent and escape index. If exponent span exceeds15, an escape stream stores all eight exponent bits. Reconstruct `sign<<15 | exponent<<7 | fraction` and bitcast to BF16 **before** tensor-core dot; FP32 accumulators and FP32 split reduction then BF16 output. No FP8/reduced-precision weight quantization. Zero/subnormal/Inf/NaN payloads roundtrip exactly as storage; the official arithmetic oracle requires finite outputs on its generated inputs, not successful multiplication of arbitrary NaN/Inf values.

GPU layout is contiguous sign/fraction and nibble streams, Kblock-major metadata, and indexed escape rows. K64 tiles never cross a K256 block. Tail lanes are explicitly zero after bitcast. An allocated dummy escape row protects zero-escape preparation and counts in the byte guard. Static complete encoded size (all streams, padding, dummy and32-byte header) must be <=78% of raw; otherwise every pooler call uses P039's original path. This guard depends only on actual weights and is fixed throughout graph capture/replay. Original+transposed weights remain for robust fallback and other M values; total resident pooler storage **increases** by about49.1MiB, to177.1MiB.

## Why this experiment, and confounders

R06's S8/BM16/BN64/BK64 BF16 split GEMM was slower than the selected library path at m8. P043 retains that split/reduction arithmetic and fixed geometry, adds M tiling for m32, and changes weight representation/load/decode rather than guessing new tile sizes. It does not repeat the failed full-K QKV or pooler single-pass experiments. The matched raw control holds every setting and preparation constant, but remains unmeasured; comparison against P039 necessarily combines library-versus-Triton and encoding effects. One official check can reject or support the **complete candidate**, not prove compression caused a win.

Three actual full pooler matrices (seeds5151,5152,7171) reduced64MiB to51,489,568/51,469,600/51,469,600 bytes including all streams and32-byte header:23.2746–23.3043%. Exact full-matrix roundtrip and unchanged weights passed. About1.44–1.50% of blocks escape. The theoretical format maximum is about24%, before escapes/padding. Both nibble fixed slots and escape data count, even though the GPU masks nibble reads for escaped blocks.

Per valid weight, decoding adds integer extraction/shift/add/OR and a bitcast; normal blocks use SM+packed-delta loads, escapes SM+full-exponent. Five metadata bytes per row are logically requested each K64 iteration (four times per K256 block). S8 scratch is1/2/4MiB for m8/16/32 and is written then read. M32 repeats weight reads for two M tiles. Transaction granularity, load redundancy, integer issue cost, spills and register/occupancy effects are **not measured**. No claim of an HBM bottleneck or actual saved physical traffic.

Reproducible model: `python -B projection_p043_model.py`. Six loads are equally geometrically combined, each family weighted62. Target pooler fractions are36.12%,29.12%,28.82% in m8/16/32 loads. Uniform target cuts5/10/20% model about+0.79/+1.61/+3.29% overall with other timings frozen. Treating the23.27% representation reduction as equal latency reduction gives an optimistic conditional whole-score value; it is not a prediction. Historical R06 m8 needs12.96% of its own latency removed merely to reach P039. The matched m16/m32 raw control lacks native timing.

## Predeclared gates

- Profile must show the codec partial/reducer on all three target shapes, proving that the static guard used the candidate rather than fallback.
- All official prerequisites and30 scored cases must pass; preserve precision and cache/state invariants.
- Each target normalized latency cut must be >=10%, no target may regress>5%, and frozen-parent target-only modeled aggregate gain must be>=1.5%.
- Report every unchanged path shift>5%. Cross-run drift and overlapping confidence intervals prevent causal overall-gain claims; an aggregate point alone never promotes.
- Any failure or gate miss preserves P039. No post-result tile/split sweep in this experiment.

## Actual local validation and limits

On macOS arm64, Python3.12.13, torch2.14.1 CPU, CUDA unavailable:

```sh
python -B -m unittest discover -s tests -p 'test_projection*codec.py' -v
python -B static_check.py --submission candidates/projection_pooler_lossless_codec.py
python -B static_check.py --submission candidates/projection_pooler_lossless_raw_control.py
python -B check.py --device cpu --suite all --submission candidates/projection_pooler_lossless_codec.py --output results/router-p043-lossless-codec-cpu.json
python -B projection_p043_model.py
```

16 tests passed. Includes65,536 bit patterns through normal and escape representation, signed zeros/subnormals/Inf/NaN payloads, exponent-span15/16, odd N/K, empty splits, independent flattened GPU-address model, sparse power-of-two probes, dense/outlier/zero inputs, m2/7/8/9/16/17/31/32, untouched weights/state and CPU same-address A→B→A. Noncontiguous and incompressible inputs fall back correctly. AST confirms all preexisting kernels and untargeted dispatch remain unchanged. The initial negative storage-guard test wrongly assumed sorted all-bit patterns were incompressible; their within-block exponents were narrow. The fixture was corrected to interleave exponents, and the test passes; no candidate change was needed for that fixture correction.

39 official cases passed the local CPU arithmetic model. This is not Triton execution, FP32 tensor-core reduction-order proof, GPU performance, captured graph A→B→A, concurrent-call validation or compiler-resource evidence. A native official check is still required. The official harness covers its own fresh inputs/graph timing; do not label dedicated adversarial graph/concurrency coverage as run unless a receipt proves it.

Independent review found no definite contract, bounds, precision or scratch ownership defect. A single bounded correctness/performance falsification check is defensible after storage and bit-exact gates. Source is our own representation/decoder built from preserved project code; no third-party implementation was executed or copied. Applicable Apache-2.0 text remains embedded for standalone upload.
