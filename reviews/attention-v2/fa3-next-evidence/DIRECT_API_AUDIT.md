# FlashInfer direct TRT-LLM APIs: not admitted for this H200/page1 workload

**Decision: stop this eligibility branch; no prototype or GPU request.**
The public v0.7.0 TRT-LLM-generated runner explicitly admits SM100/103/107,
excluding H200 SM90. Both direct APIs reach that runner. Decode's automatic
SM90 route instead selects XQA, whose installed page1 exclusion is already
established. This is not a new planner-free SM90 backend.
[Runner architecture check](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/trtllm/fmha/fmhaRunner.cuh#L28-L45),
[decode dispatch](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/decode.py).

## Version and evidence boundary

The existing CPU image inventory, SHA256
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`,
records `flashinfer-python=0.7.0.post1` and
`flashinfer-jit-cache=0.7.0.post1+cu130`. It identifies these installed files:

| File | Bytes | Installed SHA256 |
|---|---:|---|
| `flashinfer/decode.py` | 174,952 | `df105a90074609961925686572b7b7e4930bf7e065003829f788452607932d6c` |
| `flashinfer/prefill.py` | 295,141 | `bd4aaa0a24462efbb6e6ddb12b3c98da9507e7b6ff6b22da3f40167ca66beb12` |
| `flashinfer/jit/xqa.py` | 9,348 | `b47625e738bcef4324e874a20fa4c3aee91c477dafd72b24480e3d235c3d20cc` |

The inventory preserves truncated direct-API signatures, not their full bodies
or the installed runner header. The public `v0.7.0` source below is nearby
version context, **not byte-identical proof for installed0.7.0.post1**. Current
unversioned docs identify themselves as0.7.2.post1 and are not used to certify
installed eligibility. No new image/source read or runtime call was made.
The remaining installed-version question cannot turn absent proof into admission.

## Exact direct routes

| Public API | Route in public v0.7.0 | Result for SM90/page1 |
|---|---|---|
| `trtllm_batch_decode_with_kv_cache(..., backend="auto")` | Chooses `trtllm-gen` for compute-major10; otherwise XQA | Routes H200 to the already excluded XQA page-size family |
| Same API, explicit `backend="trtllm-gen"` | Calls generated paged-attention decode | Shared generated runner excludes SM90 |
| `trtllm_batch_context_with_kv_cache(...)` | Calls generated paged-attention context | Same shared runner excludes SM90 |

These are the requested direct functions, not `BatchDecode/Prefill.plan()`.
The public decode source shows the automatic choice and explicit branches;
context calls the generated context operator directly.
[Decode function/branches](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/decode.py),
[context function](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/prefill.py).

The common C++ launcher obtains a `TllmGenFmhaRunner` through its cache for both
ForGen and Context modes. The constructor tests the architecture before a
compatible workload kernel can be selected. Its Q-type guards include BF16 and
E4M3, and K/V guards include E4M3, but those dtype guards do not remove the
architecture restriction. A divisible Q/KV head count is accepted by the common
launcher; that alone does not establish a GQA8 kernel for this workload.
[Common launcher](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/csrc/trtllm_fmha_kernel_launcher.cu),
[runner constructor](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/include/flashinfer/trtllm/fmha/fmhaRunner.cuh#L28-L45).

The installed XQA JIT excerpt explicitly restricts page sizes to16/32/64/128,
excluding1. This independently supports rejecting the known XQA fallback; no
XQA experiment or page repack is proposed. The fallback mapping itself is from
publicv0.7.0 and remains subject to the stated post1 source-binding caveat.

## Requested contract: what is and is not established

| Requirement | Audit finding |
|---|---|
| H200/SM90 | Explicitly excluded by publicv0.7.0 `trtllm-gen`; installed post1 header not retained |
| D64, page1, GQA8 | No admitted SM90 direct-generated combination; dimension support on other architectures is immaterial here |
| Original FP8 KV with BF16 Q or BF16→E4M3-rounded Q | Surface dtype support is not a compatible SM90 kernel or precision-equivalence proof |
| L1 decode / L8 speculative verify | Decode exposes `q_len_per_req`; context exposes query/cumulative-length tensors. Neither bypasses the architecture gate |
| Live GPU lengths/page table | Direct surfaces accept `seq_lens` and `block_tables`; this does not certify capture-safe installed implementation, masked page suffixes or legal per-replay mutation |
| Per-call planning/cost | No ordinary `.plan()` is required by these direct surfaces. Zero preparation cost and reusable input-derived plans are **not** established |

The context signature requires query/KV cumulative-length tensors and maxima.
Any conversion or construction from live inputs would have to remain in the
per-call graph. The Python direct routes also handle output allocation/scales
and scratch/counter buffers; changing return/container layout is not a measured
speedup. Exact post1 graph capture, dynamic-length correctness, cache stores,
original output tolerances and whole-call costs remain untested.
[Context interface](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/prefill.py).

First use of the generated module invokes build/load and artifact/meta-info
resolution; installed package/JIT-cache versions do not prove that every needed
cubin is already available. This audit performs no loader call, download,
compilation or permission expansion.
[Generated module loader](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0/flashinfer/jit/attention/modules.py).

**Final state.** Public-source architecture rejection is concrete; exact installed
post1 eligibility remains unverified rather than claimed tested/denied. Stop
without prototype, service call or performance prediction. No paid resource,
credential, install, official check or source-selection change occurred.
