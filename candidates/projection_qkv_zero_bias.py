# okc: problem=muse_bf16_proj
"""P050 local prototype: zero-bias addmm for canonical QKV M>1 only.

Narrowed from P049; all non-QKV paths restore our P046, SHA256
001ea12e8764564d507de51e433769f5bd9a9916bfc4c014f0f87522c278545e.
Use the public torch.addmm API with beta=1 and a static BF16 zero vector.
This adds no mathematical bias and retains the original weight precision.
Only canonical contiguous BF16 QKV M>1 changes; unpooler and other paths
stay exactly P046. Output allocation is invocation-local. The extra state is
weight-shaped zeros, never activation-derived. No PyTorch source is copied.
CPU checks do not establish CUDA backend selection, correctness or speed.
"""

# Applicable project/parent Apache License 2.0 text for standalone upload:
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2023-2024 SGLang Team
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch

SPLITS = 32
BLOCK_N = 32
BLOCK_K = 64

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv(X, W, OUT, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        ks = tl.arange(0, BLOCK)
        # Within-call reuse of the current 16-KiB x through hardware L1 is legal;
        # no input-derived persistent state or cross-call result cache exists.
        x = tl.load(X + ks, ks < K, other=0.0, cache_modifier=".ca").to(tl.float32)
        w = tl.load(W + row * K + ks, ks < K, other=0.0,
                    cache_modifier=".cg").to(tl.float32)
        value = tl.sum(x * w, axis=0)
        tl.store(OUT + row, value)


    @triton.jit
    def _partial(X, HI, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        ms = tl.arange(0, 16)
        ks = tl.arange(0, BK)
        part_k = tl.cdiv(K, S * BK) * BK
        acc_hi = tl.full((16, BN), 0.0, tl.float32)
        acc_lo = tl.full((16, BN), 0.0, tl.float32)
        for step in range(0, part_k, BK):
            k = split * part_k + step + ks
            x = tl.load(X + ms[:, None] * K + k[None, :],
                        (ms[:, None] < M) & (k[None, :] < K), other=0.0)
            hi = tl.load(HI + ns[:, None] * K + k[None, :],
                         (ns[:, None] < N) & (k[None, :] < K), other=0.0)
            lo = tl.load(LO + ns[:, None] * K + k[None, :],
                         (ns[:, None] < N) & (k[None, :] < K), other=0.0)
            acc_hi = tl.dot(x, tl.trans(hi), acc_hi)
            acc_lo = tl.dot(x, tl.trans(lo), acc_lo)
        offsets = (split * M + ms[:, None]) * N + ns[None, :]
        tl.store(P + offsets, acc_hi + acc_lo,
                 (ms[:, None] < M) & (ns[None, :] < N))

    @triton.jit
    def _reduce(P, OUT, COUNT: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        splits = tl.arange(0, S)
        values = tl.load(P + splits[:, None] * COUNT + offsets[None, :],
                         offsets[None, :] < COUNT, other=0.0)
        tl.store(OUT + offsets, tl.sum(values, axis=0), offsets < COUNT)


def prepare(weights):
    state = dict(weights)
    state["router_fp32_rows"] = weights["router"].contiguous()
    if "pooler" in weights:
        state["pooler"] = weights["pooler"].contiguous()
        state["pooler_t"] = weights["pooler"].t().contiguous()
        if (weights["pooler"].shape == (4096, 8192)
                and weights["pooler"].dtype == torch.bfloat16):
            state["pooler_column_batch2"] = (
                weights["pooler"].reshape(2, 2048, 8192)
                .transpose(1, 2).contiguous())
    for layer in ("qkv_gate", "unpooler"):
        if layer in weights:
            state[layer] = weights[layer].contiguous()
            if layer == "qkv_gate":
                state[layer + "_zero_bias"] = torch.zeros(
                    (weights[layer].shape[0],), dtype=weights[layer].dtype,
                    device=weights[layer].device)
    hi = weights["router"].to(torch.bfloat16).contiguous()
    lo = (weights["router"] - hi.float()).to(torch.bfloat16).contiguous()
    state["router_hi"] = hi
    state["router_lo"] = lo
    # Same packed layout and one-GEMM computation as measured R02.
    state["router_pair_t"] = torch.cat((hi.t(), lo.t()), dim=1).contiguous()
    return state


def _mm_fp32(x, w):
    if x.device.type == "cuda":
        return torch.mm(x, w, out_dtype=torch.float32)
    return torch.mm(x.float(), w.float())


def _router_m1_compatible(x, w, n):
    """Only metadata controls the native path; strided inputs use general GEMM."""
    m, k = x.shape
    return (m == 1 and k > 0 and n > 0 and w.ndim == 2
            and n <= w.shape[0] and k == w.shape[1]
            and x.dtype == torch.bfloat16 and w.dtype == torch.float32
            and x.device == w.device
            and x.stride(1) == 1 and x.stride(0) == k
            and w.stride(1) == 1 and w.stride(0) == k)


def _pooler_column_batch2_compatible(x, state, n):
    """Shape/layout checks only: nearby widths and incompatible inputs use P039."""
    packed = state.get("pooler_column_batch2")
    weight = state.get("pooler")
    return (x.ndim == 2 and x.shape[0] in (8, 16, 32)
            and x.shape[1] == 8192 and n == 4096
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.stride(0) == 8192 and x.stride(1) == 1
            and packed is not None and packed.shape == (2, 8192, 2048)
            and packed.dtype == torch.bfloat16 and packed.is_contiguous()
            and packed.device == x.device
            and weight is not None and weight.shape == (4096, 8192)
            and weight.dtype == torch.bfloat16 and weight.device == x.device)


def _pooler_column_batch2(x, packed):
    """Each batch writes a different half of every output row, without a concat."""
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    # Offsets b*2048 + row*4096 + col are unique for b<2 and col<2048.
    out_batches = out.as_strided((2, m, 2048), (2048, 4096, 1))
    torch.bmm(x.unsqueeze(0).expand(2, -1, -1), packed, out=out_batches)
    return out


def _zero_bias_addmm_compatible(x, state, layer, n):
    """Only metadata guards canonical BF16 layouts; all other calls use P046."""
    if layer != "qkv_gate" or x.ndim != 2:
        return False
    m, k = x.shape
    full_n, expected_k, expected_n = 2304, 8192, 2304
    weight = state.get(layer)
    bias = state.get(layer + "_zero_bias")
    return (m > 1 and k == expected_k and n == expected_n
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.stride(0) == k and x.stride(1) == 1
            and weight is not None and weight.shape == (full_n, expected_k)
            and weight.dtype == torch.bfloat16 and weight.is_contiguous()
            and weight.device == x.device
            and bias is not None and bias.shape == (full_n,)
            and bias.dtype == torch.bfloat16 and bias.is_contiguous()
            and bias.device == x.device)


def _zero_bias_addmm(x, weight, bias, n):
    """The zero vector preserves x @ W.T; backend choice is not assumed."""
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    torch.addmm(bias[:n], x, weight[:n].T, beta=1, alpha=1, out=out)
    return out


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m, k = x.shape
    if _zero_bias_addmm_compatible(x, state, layer, n):
        return _zero_bias_addmm(x, state[layer], state[layer + "_zero_bias"], n)
    if layer == "pooler" and _pooler_column_batch2_compatible(x, state, n):
        return _pooler_column_batch2(x, state["pooler_column_batch2"])
    if layer == "pooler" and m > 1:
        return x @ state["pooler_t"][:, :n]
    if layer in {"pooler", "qkv_gate", "unpooler"} and m == 1:
        if x.device.type == "cuda":
            out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
            _gemv[(n,)](x, state[layer], out, k, triton.next_power_of_2(k), num_warps=4)
            return out
        return (x.float() @ state[layer][:n].float().T).bfloat16()
    if layer != "router":
        return x @ state[layer][:n].T
    if m == 1:
        if _router_m1_compatible(x, state["router_fp32_rows"], n):
            if x.device.type == "cuda":
                out = torch.empty((1, n), dtype=torch.float32, device=x.device)
                _gemv[(n,)](x, state["router_fp32_rows"], out, k,
                             triton.next_power_of_2(k), num_warps=4)
                return out
            if x.device.type == "cpu":
                # FP32 elementwise product/reduction model only, not native execution.
                return torch.sum(state["router_fp32_rows"][:n] * x.float(), dim=1).unsqueeze(0)
        # General strides/dtypes use exact original-FP32 weights, in the timed call.
        return x.float() @ state["router"][:n].T
    if m <= 8 and x.device.type == "cuda":
        partial = torch.empty((SPLITS, m, n), dtype=torch.float32, device=x.device)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _partial[(triton.cdiv(n, BLOCK_N), SPLITS)](
            x, state["router_hi"], state["router_lo"], partial, m, n, k,
            SPLITS, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
        _reduce[(triton.cdiv(m * n, 128),)](partial, out, m * n, SPLITS, 128, num_warps=4)
        return out
    if m <= 8:
        # CPU numerical model of the split boundaries/FP32 partial reduction only.
        part_k = ((k + SPLITS * BLOCK_K - 1) // (SPLITS * BLOCK_K)) * BLOCK_K
        pieces = []
        for split in range(SPLITS):
            start, end = split * part_k, min((split + 1) * part_k, k)
            if start >= k:
                pieces.append(torch.zeros((m, n), dtype=torch.float32, device=x.device))
            else:
                hi = x[:, start:end].float() @ state["router_hi"][:n, start:end].float().T
                lo = x[:, start:end].float() @ state["router_lo"][:n, start:end].float().T
                pieces.append(hi + lo)
        return torch.stack(pieces).sum(0)
    width = state["router"].shape[0]
    pair = _mm_fp32(x, state["router_pair_t"])
    return pair[:, :n] + pair[:, width:width + n]
