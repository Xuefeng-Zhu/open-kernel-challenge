# M64 BF16 QK compile probe

Status: **compiled on H200; native gate passed**. See `native-audit.md`. The
separate `../paired/ANALYSIS.md` reports correctness success and a substantial
whole-call loss against FA3. The original compile-only protocol follows below.

`candidate.py` is a byte-identical copy of exact A2-13
`../../mixed_fa3_fp8_qk.py`, SHA256
`e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e`.
Its existing `_fused` compile-time `QK_DOT` argument already controls the Q/K
operands. The two probe variants compile these same bytes with
`QK_DOT=tl.float8e4nv` (control) or `QK_DOT=tl.bfloat16` (candidate). There is no
candidate-source patch or runtime-dispatch change, and no A2-15 lookahead.

The Q path remains normalized Q → BF16 → E4M3 → selected operand type; cached K
and newly quantized K widen from E4M3. Both QK dots keep FP32 outputs and
`max_num_imprecise_acc=0`. PV stays BF16, softmax/partials FP32. Finite E4M3
operand values widen exactly to BF16; that CPU fact does not establish compiled
instruction choice or output correctness.

The synthetic TP8 full-verify specializations are B16/B32, L8/G8/M64/D64,
BN64/BL16, eleven 8192-token chunks, page1, QKV stride2304, page-table
stride86008, four warps, three stages. These follow the original dormant custom
path. **L8 uses its original integrated final-CTA ticket merge**
(`SEPARATE_MERGE=False`), including new-token ownership on the final split.
No separate merge kernel is substituted. B16/B32 can share compiled code because
batch is only a grid dimension; both are retained as explicit records.

`worker.py` only calls `.warmup()` and `_init_handles()` on one-element aligned
CUDA buffers, which are deliberately invalid for candidate execution. Never
launch these buffers. It saves TTIR/TTGIR/LLVM/PTX/cubin/SASS and driver resource
fields, package versions and any installed Triton VCS commit metadata. A 55s
worker alarm bounds this phase; the launcher must additionally provide a child
process kill timeout, Modal function timeout, one-container limit, no retries,
app watchdog and verified shutdown. The worker includes no launcher or credentials.

The worker reads installed FA3 Python wrapper source around scheduler/split
arguments. It explicitly labels actual CUDA scheduler dispatch unverified;
wrapper source alone cannot establish the selected compiled scheduler.

Local source checks:

```sh
python3 reviews/attention-v2/architecture-probe/probe/test_probe.py
```

Future admission gate: inspect history-QK and new-token-QK separately in
TTGIR/PTX/SASS for BF16 WGMMA, FP32 accumulation, commit/wait placement and
register/shared/spill changes. PV may already use WGMMA at M64: its presence
alone does not prove QK changed. If QK remains MMA, or needs relaxed accumulation,
reject this exact hypothesis. Compile success is never a performance result.
If QK passes, review resources before a separate representative correctness /
same-address A→B→A replay and paired whole-call comparison against unchanged FA3.
Historical custom L8 is substantially slower than FA3, so an architectural
instruction change alone does not justify adopting its dormant route.
