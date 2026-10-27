# Independent producer artifact review: declared gate FAIL; branch STOP

Reviewer: `/root/offline_transpose_review`, with an independent local trig-origin
check by `/root/offline_transpose_review/fa3_mapping_source`.
Only this report was written. No compiler, container, service, GPU, network,
authentication or Git action was performed by these reviewers.

The offline compiler reached native code generation, but candidate D1 sliding
has32 bytes of stack and static PTX/SASS local-access sites. It fails the
predeclared absolute zero-stack/local gate. **Stop the entire declared branch**:
no remaining candidate compiles, GPU/official check, threshold relaxation,
configuration rescue or retrospective B32-only promotion is justified here.
This is not a measured slowdown or demonstrated extra register spill.

## Provenance and complete retained evidence

Frozen kernel SHA256:
`eae6ca35c88a5a45612f13ee5fa7696e86dc67b0fcf4fdcb14aceed1b9f43480`.
Helper SHA256:
`f8cc6a6b72a0ba9076d2a4ece018a4ecce2140a9161ac8aa35b4eb9e4442f5fd`.
Selected source remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
its extracted control remains
`cfefc9b9a9b2b175945fe7e6990d7c5fe46efb6e752acf88683c0dee4bb05356`.

The source-bound compiler manifest is
`evidence/compiler-manifest.json`, SHA256
`2133c3ccccba2767b3513b5992b91ab342016ed52800b5c3ceb57bbf3f59fa68`.
It records Python3.12.15/aarch64, pinned Triton3.8, explicit SM90 target,
four warps/three stages and `stopped_on_candidate_resource_gate`.
All46 unit-artifact digests independently match their original local files.
No source/options change or variant retry is recorded.

I read the archive in memory, without extracting over files, and checked its
exact member set, lengths and digests against `evidence/ARTIFACT_INDEX.json`:

| Evidence | SHA256 / size |
|---|---|
| Artifact index | `53198f61678daa01deef6642169c999b13b45debbc5d153117fe90d96d2eb9dd` |
| `native-text-artifacts.tar.gz` | `edd799ac4acb474ac7166f6bd6f52640f3ad235304fa734cc141095a11e65ebd`,994395 bytes |
| Archive contents | Exactly50 regular, safe relative members;9800647 raw bytes |
| Execution receipt | `bd8aacd5e2d792f7ba92a7b2364e6d074eba552d41449c4941bdd2ee040b2f99` |

Every indexed member matches, with no duplicate/missing/extra entry. Tar uid,
gid and mtime are0. Archived compile text equals the originals byte-for-byte.
Both copied JSON receipts/manifests match originals. All six local-only cubins
match the index's published digests; cubins/wheels/credentials are not archived.
The complete generated text, frozen preparation and earlier validation failure
are retained. The index reports six secret-pattern counts0; this review verifies
archive integrity, not an independent broad credential/security audit.

## What actually compiled and stopped

Four controls compiled first, followed by D32full and D1slide candidates.
Candidate D16slide and D32slide were **not compiled**. All rows report0 ptxas
spill stores/loads and cubin LOCAL0; those fields do not erase explicit local
instruction sites or stack usage.

| Case | Registers | Stack B | Dynamic shared B | Cubin shared B | PTX ld/st.local | SASS LDL/STL |
|---|---:|---:|---:|---:|---:|---:|
| Control D1slide |96|32|12544|1024|96/64|96/64|
| Control D16slide |96|32|12544|1024|96/64|96/64|
| Control D32slide |96|32|12544|1024|96/64|96/64|
| Control D32full |96|0|20480|1024|0/0|0/0|
| Candidate D32full |74|0|34944|1024|0/0|0/0|
| Candidate D1slide |75|32|34944|1024|96/64|96/64|

D32full passes only the helper's mechanical resource screen. Its35968-byte
dynamic-plus-static/reserve model is retained separately from34944 dynamic.
This neither completes native correctness/lifetime review nor permits promotion.
D1slide fails both stack fields and all four PTX/SASS local-site predicates,
despite remaining under96 registers and40KiB dynamic shared. Missing later
cases cannot be inferred from similar source or this first sliding result.

## Local sites are inlined RoPE trig slow paths

Source `kernel.py:34` computes RoPE cos/sin. In candidate D1slide:

- PTX line40 declares `__local_depot0[28]`. LLVM IR line11 onward has32
  `[7 x i32]` allocas, corresponding to16 cosine and16 sine expansions.
- LLVM IR478 loads `__cudart_i2opi_f`;483–495 writes the reduction array;
  498–513 performs indexed reads;517 identifies
  `__internal_trig_reduction_slowpath.exit`, and602 reaches `__nv_cosf.exit`.
- PTX354 references that trig constant table. Local stores369/378 and
  loads381/382/388 are within the inlined range-reduction path and carry
  source locations for the cos/sin expression. SASS299/307 retains STL;
  LDL sites begin at319.

These are **inlined range-reduction arrays**, not out-of-line trig calls:
the PTX contains one entry, no `.func` definition and no call instruction.
The same mechanism and32 arrays exist in all three sliding controls, e.g.
`control.D1slide.ptx:341,356,365,368–375,377`. Both full cases lack it.
Ptxas reports0 register spill bytes throughout. Thus zero spills does not mean
zero local instruction sites, and this failure does not establish a new local
cost introduced by the transpose. Whether conditional slow-path sites execute
for actual RoPE angles, or contribute meaningful runtime cost, is unmeasured.
The baseline also fails our absolute criterion; the criterion is not relaxed
after observing that fact.

## Bounded native mechanism check

Both compiled candidate specializations contain eight native historical
M64N8K16 BF16/BF16-to-FP32 WGMMA sites: four QK and four PV. D32full PTX
QK1034/1040/1045/1050 maps to `kernel.py:180`; PV1258/1263/1268/1273 maps
to196, with transposed-A PV flags. Its SASS HGMMA groups are1973–1979 and
2185–2207. D1slide has the equivalent groups at PTX3015–3031 and3247–3262.
This is codegen evidence, not executed work or utilization.

The16 `mma.sync` sites in each candidate are separately identified: eight
source locations at `kernel.py:236` for the preserved N16-tail QK, and eight
at256 for tail PV. They are not historical-loop fallback. D32full PTX sites
span2965–3004 and3403–3465; D1slide spans6969–7469. No global MMA-site
rejection was applied to that intentionally preserved tail.

Basic completion/publication instructions are visible: D32full QK proxy fence
1029, commit1053, wait-to-zero1058; P/PV fence1253, commit1276 and
wait-to-zero1280. SASS dependency waits are1981 and2245. Async-copy
commit/wait/CTA-barrier sites also survive. This bounded check **does not
certify** every ring-buffer path, masked-copy lifetime, cross-warp reduction,
online carry, tail/layout transition or precision equivalence. Those broader
native semantic gates remain unresolved after the resource rejection.

An additional independent PTX parse confirms the retained
`evidence/ASYNC_COPY_WIDTHS.json`, SHA256
`25a8feb4a786785257f29d067577f4e456d0a544bdb81f01c45f55d2223ec3a3`:
both candidates have48 cp.async sites, all with4-byte copy-size fields.
Control D32full has six16-byte and twelve8-byte sites; control D1slide has
three4-byte, four16-byte and eight8-byte sites. The candidate therefore does
not preserve the control's issued-copy widths by matching stage options.
This is a concrete lowering difference and possible cost risk, not measured
traffic, stalls, executed counts or a speed comparison. It does not change
the stop decision or justify a tuning rerun.

## Closure and interpretation

The saved receipt records exact Attention container exit1 and verified removal
at2026-10-27 00:01:23.879534UTC, no shared-engine stop, no GPU/cloud resource,
no credential action, paid spending0 and official checks0. Reviewers did not
query live services. The6.28-second compiler wall time and9.93-second host
container wall time are CPU execution durations, not kernel latency.
Root later reported a read-only shared-engine observation of Stopped at
00:05:16UTC, without requesting its stop. Exact own-container absence had
already been verified while the engine was running; the later observation
does not identify an actor. This reviewer did not query or restart the engine.

The earlier attempt stopped during cross-Python manifest validation and never
called the compiler; it remains separately preserved and explained in
`HELPER_PROTOCOL_FIX_REVIEW.md`. This later invocation genuinely compiled six
units and then obeyed the first resource failure. No native GPU execution,
original-oracle/replay result or complete-call timing exists. Selected1.233633
and the A2-11 rollback remain unchanged. Full-producer admission is **FAIL**;
there is no demonstrated speed result and no justified gate-rescue rerun.
