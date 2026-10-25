# Actual image inventory: independent result review

The CPU read resolves two eligibility questions: the installed Triton3.8.0
contains the inspected Gluon/Hopper source APIs, and installed FlashInfer
XQA rejects page1. This is a supplement to the earlier, historically correct
`../ELIGIBILITY_AUDIT.md` and library notes; those files are preserved unchanged.
No attention kernel was compiled or executed by this inventory.

Result SHA256:
`91c05d95ba02b797ff12d46c7a57318cdfba0b9741f0d23a430de371dd734d78`.
The launch receipt binds image`im-cPTbGpNiX3RarKResPgX7h`, source commit
`8e43b227981861b2514f4eb9f366ff923e3d4215`, worker SHA256
`5a287e350612a5f10356a5645e50689c51cdce80c175bb8d22f017cae1321f23`
and app`ap-I3MCH1onTGtkhqXUtej9Ru`. The report completed at23:26UTC with
GPU/context requested false, zero candidate launches and no vendor imports.

## Installed observations

| Evidence | Actual result and consequence |
|---|---|
| Triton | Distribution3.8.0; Gluon `__init__.py` present, SHA`d7b607f028fed01715550c5e5ef708b0bd71881ccda220fb7390248ec29ca904` |
| Hopper MMA API | Source SHA`c427a1f7fa5277d701d6cced0dc05ac3af4e5a32f9dc2675699fa254023a99dc`; `warpgroup_mma(a,b,acc,*,use_acc=True,precision=None,max_num_imprecise_acc=None,is_async=False,...)`, `warpgroup_mma_wait(num_outstanding=0,deps=None,...)`, `warpgroup_mma_init(value,...)` and `fence_async_shared(cluster=False,...)` are defined |
| Pointer async copy | Ampere async-copy source SHA`86e23dfd1231d2709e93a126c91349254440a930ef2196c6d5e3fcefc7e8f623`; `async_load(smem,pointer,mask=None,...)`, `commit_group(...)`, `wait_group(num_outstanding=0,...)` are defined and Hopper source re-exports the module |
| Barrier/TMA | Hopper mbarrier source exposes `expect`/`arrive` and imports other barrier helpers; TMA source defines descriptor/coordinate `async_load`. This is source presence, not a proven arbitrary page1 TMA gather or compiled dependency protocol |
| FlashInfer | Module present; `flashinfer-python=0.7.0.post1`, `flashinfer-jit-cache=0.7.0.post1+cu130`; distribution named `flashinfer` is null, which does not mean the module is absent |
| XQA page size | Installed `jit/xqa.py` SHA`b47625e738bcef4324e874a20fa4c3aee91c477dafd72b24480e3d235c3d20cc`, lines82/209 both reject page sizes outside`[16,32,64,128]`. **Direct installed XQA is ineligible for Muse page1**; relabeling pages or full-KV repacking is not an adapter |
| FA3 wrapper | `sgl_kernel/flash_attn.py`,16160 bytes, SHA`610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`, exactly matches the earlier same-image identity. Public `flash_attn_with_kvcache`/`get_scheduler_metadata` definitions are present. The 1122619328-byte extension was stat-only; its previous binary hash was not freshly reverified |

The ordinary FlashInfer decode/prefill sources also contain CPU metadata
transfers and graph/plan caveats. Selected excerpts are not a complete public
API capture audit, so they admit no naive per-call adapter and do not prove
that every FlashInfer backend is unusable. Any alternative needs an exact
GPU-live planning, page1, quantization and graph-replay proof.

`sglang` was found, but its origin was omitted by the local vendor-path guard;
this means **not inspected**, not module absence or a service access denial.
Likewise `sgl-kernel`/`cutlass-dsl` metadata null coexists with present modules.
Five fixed FA3 C++ source directory candidates were missing. This establishes
only those path misses, not absence of source elsewhere or a known installed
C++ revision. No scope expansion or new search is implied.

## Bounds, closure and interpretation

Independent arithmetic checks reproduce17 examined files:16 source files read
and hashed totaling888058 bytes, plus one stat-only extension. JSON is19382
bytes and selected excerpts total4660 UTF-8 bytes; no scan/export limit was
reached. The configured limits remain64 files/8MiB source/2MiB per file/4096
bytes per excerpt/256KiB JSON, walk depth4/32 directories/256 entries, and a
30-second worker alarm. Recorded walks used25 directories and167 entries.
The inventory itself took0.330900455s; controller child wall was0.472550191s.

The launch receipt preserves its initial stopping/tasks1 observation. Separate
reconciliation at23:26:29.524201UTC verifies the app stopped/tasks0 and all
listed apps stopped/tasks0. At23:27:25.788396UTC, delayed workspace gross delta
was$0 and paid cost$0, with no new other app IDs. This is not an exact app
invoice. The modeled resource bound was$0.00711 within reserve$0.02.

**Decision:** stop the direct XQA route before compilation. Installed Gluon
source availability passes the cheap prerequisite for a separate same-geometry
serial-versus-overlap compile falsifier; importability, accepted layouts,
strict precision, safe buffers, native overlap and an advantage over current
FA3 remain unproven. API presence alone is no speed or runtime certificate.
