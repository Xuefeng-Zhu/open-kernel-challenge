# A2-15 native diagnostic: blocked at authentication

At **2026-10-25 08:31:05.922221 UTC**, the existing official Modal CLI's
read-only app inventory failed with exit **1**: **`Token has expired`**.
This is an authentication failure, not an approval-review denial or an
uncertain launch outcome. The command completed; there is no pending launch.

Command, from the project root, with the pre-existing ignored owner-only
configuration (its contents were never read or displayed):

```sh
MODAL_CONFIG_PATH=local-notes/modal-auth.toml .venv/bin/python -m modal app list --json
```

No candidate upload, app creation, GPU execution, credential change, OKC
check or repository push was attempted. No app ID was created. Incremental
GPU credit consumption from this attempt is **$0**; the approved **$0.25**
allowance is unused. Historical conservative cumulative Modal accounting
remains **$1.787819802**. Current account credits, paid cap, billing and active
apps could not be verified. Historical stopped-app receipts are not a fresh
shutdown observation. No new app from this attempt requires shutdown.

The prerequisite is valid supported Modal authentication, established through
a separately authorized action. This diagnostic expressly forbids creating
or renewing credentials, so no retry or alternate authentication route was
attempted. GPU correctness, generated instructions, registers, spills, shared
memory and scheduling effects remain **unverified**. There are no new raw
compiler artifacts or timing samples to report.

Frozen local experiment commit: `a2c9a91b1dd347d11cb333b2f96b8973951cfa03`.

| Source | SHA256 |
|---|---|
| `mixed_fa3_fp8_qk.py` (A2-13 parent) | `e3d52fbaaa28ae97add209de050834e326671ea278c9caf3247cc20418118b9e` |
| `mixed_fa3_page_lookahead.py` (A2-15) | `fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c` |

The completed local preflight remains seven source/address tests and three
diagnosis tests passed, with exact commands and evidence in `preflight.json`.
These were CPU/static checks, not Triton execution, and were not repeated for
this documentation-only attempt. Both source files remain unchanged.

Independent scope review found two preparation issues, recorded before any
upload: the old proposal includes A2-14 and extra fixture files, beyond the
latest approval; the old replay probe omits current compile parameters.
`NATIVE_DIAGNOSTIC_PLAN.md` now permits only the two approved candidates and
one diagnostic program and identifies the required narrow instrumentation.
That native program is not yet implemented or GPU-tested. No older wrapper
may be launched unchanged.

中文实验卡：目标是确认下一页页号预取是否改变原生指令排程。现有 Modal
凭证已过期，检查在上传前失败；没有启动 GPU、消费额度或修改凭证。
候选代码和已有本地测试保留，排程与性能假设仍待验证。
