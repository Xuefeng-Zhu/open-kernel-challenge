# KernelAgent / KDA H200 trial

The selected Attention source remains SHA256 `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`. No kernel math or champion dispatch changed.

KernelAgent was installed in an isolated local environment at upstream `e0647170da36ef9b059ac0bd3d60103aa4ed378b`. **27 upstream CPU tests passed**; its parser preserved all four values in the existing wide-format FA3 NCU export. Actual `get_gpu_specs("NVIDIA H200")` returned `None`. The stock candidate benchmark uses eager `triton.testing.do_bench`, so it cannot replace OKC graph scoring. Full generation/optimization was not run: no external model provider was authorized or configured, no keys were searched, and the intentionally credential-free provider check returned “No available provider”. No generated candidate or performance gain is claimed.

KDA and its exact pinned KernelWiki/ncu-report-skill were read locally. Blackwell defaults were adapted to the existing H200/NCU command contract; no external agent was started. Existing Modal access worked without credential renewal.

One H200 diagnostic, `ap-NIeb23lk3RtcAOgTimYjQM`, failed in the **first baseline case**, before paired timing or NCU capture, because our byte-comparison helper called `view(uint8)` on a 0D floating tensor:

```text
RuntimeError: self.dim() cannot be 0 to view Float as Byte (different element sizes)
```

The worker ran 21.219 seconds. The app was explicitly stopped and independently observed **stopped, tasks "0"**. The old controller initially reported a false shutdown failure because it required an integer task count; the supported CLI returned a string. Raw failure artifacts and original source remain immutable locally. Actual charge was not queried; the run reserved $1.20 of existing free credit, with the existing $0 paid limit and no billing changes.

The isolated repair flattens tensors before byte views and accepts strictly validated integer/string task counts. Real macOS PyTorch 2.14.1 CPU reproduced the old scalar error before the fix. **40 combined CPU tests passed**, including scalar/BF16/FP8/noncontiguous ownership checks, actual shutdown records, command construction, report validation and bounded lifecycle tests. The repair has **not** been run on a GPU; no baseline timings, paired gains, counters or bottleneck conclusion exist from this batch.

Native execution is paused under the latest instruction to fix only the local harness. Only Frank may run future `okc check`; nobody runs `okc submit`. The earlier v4 check and exact admission chronology are documented separately in `../user-v4-probe/RESULTS.md`.

`harness-repair/` is the exact repair source snapshot. Its execution expects the original isolated sibling workspace layout (`kda-h200-scalar-repair/` beside `attention-fusion-lookahead/`), supported existing local config, and the frozen local manifest. It is a diagnostic harness, not an OKC candidate. Next verification, when resumed, is the frozen same-GPU champion/v4 batch; no tuning or additional official evaluation has been performed.
