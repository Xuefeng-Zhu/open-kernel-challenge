# Independent review of the partial runtime result

**Partial diagnostic; do not relabel the remote run PASS.** The frozen worker exited 1 after 43.306 s with `ValueError: Expected one actual FA3 mainloop: full_verify_b32`. The failure is a parser attribution assumption, not missing FA3 execution or failed numerical correctness. This review used saved files only; it made no service, GPU, authentication or Git call and changed no frozen worker/candidate source.

## Exact failure and recoverable attribution

The predicate in frozen `compact_trace` selects GPU events only when their whole interval lies inside a CPU `user_annotation` interval, then requires exactly one `FlashAttnFwdSm90` match. B32 produces **zero** matches under that predicate, not multiple matches. The complete raw trace nevertheless contains exactly one B32 mainloop and one B16 mainloop.

| Evidence | Full verify B16 | Full verify B32 |
|---|---:|---:|
| CPU annotation start, µs | 2216356991130.737 | 2216356991993.452 |
| CPU annotation duration, µs | 289.499 | 422.343 |
| In-range CPU `cudaGraphLaunch` correlation |113|232|
| Actual graph ID |2|5|
| GPU-side annotation External id |16|32|
| GPU-side annotation start, µs |2216356991144.795|2216356991950.472|
| GPU-side annotation duration, µs |193.882|372.854|
| Actual correlated graph kernel count |5|5|
| Mainloops found by frozen CPU-containment predicate |1|0|
| Mainloops found by launch correlation/graph |1|1|

B32's FA3 mainloop begins at 2216356991953.492 µs, 39.960 µs before its CPU annotation timestamp. Its preparation and scheduler also precede that CPU timestamp. Conversely, B16's CPU range incorrectly admits two subsequent graph 0 DtoD copies with correlations 122/130; these are not selected graph 2 nodes. Thus CPU/GPU timestamp containment is unsuitable here. These facts do not establish why profiler clocks/ranges differ or actual out-of-order execution.

A separate offline parser can find the unique same-thread host `cudaGraphLaunch` within each host annotation, associate device events using launch correlations 113/232 and graph 2/5, and cross-check the matching GPU annotation's External id. It should report the observed number of mainloops rather than assume every future route has one. Preserve all names, nodes and correlations; reject ambiguous associations rather than widen timestamp windows. This is a proposed correction to analysis of retained data, not a change to the frozen worker or a reason to rerun the GPU.

## Current executed specialization and phases

Both cases have one identical full mainloop symbol, SHA256 `33239a4fe9825c98df5c44acd8bc412ee2c441fab7b704ffa6124fb68de96711`. It contains `CollectiveMainloopFwdSm90`, the tile tuple `cute::tuple<cute::C<64>, cute::C<128>, cute::C<64> >`, `cutlass::float_e4m3_t`, `float`, `cutlass::bfloat16_t`, and `VarlenDynamicPersistentTileScheduler<64,128,128,128,...>`. The complete symbol is retained untruncated in the raw trace. For these two actual selected Modal calls, it is the 64×128×64 mainloop tuple, not an inferred upstream 192×160 default. This does not map every boolean, PackGQA value or internal QK/PV instruction/operand type.

Both mainloop launches report grid `[132,1,1]`, block `[256,1,1]`, 255 registers/thread and 56,320 shared bytes. These are launch/static fields, not achieved residency, throughput or spill measurements. The trace's estimated occupancy 0 is not a valid conclusion about actual execution.

| Correlation-associated phase | Count in each case | B16 activity duration µs | B32 activity duration µs |
|---|---:|---:|---:|
| `_fa3_prepare_fused` |1|1.730|2.138|
| `prepare_varlen_num_blocks_kernel` |1|3.365|3.616|
| `FlashAttnFwdSm90` |1|187.621|365.462|
| `FlashAttnFwdCombine` |1|3.270|3.868|
| `norm_gate_kernel` |1|1.919|2.170|

Scheduler symbols contain `<1,true>`/`<2,true>` respectively; combine grids are `[8,1,16]`/`[8,1,32]`. None is labeled a resolved split count. Adjacent kernel timestamps show small apparent overlaps despite the same stream 7, so these values do not prove actual concurrency or a critical path. They remain instrumented activity, separate from unprofiled whole-call measurements.

## Identity chain and completed correctness

The current image reports H200/SM90, 132 SMs, Python 3.12.3, Torch 2.14.1+cu130, CUDA 13.0 and Triton 3.8.0. The selected source binding is unchanged `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The freshly read public wrapper is 16,160 bytes, SHA256 `610747f2c68495c5f3ba67f1247f0a1b4e4f7df116e2f89cd8e8983f0271f9ac`; the selected callable is the same wrapper object. Fresh schema exposes 37 arguments and four Tensor returns. Defaults remain `num_splits=0`, `pack_gqa=None`, `sm_margin=0`, `ver=3`; both verify fixtures supply scheduler metadataNone. The return-only flag is absent from native arguments. Actual auxiliary-return replay was not reached.

The extension file `/opt/sglang/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so` was completely hashed: 1,122,619,328 bytes, SHA256 `cdea0997a51412b8db8bf8da49d896a6c9d12a39bb1b548d840bb8dc428f13ef`, GNU build ID `e40ad4a2cb6c0e7d20e7d43f5d5c6647715f3f01`. The loaded mapping and current file both report inode 2100. This supplies a current Modal file/mapping identity chain, not an official-judge binary identity, exact source revision or memory-image digest. The bounded header search found no matching source in its stated locations. PackGQA, resolved splits and instruction-level QK/PV semantics remain unresolved.

Both original-oracle checks passed. Ten subsequent recorded byte/address/cache-invariance checks passed (two warmups and three pre-profile samples per case), for 12 completed checks total. Before-profile whole-call samples remain valid under the declared same-fixture CUDA-event protocol:

| Case | Three retained samples µs | Median µs |
|---|---|---:|
| B16 |199.423999,199.552000,200.320005|199.552000|
| B32 |385.472000,383.552015,383.103997|383.552015|

No after-profile samples, post-profile oracle checks, auxiliary replay or counter gate ran. No before/after invariance, serving concurrency, optimization speedup or numerical validity of the profiled outputs is claimed. Fresh inventory found NCU 2025.3.1, NSYS 2026.4.1 and CUDA 13.0 cuobjdump/nvdisasm. `RmProfilingAdminOnly:0` is a setting observation; counter access remains untested. Hardware-counter sessions: 0.

## Integrity and closure

| Artifact | SHA256 |
|---|---|
| Unmodified `result.json` | `8c23f8cc7bb2c368f5af4e0a6259bac3641a2ce69da0099c321cd2fad3e82fc9` |
| Raw trace,62,131 bytes | `c9ab784e4584f585060bb2c4b5dbf5b70479fb935b5805bf678e2fe1597d275d` |
| Trace gzip,5,670 bytes | `83c54f833471cc826008b97e7275c0f7da8f5ce3f558eafcd7d8c85d47f8216c` |

The saved `independent-cleanup.json` observes app `ap-G4ZFD70w5GGIZfRCYV4ehw` **stopped, tasks 0** at 2026-10-27T03:57:07Z, superseding the controller's earlier stopping/tasks 1 snapshot. Both listed apps are stopped with zero tasks, and billed cost is 0. This reviewer did not independently call the service. The selected 1.233633× source remains unchanged.

## Final narrative and offline-analysis verdict

**PASS** for the saved `RESULTS.md`, `COSTS.json` and separate offline interpretation. This verdict does not upgrade the remote diagnostic status. The narrative distinguishes the observed mainloop tuple from unresolved boolean/ISA semantics, Modal identity from judge identity, the valid pre-profile measurements from unexecuted after/oracle/auxiliary/counter stages, and provisional account-level metering from an invoice. Remote-entry elapsed 45.678 s and internal-worker elapsed 43.306 s are now labeled separately. The controller gross-summary delta $0.06162934 and ephemeral-breakdown delta $0.06505812 are both provisional; the $0.70 reserve is not a charge.

The separate `analyze_partial.py` uses same-thread host launch correlation plus matching GPU annotation/External id, reports the observed number of mainloops without an assumed cardinality, retains raw failure status and exposes the original CPU-containment mistakes. Its seven CPU tests passed independently. Results agree with this review's independent direct reading of the raw trace. No raw source or evidence was rewritten.

| Reviewed final artifact | SHA256 |
|---|---|
| `RESULTS.md` | `c3aa73e42c79573708c047f90e7573c7fad027203df8cf74757e2da22583e90e` |
| `COSTS.json` | `5821f96d0e777fb727868f4cbbe397d7234808d8312c9182a7a73f786a854b4b` |
| `partial-analysis.json` | `d3543332729660043ea59e5386573bcb14ba5377be6d5ce3fbbd39507aef0371` |
| `analyze_partial.py` | `d89e2f6dac4d8afd9105b713ce255407df8e9253c330a1e727d8193a220c5ec8` |
| `test_analyze_partial.py` | `bf7fca2825cf2dec40cc3bfaae2d4a1dc19d2e0ee9515548aa3a1a809b893471` |
# Root post-review formatting note

After the independent PASS, the root agent inserted ordinary word/number spaces
in `RESULTS.md`. A programmatic comparison proved all non-whitespace characters
unchanged. Reviewed prior SHA256:
`c3aa73e42c79573708c047f90e7573c7fad027203df8cf74757e2da22583e90e`.
Current SHA256:
`cd328201f459237bfb7524cfb7c7e61580d430e0f487336151091b08b8d14887`.
Both whitespace-normalized documents hash to
`72ee45d2de82d48f9f52fb496753fded076fd61227a3978fcb3fb13a82d70d20`.
No factual assertion, source, result or raw artifact changed in this formatting.
