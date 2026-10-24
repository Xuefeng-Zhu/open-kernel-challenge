# Continue the same task — no new approval needed inside the granted budget

This is a coordinator checkpoint, not completion of the autonomous iteration.
The new budget permits **10 off-board dense-projection checks**, $0; five are admitted
and five remain. Prior baseline e5a5b2d20da54187 is excluded. Scored/final submission,
new grants/credentials, new material agreements and paid compute are not authorized.

## First: follow the two already admitted jobs

```sh
python3 remote_check.py poll R04-router-triton16
python3 remote_check.py poll R05-wide-layout
python3 remote_check.py quota
```

R04 **e3bc1481851f4a23**: `candidates/router_triton_split16.py`, source commit
129969fc2ba10836f25664df8f13f0734d8001dc, SHA256
64aeef36775c050908dd5ddcbd9125a10083ea0d6a2bdbaad704b348debb7725.
R05 **1631810d19af4af4**: `candidates/router_fused_view.py`, source commit
b5f6eddf6f7e6ec59692aaefb2f3600d8f4738a4, SHA256
a7ba4be49f33c4eae5eba30a4b16950fcb867091cb662a9ef1ad57448a47c53c.
Both were confirmed queued at 08:44 UTC. CLI background monitors also read these
same jobs; no request outcome is uncertain. Never POST these experiments again.
If a wait process is unavailable, resume read-only status by the exact accepted ID.
Bounded status calls work; no approval/tool blockage exists.

For each terminal result, run `python3 summarize_check.py <experiment>`, inspect the
actual correctness, all shape times/profiles, interval and error/log if failed. Record
and commit that experiment independently; preserve failures. Then `python3 update_progress.py`
refreshes derived files/index without a network call. Push sanitized commits to the
existing private repository. The admission wrapper enforces check mode, source commit
hashes, 10-request ceiling, at most two active jobs, rules-version guard and no POST retries.
An uncertain POST stays reserved until own jobs/status reconcile it.

## Completed evidence and next hypotheses

- R01 59abe5825690427c: correctness pass, 1.0012943× SGLang, parity. Result commit a05080e.
- R02 89fe00c5034a4198: correctness pass, **1.0476946×** [1.0319792,1.0634100]. Result commit 6e3115e.
  Router m=16/32/128/256 is 9.60/9.82/10.06/11.30 µs versus 14.93/15.28/16.13/17.73.
- R03 718e0730e1bd4c30: correctness pass, **1.0267804×** [1.0113787,1.0421821]. Result commit 34db70b.
  Its m=8 router is **14.54 µs versus SGLang 26.30**; larger-M paths are R01 control.
- R04 isolates a new low-batch fused hi+lo split-K Triton router; larger-M path stays R01.
- R05 changes only R02's packed-weight memory layout. R02 remains best completed source.

Once R04 demonstrates usable Triton compilation/correctness, a separate pooler candidate
is ready: `candidates/pooler_triton_split8.py`, local record 007, source commit 971ade4.
It passed 39 CPU cases + 3 focused model tests, but **no GPU validation**. It changes
only low-batch BF16 pooler atop R01; original weights remain BF16. Check R04's errors
before sending it so shared compiler mistakes are not knowingly repeated.

After pending results, combine only measured winning shape regions (for example R02
large-M wide GEMM with R03 or R04's validated low-M path, and a pooler change only if
its own run wins). Preserve single-file submissions and root baseline; create a new
candidate, CPU/static gates, source commit, then one admitted check. Do not modify
pending sources. Additional tiling/split experiments need an actual profile-driven
hypothesis; do not use all ten merely to exhaust the budget. No scored entry.

All new admissions go through `python3 remote_check.py send <new-ID> candidates/<file>.py`.
Use IDs R06 onward; the wrapper refuses existing IDs. Review results before filling a
free slot, keep at most two active jobs and respect stricter server limits. The shared
queue has caused 13–26 minute waits; do not guess an ETA or launch replacement requests.

Attention review is finished, local-only under `local-notes/attention-review/REVIEW.md`.
Its attachment was never committed/uploaded and must stay separate. The kernel project,
private repo, original baseline and all numerical/provenance records are preserved.
