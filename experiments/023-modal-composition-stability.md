# 023 — preregistered composition and graph stability comparison

One bounded cached-image Modal H200 app, within the existing cumulative $2 credit
stage and net paid cap $0. No new credentials, settings, OKC check or push.
Budget ledger: experiments/modal-diagnostics-budget.json, experiment M23-composition.

Five same-app runners form an auditable two-factor comparison plus official control:
candidate022 full composition; native_combo021 without pooler layout; pooler_layout
P2-01 without new QKV/unpooler; r10 frozen original best; sglang official baseline.
Pooler cross-run m>1 normalized improvements2.7–7.9% justify testing, not assuming
a win. Compare full/native and P2-01/R10 on each pooler shape to isolate it.
Original R06 regressions and all independent source/results remain preserved.

**Protocol fixed before launch:**

- Full39 native correctness cases per runner, seed offset1000000.
- Native call warmup3 and graph capture as before; then exactly6graph replays per
 runner/shape as declared warmup, timed and saved but excluded from steady metrics.
 Warmup uses distinct fresh seeds (offset+1000000+case seed+round), cold L2, and
 validates every output. Six is selected before this run, following the earlier
 first-two spikes; no post-hoc changes or samples removed.
- Steady30replays per runner per each of30shapes, fresh input seeds, cold-L2 read
 before each CUDA-event interval. Store every sample and validate every output.
- Five cyclic rotations of candidate/native_combo/pooler_layout/r10/sglang plus
 their five reverses, repeated3times. Every runner occupies each position twice
 per block; every order has a reverse. Fifteen reversed-order pairs per shape.
- Summarize gross medians/p10/p90/max/MAD, all warmup samples and paired ratios.
 Weighted load sums use official multiplicities and equal load geometric mean;
 this diagnostic includes replay overhead and is not an official score or CI.
- Driver/clocks/memory/SM-count snapshots recorded outside timing when available;
 no clock/settings change and no inference of GPU-busy time from worker wall time.

Same H200 image, one max/min0/buffer0, CPU2 request/limit, RAM8GiB hard limit,
timeout600/startup120, internal540/wrapper555, retries0,scaledown2, no detach.
Explicit19source/license files plus manifest (adds only021 and P2-01 controls to
the prior17file bundle). No auth/config/repo/cache/other-project files. Compare
both compositions in one app to avoid spending another run on layout ambiguity.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile modal_preflight.py modal_gpu_worker.py modal_diagnostics.py modal_analysis.py finalize_modal_diagnostic.py
.venv/bin/python modal_preflight.py --candidate candidates/projection_full_composition.py --build local-notes/modal-m23-bundle
.venv/bin/python modal_diagnostics.py --experiment M23-composition --bundle local-notes/modal-m23-bundle
.venv/bin/python finalize_modal_diagnostic.py M23-composition
```

Local92unit tests pass, including position-balanced pairing, exact-once sample
use, no outlier deletion, weighted sums and equal-load aggregation. Each composed
source passed39CPU+9focused cases and static/syntax. GPU result pending at source
commit. Follow the exact accepted app to stopped and reconcile costs; never
replace an uncertain request. Report stability and best composition before expansion.
