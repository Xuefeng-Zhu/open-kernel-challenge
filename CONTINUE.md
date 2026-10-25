# Projection P038 native check active — primary coordinator only

Worktree branch projection-pipeline-preflight. Preserved best remains
projection-bf16-local0eabf0e, sourcec34c497b, official1.1486499523×
job48b1357016134e52. Read experiments/038-pooler-m256-pipeline-local.md.

P038 TMA and identical-geometry control committed60a8b4bdcdd4f3c48d2b7a8a0db4907c394ba84a.
TMA SHA443128591305c4a8f79f2e8669d59b3cc402a5243b9920273676583b825f518d;
copy SHA1b6fe24df633fb8e01e69cfa0cc3d78992af6c8deae033f6b08093c668d457cb.
Only aligned dense pooler m256 changes; other paths preserve best. Each39CPU
oracles,117regressions,5focused and2extra coordinator tests pass. Source API is
pinned Triton3.8 host TensorDescriptor, no allocator mutation/manual barriers.
Actual native compile/correctness/capture/performance remain unverified until
accepted official check6799753227ed463e reaches terminal. No other GPU active.

ExperimentP2-03-pooler-m256-tma accepted00:23:24UTC; latestqueued00:27UTC.
Projection user allowance7remaining, attention1remaining (separate). No new
request until this exact job terminal; copy control only if meaningful gain or
concrete diagnosis warrants it. Primary task owns the shared GPU slot; subagents
cannot launch. Never duplicate unknown/uncertain jobs. Poll with existing primary
Python/CLI, using OKC_CLI=<primary-workspace>/bin/okc and
OKC_GPU_SLOT=<primary-workspace>/local-notes/gpu-slot.local.json:
<primary-workspace>/.venv/bin/python cycle2_check.py poll P2-03-pooler-m256-tma

Attention E5 already terminal/pass0.8919466283×,12.07% above preserved E2. Its
source/results stay on primary attention-fa3-local branch; not merged here.
No GPU overlap. No push/ranked entry/new credentials/payment/cap changes or paid
compute. Modalguard$1.708917914/$2 remains cumulative; this worktree's$1.146 ledger
is historical. Prior A09 BlobCreate401 unresolved; no app restart or auth renewal.
