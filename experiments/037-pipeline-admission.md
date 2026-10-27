# P037: projection pipeline research, no GPU admission

Parent/source and prior failures: PROJECTION_PIPELINE_PREFLIGHT.md.
Only one structural mechanism retained for future prototype work: pooler m256
TMA/conventional-copy controlled BF16 pipeline. No new kernel or GPU result.
Offline projection_admission.py reconstructs official1.1486499523× and quantifies
minimum single-shape effect:27.01% latency reduction for1.5% aggregate. Smaller
10% latency reduction yields0.5373% aggregate, below ordinary run uncertainty.

Decision: no speculative official run or repeated failed tile/layout. Eight user
checks remain. Existing candidate, tests, receipts and failures preserved. Primary
attention coordinator owns GPU scheduling; no launch from this isolated worktree.
