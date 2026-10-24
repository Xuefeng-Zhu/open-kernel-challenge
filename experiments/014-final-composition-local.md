# 014 — final composition of independently measured winning regions

Compose R08's small-router split32, R02/R07's contiguous wide large-router GEMM,
and R09's m==1 SIMT pooler. These regions were each H200-correctness checked and
measured separately. Relative to R07, exactly two supported regions change: router
m<=8 split count16→32, and pooler m==1 uses R09. No new tile/warp/precision parameter
is tuned, no regressing R06 pooler path is included, and no speculative QKV path.
The sole hypothesis is that the independent gains survive composition/capture/state
layout in one file. Do not add component speedups to predict its overall score.

Source `candidates/router_pooler_hybrid.py`; SHA256 `891b2063edde9ef9c508c2983e6cf10804bf5066fc036d63912c47ddeb8fc9ee`.
Controls R07 `26ea99ee0869482a`, R08 `ab74f25caa2e46a8`, R09 `97aa5690ff134369`,
R02 `89fe00c5034a4198`. Exact native kernel ASTs `_partial/_reduce` match R08 and
`_gemv` matches R09. The wide branch retains R02's contiguous[K,2N]BF16 weights,
FP32 GEMM output and FP32 half-sum. All other BF16 paths remain original starter.

```sh
.venv/bin/python check.py --submission candidates/router_pooler_hybrid.py --device cpu --suite all --output results/router-final-hybrid-cpu.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python static_check.py --submission candidates/router_pooler_hybrid.py
.venv/bin/python -m py_compile candidates/router_pooler_hybrid.py tests/test_pooler_candidate.py
```

Actual result **39/39CPU full-size contract cases**, **57/57combined regression tests**,
static/syntax checks pass. Seven focused tests cover this combined candidate:4router,
3pooler; full-width fresh one-row probes, irregular tails, original weights and
m1/2/7/8/9/16/32/128/256 boundaries. Environment macOS M2 arm64,Python3.12.13,
PyTorch2.14.1,noCUDA,4threads full-size suite. These models do not compile/run native
kernels. **This combined source has no GPU result yet**, despite prior component runs.

Exact offline native-body identity check performed:

```sh
python3 - <<'CHECK'
import ast
from pathlib import Path
def bodies(path):
    return {x.name: ast.dump(x, include_attributes=False)
            for x in ast.walk(ast.parse(Path(path).read_text()))
            if isinstance(x, ast.FunctionDef)}
m = bodies('candidates/router_pooler_hybrid.py')
r = bodies('candidates/router_triton_split32.py')
p = bodies('candidates/pooler_simt_row.py')
assert all(m[n] == r[n] for n in ['_partial', '_reduce'])
assert m['_gemv'] == p['_gemv']
CHECK
```

Use the final remaining approved off-board check only after this source is committed.
No publishing/scored/final entry or paid compute; root submission.py remains the
unaltered original baseline. Retain all components and unsuccessful experiments.
