# Local candidates — ranked by validation priority, not measured speed

The proven baseline remains `../submission.py`, source commit
`fde8a042944c742c20b76f27fcc030d767fd0058`. Its sole H200 check passed correctness at
0.9076× SGLang. **None of the candidates below has been uploaded, compiled/executed
on CUDA, captured in a CUDA graph or timed on H200.** CPU branches explicitly model
BF16 operands using FP32 products; they do not exercise the CUDA `out_dtype` path.

| Priority | Candidate / isolated hypothesis | Actual local result | What target validation must resolve |
|---|---|---|---|
| 1 | [router_split_after8.py](router_split_after8.py): reproduce SGLang's allowed two-product hi+lo router above 8 tokens | 39/39 full-size CPU cases, 3 focused tests, static checks pass | CUDA BF16→FP32 GEMM/capture correctness, all 30 shape times and six-load aggregate; establish a faithful control for the measured large-M deficit |
| 2 | [router_split_after1.py](router_split_after1.py): change only the dispatch boundary, preserving m=1 FP32 | Same CPU gates pass | Does the m=8 router improve without regression? No small-M threshold is inferred from CPU timing |
| 3 | [router_fused_wide.py](router_fused_wide.py): one wide-N GEMM produces hi/lo halves, then sums in FP32 | Same CPU gates pass | Does one fewer GEMM launch and wider N outweigh extra output traffic/tactic changes? Validate every accuracy case, not only the fastest shape |
| 4 | [router_transpose_view.py](router_transpose_view.py): change only weight layout to transpose views | Same CPU gates pass | Strided CUDA `out_dtype` behavior, internal copies/tactic choice and cold-L2 latency; lower priority because a layout bottleneck is not established |

Each candidate has a separate commit and experiment record. See
[the history index](../experiments/index.json) for exact source/result commits, SHA256s,
configuration, commands and actual CPU evidence. Combined local regression suite:
**17/17 tests pass**. Maximum router relative L2 in each full CPU suite is 2.729e-6
against a 1e-4 limit. No CPU performance ranking or GPU speedup is claimed.

## Smallest next H200 validation plan — requires approval

Approve **one** off-board check of priority 1, then stop and review its result:

```sh
okc problems --json
okc check candidates/router_split_after8.py --no-wait --json
okc status <accepted-job-id> --wait --json
```

Before the POST, verify the committed candidate hash and current quota, confirm the
user's $0 cap and source-upload authorization, then save the returned job ID. Do not
repeat an uncertain POST: inspect jobs/status first. A check uploads the complete
single source file and is not a confidentiality guarantee. This plan itself grants
no permission, and it was **not executed**.

Compare with baseline job `e5a5b2d20da54187`, keeping its complete per-shape and load
data. Judge each new variant by correctness first, then all six weighted loads and
the official confidence interval. Review priority 2/3 only after that control is
established. Each distinct candidate needs its own separately authorized validation;
do not bundle four methods into an opaque submission to circumvent that limit.

No paid rental, scored submission or additional remote check is currently authorized.
