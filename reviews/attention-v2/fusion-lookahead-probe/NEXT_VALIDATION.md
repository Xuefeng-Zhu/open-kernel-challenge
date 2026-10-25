# Next validation: generic DP lookahead with a real split merge

The current evidence already tests changed GPU lengths and page mappings in
the same captured graph at fixed addresses. Do not repeat that as a supposed
gap. A narrower gap remains: the new lookahead guard also enables the DP head
configuration, but its only native true-lookahead DP case is a 65-token smoke
test with one split. No saved official prerequisite has batch32, so none
exercises true-lookahead DP. A focused multichunk DP robustness check can test
the composition's admitted generic scope before broader deployment. It is not
a new predicted performance improvement or a reason to consume check quota.

This is a local preparation document only. No source, harness, GPU job,
official request, billing setting or credential was changed by this review.
Projection owns the serialized measurement slot. No run is ready or admitted
solely by writing this plan.

## Existing coverage, checked against source

- `runtime/worker.py:243` captures a graph once for each runner, using a clone
  of fixture A. `replay(version)` at lines265–268 calls `helpers.restore` into
  those same allocations before replaying that retained graph. It does not
  capture a new graph for fixture B.
- `../modal_worker.py:56–64` copies every tensor into the existing work tensor,
  including `seq_lens`, `page_table`, `cache_loc`, scales and supplied scheduler
  metadata. Shape and dtype changes are rejected. Configuration remains fixed.
- Worker lines282–284 assert that fixture A/B lengths, QKV and page tables
  actually differ. Lines251–254 compare input and output addresses to capture
  addresses. A→B→A validation and exact return-A checks therefore already cover
  live same-address length/page-table replacement.
- Worker lines37–45 and55–59 show TP8 page64 boundaries through8193 tokens,
  but DP only at ctx65 with lengths `[0,63,64,65]`. Candidate lines303–306
  derive split capacity from page-table shape and8192-token chunks. DP ctx65
  is one split; it never executes the separate merge at lines326–329.
- The candidate guard at line316 uses only B32/L1/full/no-RoPE. It does not
  restrict head count, page size, scale or number of chunks. Official reference
  `HEADS` at line18 maps DP to128 query/16 KV heads, versus TP8's16/2.
- The saved official job's16 prerequisite cases contain no B32 case. Its B32
  scored full-decode case is TP8/page1. Official success therefore does not
  directly fill the DP multichunk lookahead gap.
- Local integer address tests already cover DP-like KV strides, page1/page64,
  multiple chunks and64-bit offsets (`test_source.py:145–193`). Those are useful
  source/CPU evidence, but do not establish native DP partial/merge behavior.

## Smallest useful protocol

Keep the current candidate and fusion control byte-identical. Add a separate
diagnostic harness and fresh fixtures; do not modify the frozen worker to
reinterpret its old run. Use the existing image, oracle, precision, tolerance,
cache-byte, immutability, address and A→B→A helpers.

Two fixed cases are sufficient initially:

| Case | Dimensions | Page/scale/distribution | Purpose |
|---|---|---|---|
| DP multichunk | B32, L1, full, ctx8193, DP128Q/16KV, D64 | page64, scale1/64, soft | Two8192-token chunks, native merge, empty historical splits, page/tile/chunk tails |
| DP multichunk remap | Same dimensions and split capacity | page1, scale1, coherent | Different address arithmetic and official-style coherent context under the same admitted DP scope |

For each case use two new seeds, e.g.135721/135722. Reuse lengths
`[0,1,63,64,65,8191,8192,8193]×4` and its reverse. Rebuild the serving-supplied
FA3 metadata for each fresh fixture outside capture, then restore it and all
other tensor contents into fixed graph allocations between A/B/A replays.
Scatter physical pages, preserve live physical page0 and mask unused table
suffixes negative. Assert only valid history/new rows are addressed.

Require both complete entrypoints to pass the original oracle on A→B→A:
12 oracle calls total for two cases/two arms/three launches. Retain six replay
labels across the two cases, with K and V each compared after every label,
for12 cache-byte
comparisons. Require four exact return-A comparisons, immutable inputs and
untouched rows, fixed addresses, and no NaN/Inf. The true-lookahead guard,
SPLITS2 and separate-merge branch must be confirmed from fixed configuration
and shape; do not infer execution merely from a successful import.

The initial run should be correctness-only. Existing paired timing already
supports selection on scored TP8 cases; DP throughput is not scored. If
nearby-shape latency is later needed for deployability, declare a separate
bounded timing question and retain it as nearby robustness evidence, without
claiming a whole-score gain.

Stop on any source/hash mismatch, oracle/cache/replay failure, access denial,
watchdog expiry, paid-cap/free-credit admission failure or unexpected extra
app. Use zero retries, one serialized app and explicit independent stopped/
tasks0 closure. The existing160s-worker/210s-watchdog and$0.58 reserve envelope
is a conservative upper bound to review again against live rates and free
credits; this document neither reserves nor spends it. No new install,
credential or permission is needed in principle, but a new hash-bound worker
and independent local review are required before launch.

If it passes, the actual gap is closed without changing the candidate. Do not
submit the same source again merely to use one of the two remaining official
checks: the CLI deduplicates identical source, and these custom DP cases are
outside its fixed prerequisite set. If it fails, investigate the precise
failure before editing; a justified fix or narrowly scoped guard would then
be a distinct candidate requiring full correctness and timing review.

## Source binding and selection value

| Artifact | SHA256 |
|---|---|
| Current candidate | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| Current native worker | `03c1261d8cb18b9c2ea266556463d087ee83d2da8cc121e1be80ff04fdc3cb15` |
| Native result reviewed | `5e6447008a1f55baff65939aa2af75521235891fad976434ac378b4dbc55395d` |
| Published official result reviewed | `a66e15b71116d6758cd1aae75138f19c0a7db4cb2e89ba27394d6d09df2d61cb` |
| Existing clone/restore/graph helpers | `9480d32a40872bb5cfe1e38a09eed4256d00acbcb3431d14006417276768745e` |
| Existing official reference | `386f7f67a1d788501e8fb9005d340423d098a0946e23810ba10878228c11a37b` |

Source freeze remains`c9062363c665e37535d3370c1b6bfb54e04fc7c8`. The current
result supports the composition as a modest validated scored candidate. This
additional check would support its generic head/page/scale dispatch scope and
fallback confidence. It cannot prove aggregate statistical superiority or
justify a broad new performance claim. No credible new performance mechanism
was established by this local gap audit.
