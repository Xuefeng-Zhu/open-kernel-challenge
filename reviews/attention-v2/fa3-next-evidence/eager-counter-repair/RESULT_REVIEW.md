# Independent result review: PASS, capability only

Reviewer: `/root/counter_repair_review`. Read-only local evidence/source checks
and writing these two result documents only. No service, GPU, credential, browser
or Git action, no frozen-source edit and no additional profiler invocation.

The raw report is complete/passed. Version, actual installed help, fixed metric
query and NCU profiling commands all exited 0. The installed version is 2025.3.1.0;
the 34,543-byte help and its digest were saved before query/profiling. Corrected
argv contains no `--units`, while retaining the one queried metric, kernel replay,
application-only scope and skip1/count1 unique-kernel selection.

I independently parsed the NCU stdout with Python's CSV reader. It contains one
actual data row, ID 0, Kernel Name `okc_counter_gate`. The exact
`sm__cycles_elapsed.avg` column is 259 of 266; its separate unit row says `cycle`
and its data value is `5,541.35`. This agrees with the worker's 5541.35 value.
Automatic device/launch/derived columns are not separate requested experiments.
This is a real result row, not the metric query text or a child success marker.
Connection and disconnection messages are present; pre/measured/post pass
markers appear in order and are all true. The 6.086596 s command and 7.149576 s
worker completed without timeout or explicit permission error.

The independent cleanup receipt at 04:38:24 UTC names app
`ap-utcDysQOu5e1m3MSYhEZR4`, stopped/tasks0, and all listed apps stopped/tasks0.
Paid spending is 0. The observed account delta 0 is explicitly qualified as possibly
delayed, not a claim that the run cost zero or a final per-app invoice.

Current selected and rejected source hashes match the result document. Their
`kernel_fn`, `_fa3_full_verify_fused`, `_fa3_sliding_verify_fused`, `_num_splits`
and `_merge` functions are pairwise AST-identical. The rejected source's new
producer predicate requires length 1; full B32/L1 reaches its changed `_e2` producer,
whereas full B32/L8 reaches the same FA3 helper in both files. The next-comparison
geometry warning is therefore supported by current source, not inferred from
an old report. No cross-L1/L8 latency comparison or duplicated L8 arms is endorsed.

The result establishes this tiny eager/one-metric collection only, not a real
Attention bottleneck, every counter, graph replay support or an optimization win.
No candidate or ranked-entry change is claimed.

| Evidence | SHA256 |
|---|---|
| RESULTS.md | `60e0598bcea71b987136ae83dbad92ec7486a2054baa8862f28de79ac71b1127` |
| result.json | `446ca92bf73712ea0514005d03683d910e031d89d7837ddab734523edbba3555` |
| independent-cleanup.json | `984ef87236b263d978086d34e8b82cde43898dd9e8bdf40bff56ef261ac1515a` |
| worker.py | `7391280b4577a2d1f242bdffdc42777a26a180b1cdcd6e021dc47c2866c1137a` |
| controller.py | `af759a477f7916aa48b0e50fb221aaa41c449b054dcb905d45e22add531f90a5` |
