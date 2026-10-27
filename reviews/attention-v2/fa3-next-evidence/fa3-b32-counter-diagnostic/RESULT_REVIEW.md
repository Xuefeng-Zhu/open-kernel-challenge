# Independent result review: valid collection, failed graph-transfer gate

Reviewer: `/root/fa3_counter_review`. Offline reads only; no service,
authentication, GPU, Git mutation, candidate edit or additional measurement.

Direct standard-library CSV parsing of raw `commands[-1].stdout` found one data
row with 275 columns. The four exact requested columns have percent units and
values 11.47,36.77,43.40,12.49. This agrees with the worker parser; a query list,
kernel-name occurrence or success marker was not treated as a metric sample.
All seven version/help/query/profile commands completed with returncode 0.
The profile command took 19.310275328 s and the worker 61.937419562 s.

All 14 graph checks and three ordered eager-child checks are recorded passed.
Three main plus two child checks actually called the original oracle. The
unprofiled medians are 383.1360042095 µs before activity, 381.2800049782 µs after
activity, and 381.0560107231 µs after NCU. Post-NCU graph oracle and three samples
were reached; none of these times is an instrumented counter duration or a new
optimization score.

The exact identity gate is genuinely false: the graph symbol has 1380 characters
and NCU's has 425. Grid 132/block 256, full fixture, output/new-KV bytes and loaded
wrapper/mapping checks match. Captured installed help documents
`--rename-kernels arg (=1)` and default simplification, consistent with the
observed shorter representation. The raw name gate must remain failed; this
review did not rewrite names or grant a retrospective pass. Unresolved native
split/capture state also remains unknown. Thus production-graph bottleneck
attribution is **inconclusive**, despite valid eager counter collection.

The report's memory-dependency/latency-hiding hypothesis is conditional on the
eager launch. Long-scoreboard wait does not itself identify HBM; aggregate DRAM
and tensor activity do not prove saturation; measured warps do not establish
register pressure as a cause. The proposed next measurement is the separately
help-validated `--rename-kernels 0` identity confirmation, not a kernel change.

The independent receipt at 05:00:58 UTC confirms exact app
`ap-FrbiCpeL0XEh1biFdE7zTF` stopped/tasks 0, all listed apps stopped, billed $0
and provisional account delta $0.05742655. No further GPU retry was made by this
reviewer.

Evidence SHA256:

| Artifact | SHA256 |
|---|---|
| `result.json` | `3abcaeb2b52b57ccd94f8dd56110c4d2908815e8039b2fb1315f53475c71983b` |
| Raw NCU stdout in result | `002113f30f1827eb05268ea69c9688e0f5717d1b622ec96485333618b6c57cb1` |
| Captured installed help stdout | `ac75d8e8b9801ec2b43d18ffa925b9d9e6917ea5ed0c47634af7d67ae1eae7c9` |
| `independent-cleanup.json` | `d2f4903432025a1df893b04f8195d51f28e38ceb46d01eacb043ad63d5f31197` |

Frozen worker SHA256 remains
`9b65cfc93627bc5211b92e96b1f73e3eacbf6099747f05927a6b6cb2806cc056`;
selected source remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The concise report `RESULTS.md` is SHA256
`9d98ccdafd983f5174687836a61617c2ed717a6b8c79bbd6e1373e4b15e87ba7`.
The independently cross-checked offline summary `analysis.json` is SHA256
`e724146b9a48a518fd36ce262e59e97c38b241b965e0d2ade5fcd3c509a10834`.
Its values, check counts, failed identity gate and conditional interpretation
agree with the direct raw-CSV/JSON inspection above. No raw artifact was edited.
