# Branch integration — 2026-10-27

All six pending remote branches are incorporated into `main`, preserving their
commits, evidence and branch names. No kernel or official oracle was edited.

| Branch | Integration |
|---|---|
| `projection-pipeline-preflight` | P046, diagnostics and complete rejected P070 evidence |
| `attention-fusion-lookahead` | Selected fusion/lookahead source, ranked receipt and later rejected probes |
| `attention-fa3-sanitized` | Already an ancestor of fusion/lookahead; included without duplicate merge |
| `attention-fused-preparation` | Already an ancestor of fusion/lookahead; included without duplicate merge |
| `attention-sliding-unsplit` | Archived experiment and evidence; not selected |
| `attention-architecture-probe` | Archived architecture/profiling evidence; not selected |

Only shared progress, authorization-snapshot and third-party-notice files
conflicted. Notices from both tracks are retained; dated handoffs remain in Git
history, with current entrypoints clearly labeled. README and the manifest now
identify P046 and Attention fusion/lookahead, distinguish same-source ranked
receipts from earlier off-board results, and preserve rollback sources. The root
`submission.py` remains the original Projection starter. No failed experiment
was promoted. Historical authorization files do not grant new permissions.

Validation: **718 root CPU/mock tests + 41 focused Attention tests passed**;
all **466 tracked Python files** parsed. Source-hash checks confirm both champions
and the starter are byte-identical. All six remote tips are ancestors of the
integration. No GPU run, official check, credential action or billing call occurred.

The initial 716-test run had 14 errors from two stale test contexts. P041's
historical tests now use isolated period-correct P039 inputs, while its live
selection guard is unchanged and negatively tested. P068's exact notices digest
was refreshed for the combined notices; all computational hashes remain frozen,
and a changed-notices negative test still rejects. Historical native receipts
retain their original source hashes; no native rerun is claimed for this tooling
update. Both repairs received independent review.

[Machine-readable validation and source hashes](../results/branch-integration-20261027.json).
Existing scored results are [Projection](../experiments/069-p046-ranked-submission.md)
and [Attention](../reviews/attention-v2/fa3-next-evidence/selected-ranked-submission/RESULTS.md).
Repository visibility is public, as separately requested. Private-only legacy
push helpers remain unchanged and will refuse that target; this merge uses an
explicit ordinary fast-forward push to `main`, with no force or branch deletion.
