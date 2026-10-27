# Independent native result review

**Real counter capture and report import succeeded. The frozen diagnostic's
overall acceptance remains FAILED.** The sole failed substantive gate is exact
metric-value equality between the two-decimal capture CSV and six-decimal
same-report import CSV. No tolerance change or posthoc pass was applied.

App `ap-iWkGTqu50RkaYoIsbXg8aq` used frozen commit
`c1208f41841a560a08726a4c95b94e7a44dcf4f3`. This reviewer used local files only:
no services, GPU, authentication, browser, Git or implementation changes.

## Native evidence verified

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `result.json` | 182695 | `2dd7a86c3b8d42599ffdd69becc2e4c6e6c4746a692e1f4f11c3faf260a34ca6` |
| `artifacts/baseline.ncu-rep` | 35756 | `4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6` |
| `artifacts/profile-original.csv` | 19879 | `aed2fe380dd1293d3fcd54ec5c8bf5dc4bf007895c2d13e7987c6578c95115d8` |
| `artifacts/profile-imported.csv` | 14387 | `8ffaa246a5b98a7a6eff8b9d42e8447e21fcfa625ce92e1297e70383385003e8` |

All three durable files are regular, nonsymlink and nonempty. Their byte counts
and hashes match both transport and worker manifests. The retained binary has
the `NVR` report header, not the CPU simulation marker, and actual NCU imported
that exact report successfully with exit 0. Report size/SHA256 are unchanged
before and after import. Compressed and decompressed graph-trace hashes match.

Actual runtime NCU is **2025.3.1.0, build 36398880**. Its captured help SHA256 is
`ac75d8e8b9801ec2b43d18ffa925b9d9e6917ea5ed0c47634af7d67ae1eae7c9`.
All four metric-support queries completed successfully, and capture/import
flags are advertised. Capture used the family regex with kernel replay,
config/rename disabled and explicit export. Import has no application target.
Capture took 25.223 seconds; target-free import took 0.441 seconds. Neither
duration is an uninstrumented kernel-performance result.

Both actual CSV files contain exactly one data row with the exact expected
870-character native signature, grid `[132,1,1]`, block `[256,1,1]`. Independently
derived runtime bindings match all 15 frozen selected/reference/wrapper/binary/
fixture hashes. Main/child full fingerprints, output/new-KV byte hashes and
wrapper/mapped-binary facts also match. All 14 main checks and three ordered
child checks pass, including five unchanged original-oracle calls. All 15 local
payload hashes match the source-bound admission ledger.

## Exact acceptance failure

| Requested metric | Capture CSV (%) | Same-report import (%) |
| --- | ---: | ---: |
| Tensor pipe activity / peak sustained elapsed | 11.42 | 11.418343 |
| DRAM throughput / peak sustained elapsed | 36.55 | 36.551689 |
| Long scoreboard stalls / active warp | 43.37 | 43.368810 |
| Active warps / peak sustained active | 12.49 | 12.494386 |

Every imported number rounds to its capture CSV value at two decimals. That is
consistent with presentation precision; it does not pass the predeclared exact
numeric-equality rule. Units, names, launch geometry and report hash agree.
Independently recomputing the native gate reproduces the stored dictionary and
its false result. Local controller validation also returns false. Worker exit
is 1 and `completion_verified=false`.

The actual failure occurred after successful import at the roundtrip/native
gate in `collect_counter`, rather than in the original oracle. The stored
phase label still names the earlier post-counter graph confirmation. No
permission-denial message or timeout was observed. The shared gate's hardcoded
`execution_state="plan_only_not_executed"` is stale metadata; the native app,
actual binary and successful import establish that this was executed.

## Limits and closure

Graph/NCU raw name equality remains false; graph ABI/mangled-symbol identity
and internal packing/split/PDL/scheduler equivalence remain unresolved. The four
percentages use different denominators and cannot be added or interpreted as
whole-call latency shares. This review makes no speed or bottleneck claim.

The controller ledger reports stopped/tasks 0 for this app and all listed apps,
paid spending $0 and provisional gross account delta $0.08. These account
readings are not a final per-app invoice; root owns separate service-side
cleanup and cost verification. All 40 prior recapture manifest files and all
11 preexisting executor-manifest files retain their recorded bytes/hashes,
including the CPU dry-run evidence. Selected candidate/source is unchanged.

The retained report now permits an offline precision/export investigation;
another GPU run is unnecessary for that question. This review performs no
re-export, modifies no gate, and adds no launch authority. The frozen failure
must remain preserved while any later analysis uses separately stated rules.

Cleanup confirmation added: `independent-cleanup.json`, SHA256
`ec913447f36430b7a02d52eee97af529e35b514f2505232542cbc5b403db1642`,
records root's separate 06:09:40 UTC verification that the native app and all
listed apps were stopped/tasks 0, with billed cost $0. This local review read
that receipt without repeating any service operation. The later CPU-only
canonical export is reviewed separately in `CANONICAL_REVIEW.md`; it does not
change this frozen failure or its original CSV comparison.
