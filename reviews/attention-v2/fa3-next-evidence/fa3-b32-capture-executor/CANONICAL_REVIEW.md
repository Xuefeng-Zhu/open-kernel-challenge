# Independent CPU canonical-export review

**PASS for recovered artifact verification.** Two actual NCU imports of the
retained report produce byte-identical canonical output and exactly reproduce
the original capture's four metric strings/units. This is a separate CPU-only
artifact check. The frozen H200 diagnostic remains failed and unchanged.

Reviewed `OFFLINE_EXPORT.json`, 70,334 bytes, SHA256
`4b7fe7e4511d7d0379c125977930fc2e1a43e7c3aaae6afb2a73e87cb3163b38`.
Both imports use actual NCU 2025.3.1.0/build 36398880 and return 0, with
`--config-file 0 --rename-kernels 0 --print-kernel-base demangled
--print-units base --print-fp --page raw --csv`. They have only an import
target, with no application, replay or capture/export command.

The uploaded report is the same 35,756-byte artifact, SHA256
`4104eaf52a0f44bbfc03703af2355902d16cb1f19bebde4816c3b87a58a3b6b6`.
Its local bytes and recorded before/after CPU-import hashes agree. Both stdout
outputs are **14,985 bytes**, byte-identical, SHA256
`80177532081993b7cbaff5fcc872ae8ce2419aa02a71b889f47999c0c877fdca`.

Independent parsing finds exactly one row in each output. Full native name
(870 characters), grid `[132,1,1]` and block `[256,1,1]` exactly match each other,
the expected manifest and the original capture. For every requested metric,
the raw numeric string, unit and parsed value exactly match the original
capture and both canonical outputs:

| Metric | Raw string | Unit |
| --- | --- | --- |
| Tensor pipe activity / peak sustained elapsed | `11.42` | `%` |
| DRAM throughput / peak sustained elapsed | `36.55` | `%` |
| Long scoreboard stalls / active warp | `43.37` | `%` |
| Active warps / peak sustained active | `12.49` | `%` |

No rounding, tolerance, string rewriting or numeric normalization was applied
in this comparison. NCU itself produced these matching strings under the
explicit output policy. The original default-import six-decimal output is
still preserved. This establishes a reproducible representation of this
retained artifact, not exact underlying physical-counter precision.

The CPU copy was imported at
`/opt/attention-retained-report/baseline.ncu-rep`; the original capture exported
to `/tmp/attention-fa3-b32-capture-executor/baseline.ncu-rep`. Report content is
bound by byte count/SHA256, not a fabricated identical path. The old same-path
gate was not rerun with substituted paths or retroactively passed.
`result.json` remains SHA256
`2dd7a86c3b8d42599ffdd69becc2e4c6e6c4746a692e1f4f11c3faf260a34ca6`,
with `passed=false` and its native acceptance false.

The CPU app `ap-1XKhhpozlSiSFkrtthxHp7` requested no GPU and launched no
profiling target. Root's independent `offline-export-cleanup.json` receipt,
SHA256 `64b574c12f5bec637f14238856cb1b57a0e15f3d952e8c76e07034e234113345`,
records stopped/tasks 0 for it and all apps at 06:21:34 UTC, with paid cost $0.
Account metering remains provisional; no per-app price is inferred here.

Graph ABI/mangled-symbol and internal scheduling-state equivalence remain
unresolved. These percentages have different denominators; this review makes
no speed, whole-call latency-share or bottleneck claim. No new hardware
measurement, candidate change, gate relaxation or further launch follows.
The reviewer used only local reads and wrote these review documents, without
service, GPU, authentication, browser or Git operations.
