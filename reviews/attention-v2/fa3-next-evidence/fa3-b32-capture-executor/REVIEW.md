# Independent offline executor review

**PASS for the implemented artifact-capture executor and CPU integration test.**
This is executable collector/controller integration, not another command plan.
Native NCU report generation, GPU correctness/timing and graph ABI/scheduler
identity have not been validated by this offline work. No GPU, service,
authentication, browser or Git operation was used by this reviewer.

## Reviewed snapshot

| File | SHA256 |
| --- | --- |
| `worker.py` | `d3712de360fe435c9a3f8761aa431af899803cd7ebeb92739b51e0bd1adc9c34` |
| `controller.py` | `46ce63fc7f804f295c9e779e92200929c660d38d02e1906d12004d2b847ea74e` |
| `test_worker.py` | `df778b67896aa8fc386f15268c5cd93f09aefdc4d7476de29659665f269d4606` |
| `test_controller.py` | `cb93ada1488eb995bbfd3f33bca621888e179e62e333a9e51cd564b1de51ed11` |
| `dry_run.py` | `80e84a4be82fcd4f5cce7aa5f5d4b7cf7aabf754bc9770b3af037468538bb55b` |
| `expected-native.json` | `f0ac3f62f101201d15e66534ce62b1534cbceaa71243d97c72c2865ad2170795` |
| `capture_plan.py` | `be067afbc933020f41ea12f103c15263bc0e1c991c4a80ec1a1b6c1b763c26db` |
| `csv_helpers.py` | `a00f901febe900e4ed9f48680b990acf813f0c0d76680b1c798a664f017759df` |

`expected-native.json` independently matches the immutable failed-capture
fixture: exact 870-character NCU available-name spelling, grid `[132,1,1]`,
block `[256,1,1]` and all 15 selected/reference/wrapper/binary/tensor hashes.
It is an expected future target, not a previously captured counter-row proof.

## Integration checks

- The actual worker calls the shared `collect_counter` path with the stable
  family filter, config/rename disabled, explicit report export and target-free
  same-report import. Both caller and collector require at least 100 seconds
  remaining before the bounded 75-second capture. Post-profile original-oracle
  and graph confirmations remain in the native caller.
- Empty/no-row/no-report captures reject even exit 0. Missing or changed report,
  failed import, different native signature/geometry/metric/unit or deficient
  bindings reject. Report, original CSV and imported CSV remain required.
- The actual launch uses `materialize_result`, also exercised by the dry run.
  It retains partial outputs before final acceptance and locally reparses the
  transported CSV. The controller independently derives all runtime bindings,
  loads the frozen expected manifest and recomputes the same native gate.
- Raw graph/NCU name equality remains recorded separately. Artifact capture
  can complete while graph ABI, packing, split, PDL and scheduler-state
  equivalence remain explicitly unresolved. No cast/alias normalization or
  claim of whole-graph counter attribution was introduced.

## Independent CPU execution

All **31 focused tests passed**. An additional independent run of `dry_run.py`
exercised the real collector's subprocess/file handling, actual remote-entry
artifact packaging and actual local materialization. Only the NCU executable,
native worker process and remote path prefixes were substituted.

The success fixture retained all three artifacts and passed same-report import
and local native gating. The actual prior exit-zero empty-capture fixture was
rejected before import. Corrupt report transport was rejected. The saved tiny
counter control was parsed, and old four-metric values were used only in an
explicitly synthetic CSV/signature combination. Synthetic report bytes are
marked `OFFLINE_SIMULATED_REPORT_NOT_NCU` and are not valid Nsight reports.

Independent summary SHA256:
`857f16e5fee7b3f87c20ae5879cb6b638d542e2726e819e59b0bb11221263459`.
It records `new_gpu_apps=0`, `credential_or_service_calls=0` and
`native_ncu_report_validated=false`. The dry run checks post-capture plumbing;
it does not execute the native CUDA prefix, oracle or graph-timing stages.

All **40 files** in the prior recapture's `RESULT_MANIFEST.json` independently
retain their recorded byte counts and hashes. The selected candidate is
unchanged. No new official check, ranked entry or performance result follows.

No implementation blocker was found in this snapshot. Any native admission
still needs its source freeze plus current authorized credit/authentication,
serialized-slot, timeout and cleanup checks. This review adds no launch
authority and does not relabel earlier failed captures as successful.
