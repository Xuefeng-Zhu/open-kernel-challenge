# Native ownership probe: bounded H200 run completed

A13 ran the exact selected A2-11 source on H200: **42/42 checked replays passed**
across seven reduced-context cases. Actual cross-graph overlap was observed for
full decode B32 only; six other profiled cases did not overlap. See
[A13-RESULT.md](A13-RESULT.md) and [coverage matrix](A13-coverage.csv).
Nine current CPU tests pass. The single Modal run is complete and stopped; its
specific authorization does not permit another launch.

The original default 30-replay full-context command below remains unexecuted.
The old five-test preflight is historical evidence; A13's source, commands,
environment, hashes and nine-test report are separate.

## Historical preparation, before A13


Five CPU tests and a dependency-free dry run pass. Exact environment/source hashes are in NATIVE_OWNERSHIP_PREFLIGHT.json. This is harness preparation, not a new benchmark result.

Default plan: five case families, two independent graph captures each, one A1/B/A2 round = 30 checked replays. Three sliding decode batches, full decode32 and nearby fullB2/page64/scale1/64. Every family varies actual lengths; decode1 therefore differs from its exact fixed-length scored fixture. Two streams are enqueued before waiting; this alone does not prove physical GPU overlap. Never overlap a single graph's shared output/scratch or explicitly share graph memory pools.

Each replay checks official adaptive output bounds, new-cache8% tolerance, finite reference, all other inputs/untouched cache rows, stable addresses and A2 bitwise equality with A1 output/newKV. Input snapshots are distinct for both arms and immutable. Warmup/config preparation happens before overlap. Oracle and snapshot operations finish before the next restore. Default180second deadline has a hard child-process watchdog; partial failures/timeouts remain in a fresh report file. The default30 count excludes warmup calls, which are not correctness-checked by this harness.

Local commands already run:

```sh
.venv/bin/python -m unittest discover -s reviews/attention-v2 -p test_native_ownership_probe.py -v
.venv/bin/python -B reviews/attention-v2/native_ownership_probe.py --list-cases
```

Future native command, ONLY on a separately authorized H200 host with official SGLang/Triton dependencies (not this Mac environment):

```sh
python reviews/attention-v2/native_ownership_probe.py   --candidate reviews/attention-v2/mixed_fa3_decode_parallel_merge.py   --expected-sha c9a0847502a0b2fe2be55f546a2b3ea7eefa931c851e328e680302ebd828baf0   --output /tmp/attention-ownership-new-report.json --rounds 1 --deadline 180
```

This command has NOT run. Existing-output protection prevents overwriting evidence. It neither uploads to OKC nor starts cloud resources; it uses the supplied local GPU only. Do not upload this harness as a challenge candidate. Current phase permits ordinary free official checks but no new Modal launches/credentials/paid resources, and the Mac has no CUDA. Thus bespoke native stress remains unverified; no such GPU request was attempted or denied.
