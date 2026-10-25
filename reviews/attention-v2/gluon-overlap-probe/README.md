# Gluon overlap: local compiler-mechanism preparation

The matched kernel and fixed configuration are in `kernel.py` and `config.py`.
Read `EXPERIMENT_CARD.md` for the exact math scope and predeclared mechanism,
ownership and resource gates. The existing working attention candidate is not
changed or routed through this standalone experiment.

The modular worker is `native/worker.py`; `native/resources.py` handles actual
CUDA metadata queries and conservative resource classification. Default local
operation performs source/config preflight only:

```sh
python3 -B reviews/attention-v2/gluon-overlap-probe/native/worker.py
python3 -B reviews/attention-v2/gluon-overlap-probe/native/test_worker.py
```

These commands import no Torch/Gluon/driver code and execute no GPU work. Native
compilation requires the explicit flag:

```sh
python worker.py --compile --output /tmp/gluon-result.json --deadline 90 \
  --artifact-dir /tmp/gluon-compile-artifacts
```

The parent must independently review/freeze all source and obtain a serialized
slot through a separately reviewed controller before that native command.
The worker contains no remote controller or launch action. There are
exactly two warmups, with the same arguments/options except `OVERLAP`.
Deliberately undersized one-element pointers make kernel invocation forbidden.

The result schema uses `status: compiled`, `compile_complete`,
`source_sha256` for kernel/config/resources, `worker_sha256`,
`protocol.compile_only`, `kernel_launches: 0` and two kernel rows with
`arm: serial|overlap`, `unit: core`. Compilation completion is distinct from
`resource_gate` and `mechanism_status`. Resource pass leaves machine-code
dataflow/ownership review pending; it never admits execution or performance.

Artifacts are named `serial.core.<format>` and `overlap.core.<format>` for
TTIR/TTGIR/LLIR/PTX/CUBIN/SASS. TTGIR/PTX/CUBIN are required compiler outputs;
TTIR/LLIR may be absent in a Gluon pipeline. Missing or unreadable SASS means
inconclusive. The existing gzip+base64 bundle schema carries compressed/raw and
per-file hashes/sizes, with exact filename allowlisting and bounded sizes.

Resource queries use only existing CUDA headers/driver APIs after warmup/module
load. Unknown headers/enums/API/resource metadata produce inconclusive evidence,
not assumed limits. The CUDA occupancy API reports a legal compiled maximum
CTA/SM, not an observed CTA trace. Static instruction counts similarly provide
an inventory, not a causal bottleneck or a next-tile ownership proof.

No numerical, full-call performance, speedup, formal-entry or serving claim is
made by this local preparation. No kernel, GPU service, profiler, package
install, credential operation or push was performed by its author.
