# CPU-only source inventory

`worker.py --output /tmp/fa3-image-inventory-result.json --deadline 30` reads
existing vendor metadata and selected source files. The default deadline is
30 seconds, and the accepted range is `(0,30]`. `collect_inventory()` is the
read-only Python entry point; `main()` additionally hashes the uploaded worker
and atomically writes the designated result. No real image inventory has been
executed during this local preparation.

Discovery uses only five top-level `importlib.util.find_spec` calls (`triton`,
`flashinfer`, `sglang`, `sgl_kernel`, `cutlass`) and metadata versions for
`torch`, `triton`, `sglang`, `sgl-kernel`, `flashinfer`, `flashinfer-python`,
`flashinfer-jit-cache`, and `cutlass-dsl`. Vendor code is parsed as AST/text;
it is never imported or executed. Missing distribution metadata and missing
module locations are reported separately.

Sources must resolve inside their discovered vendor module root or the fixed
`/opt/sglang` attention source subtrees. User/config/cache roots are rejected.
The fixed `/opt/sglang` root must remain canonical `/opt/sglang`; files or
directories escaping an allowroot through a symlink are skipped. The worker
looks at a fixed list of small Python sources and a bounded list of relevant
header names. It never walks an entire filesystem or the package root.

The hard limits are 64 existing files, 8 MiB source bytes read, 2 MiB per file,
4096 UTF-8 bytes per selected excerpt, 64 KiB total excerpts, 24 KiB total
serialized signatures, and 262144 bytes for the complete JSON. Header walking
is limited to depth4, 32 directories and256 entries across all searches.
`flash_ops.abi3.so` is stat-only; its 1.1 GB contents are never read or hashed.
The source byte budget excludes the explicitly uploaded worker's own hash.

The JSON binds `worker_sha256` and includes `gpu_requested=false`,
`cuda_context_requested=false`, `candidate_kernel_launches=0`,
`package_imports_executed=[]`, limits, module/distribution records, file sizes,
source hashes, selected API signatures/lines, missing candidates and limits
reached. `status=complete`/`passed=true` means the bounded scan completed;
it does not establish graph safety, page1 eligibility, precision or speed.
Unavailable source remains unknown rather than evidence that a binary lacks
an implementation.

Local validation: 16 worker tests and8 controller tests pass. All worker tests
use synthetic temporary vendor trees and mocked discovery/metadata. They cover
AST-only inspection, separate module/package absence, exact source fingerprints,
large-binary stat-only behavior, symlink escape rejection, byte/file/walk/export
limits, watchdog propagation and atomic bounded output. They neither inspect
the actual Mac's vendor packages nor start any remote service.

Frozen local worker SHA256:
`5a287e350612a5f10356a5645e50689c51cdce80c175bb8d22f017cae1321f23`.

No GPU, network, subprocess, HOME/config/credential read, install, commit or
remote action is performed by this worker. The separately reviewed controller
owns any later fixed-image CPU admission and resource closure.
