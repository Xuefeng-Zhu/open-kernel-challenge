# CPU-only installed FA3 identity worker

Local preparation only. No service, authentication, upload, GPU, official check
or Git action was performed. The actual Modal image has not been inspected by
this worker yet; local smoke success is not a remote metadata result.

The worker uses only the Python standard library. It never imports Torch,
SGLang, sgl_kernel, Triton or Modal, loads a library, executes a subprocess,
contacts a network endpoint, requests a GPU or creates a CUDA context. Top-level
module specs and distribution metadata locate allowed vendor installations
without executing their package initialization. Private/config/cache paths and
symlink escapes are rejected. Source reads and complete binary hashes check
stable file identity before and after reading.

It can establish installed wrapper source/API AST, complete extension file
SHA256 and GNU ELF build IDs, plus bounded source/registration-text matches.
The pure Python ELF reader handles 32/64-bit little/big-endian PT_NOTE records;
missing/oversize/unmatched metadata stays inconclusive. GNU build ID is a file
identity value, not a source revision or proof of an executed specialization.
The search is deliberately bounded, not proof that headers do not exist anywhere.

It **cannot establish** a runtime operator schema without loading the extension,
resolved PackGQA/split counts, executed tile, internal QK/PV instruction types or
an official judge binary identity. Report fields retain UNKNOWN. Source matches
are raw evidence, not automatically assigned meanings for template booleans or
auxiliary dimensions. No native control flag is changed.

Run locally from repository root:

```
python -B -m unittest discover -s reviews/attention-v2/fa3-next-evidence/dispatch-identity-cpu -p 'test_*.py' -q
python -B reviews/attention-v2/fa3-next-evidence/dispatch-identity-cpu/worker.py --deadline 45 --output /tmp/dispatch-identity-cpu.json
```

A remote CPU-only payload must supply unchanged selected and reference files
using `--selected-source PATH --reference-source PATH`; their exact full hashes
are checked before reporting. No candidate or reference module is imported.
Source bindings are selected `3da22a...` and original reference `386f7f...`.

Limits: 45-second alarm/deadline, 256 KiB final JSON, 16 KiB combined excerpts,
128 MiB source/ELF-metadata bytes, 8 MiB per source, 96 source files,
128 directories, 2048 entries, depth5. Complete extension hashes are a separate
streaming exception: at most1.5 GiB total,1 MiB chunks,16 files, under the same
deadline. No partial binary digest is published as a complete hash. Oversize
final output fails without truncating evidence. Only targeted header names and
fixed installation roots are searched; no arbitrary filesystem walk.

Thirteen CPU tests pass. They cover all four ELF class/endian combinations,
malformed offsets/notes, API return-only versus forwarded flags, private paths,
binary/source limits, symlink escapes, bounded directory enumeration and the
absence of vendor imports/subprocess/dynamic execution. Deadline exceptions must
propagate through every read/walk handler and the final completion check;
regression tests verify that no expired audit reports completion. Distribution
versions are parsed from guarded, byte-budgeted METADATA files, avoiding the
unbounded distribution.version convenience accessor. A real saved wrapper
was also parsed:37 native arguments, scheduler/split/pack at31/32/33, and
`return_softmax_lse` absent from the native arguments. Actual local smoke passed
source bindings and reported zero GPU/kernel launches. Neither check executes
FA3 or resolves its dispatch. Independent review and normal authenticated,
free-credit/$0-paid admission remain necessary before remote use.
