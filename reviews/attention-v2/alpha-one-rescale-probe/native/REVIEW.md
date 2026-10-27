# Independent alpha-one compile-probe review

**PASS for one source-bound compile/disassembly probe.** No material local
source, scope, transport or admission-control blocker remains. The parent must
freeze these exact bytes, run real frozen-head preflight and satisfy its granted
serialized slot plus fresh free-credit/paid0 controls before launch. This review
performs no service, GPU, network, browser or credential action. It does not
claim a successful compile, strict native equivalence, useful frequency or speedup.
Only this review document was written.

## Isolated source and exact predicate

The control is the selected working composition, SHA256
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`;
prepared candidate is
`e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764`.
Direct source diff and six offline tests confirm that only one historical-loop
numerator update is wrapped under existing constexpr `PAGE_LOOKAHEAD`.
All other functions, dispatch, scales/precision, denominator update and new-KV
last-split tail are byte-identical to control. Replacing the inserted block with
the old assignment reconstructs the entire control source exactly. The false
path itself contains the exact original rescale expression.

The predicate is `sum((row_ok & (alpha != 1.0)).int32)==0`, using the already
computed FP32 alpha. Invalid query rows cannot veto or admit the valid-row fast
path. There is no epsilon, approximate threshold, deferred maximum or row-max
equality substitute. For the target L1/G8/BLOCK_M16 geometry, eight rows are
valid and eight padded. Valid NaN/non-one alpha takes the old expression. The
first ordinary finite historical tile has alpha0 and also takes the old path.
The denominator, logits/exponentials/P conversion, K/V lookahead and last-split
new-KV ownership remain unchanged.

Finite multiplication by positive one has an exact scalar identity basis,
but it does not prove native operation order or flush-mode equivalence. The
archived working producer feeds `acc*alpha` as PV MMA's C operand. The fast path
must preserve old carry as C; PV from zero followed by an added carry would be
a different rounding order. Native control-flow/dataflow, FTZ, buffering and
resource inspection therefore remain mandatory. This probe cannot promote the
candidate or bypass the later original-oracle/replay/cache gates.

## Exactly four compile-only units

Worker source contains only `_fused.warmup` and `_merge.warmup`, followed by
module-handle initialization, for two arms and two units. There is no bracketed
kernel launch, `kernel_fn`, graph replay, timing loop, profiler, counter or
instrumented algorithm. Typed one-element allocator pointers provide dtype
and16-byte alignment only; they are deliberately not valid execution fixtures.
The reviewed path never executes them as candidate kernels.

Both arms use the same target specialization: B32/L1 TP8,16Q/2KV, G8,D64,
BLOCK_M16/BLOCK_N64/BLOCK_L16,8192-token chunks,SPLITS11, full/no-RoPE,
separate merge and PAGE_LOOKAHEAD true, four warps and three stages. Producer
grid is(32,2,11); unchanged merge grid(32,16), BS16. QKV stride2304 and
page-table stride86001 match the scored-family shape bound. Pointer dtypes
match the current source signature, including int64 new locations, E4M3 cache,
BF16 QKV/output and FP32 partials/scales. Compile arguments retain BF16 PV,
E4M3 QK and the exact source's `max_num_imprecise_acc=0`.

Each unit must return TTIR,TTGIR,LLIR,PTX,CUBIN and SASS:24 files total. Driver
metadata records registers, raw spills and shared bytes without claiming achieved
occupancy. Static PTX instruction-family counts are explicitly textual; they
cannot establish dynamic skip frequency or a bottleneck. Only an already
installed `cuobjdump` is used, with10s command timeouts; missing/failing tools
or oversized artifacts produce partial/failed evidence. No installation or
NCU invocation is present. PAGE_LOOKAHEAD-false specializations are not compiled
in this first bounded probe; their local source identity is a separate claim.

## Transport and execution bounds

The worker enforces90s including compilation/artifact work, a4MiB per-file cap,
an8MiB uncompressed JSON archive cap, a24-flat-name allowlist and a2MiB entire
report limit. The archive body contains base64 file bytes, so its8MiB JSON cap
is conservative relative to the decoded file total. It uses deterministic gzip
and whole-bundle/per-file SHA256 and size metadata. Partial artifacts remain
bounded diagnostic evidence, not successful completion.

The controller independently rejects invalid/duplicate JSON keys, malformed
base64, wrong encoding/hash/size, oversized files, manifest/content mismatch,
unknown names, traversal, directories and alternate suffixes. A single bounded
zlib decompression call allows at most8MiB plus one detection byte; EOF,
unconsumed-tail and unused-data checks reject bombs, truncation, trailing data
and concatenated gzip members. No unlimited decompression/flush or tar/zip
extraction occurs. Optional compressed size, if present, must match actual bytes.
The complete bundle is validated before fresh-directory, exclusive flat writes;
existing artifact destinations cannot be overwritten. Compilation completion
requires exactly all24 artifacts and all four unit records with frozen hashes,
compile-only protocol and integer kernel_launches0. This completion is not a
semantic/performance/official admission.

The five-file uploaded allowlist is worker, two exact sources, license and
third-party notices. The controller is hash-checked locally but not uploaded.
The remote entry is closure-free; local receipts/configuration/credentials do
not enter its payload. Source imports use only the frozen reviewed modules and
existing image, with network disabled and Modal access restricted remotely.

Local explicit launch validates committed HEAD, fresh exact parent slot and
paid-cap0/free-credit receipt, existing protected credential through supported
config/profile paths, supported live actual-paid0, stopped/tasks0 app inventory
and live resource rates. There is no credential renewal, environment expansion,
billing change or removed artificial stage cap. Reserve is at most$0.40 and
must cover double the full140s live rate plus$0.02. Existing image is
`im-cPTbGpNiX3RarKResPgX7h`, one H200,2 CPUs,8GiB host memory, one invocation,
one container and zero retries. Bounds are90s worker,100s child,115s function,
20s startup and140s independent watchdog.

An exclusively created ledger/result prevents duplicate launch. Raw response
SHA256 and size are verified before artifact decoding. Worker partial reports
are retained. Normal and exceptional paths request app stop and independently
record app state/billing. A stopping/nonzero-task observation remains unclosed
and requires separate reconciliation; the receipt is not rewritten to imply
closure. This review supplies no actual runtime state or cost observation.

## Interpretation and offline verification

The predeclared `ADMISSION.md` requires real skipped carry work, unchanged PV
C/ordering and denominator/tail behavior, preserved buffering, no new spills or
local-memory accesses, no increased producer register count and no reduced
static residency resource bound. Raw generated artifacts, not compilation
completion, determine those gates. Success may justify preparing a separate
frequency diagnostic; numerical correctness and whole-call benefit remain
later independent gates. The working composition stays selected throughout
this compile-only stage.

This reviewer independently ran **6 source +10 worker +12 controller offline
tests, all passing**. These include exact patch scope/padding/scalar IEEE models,
four compile units/no execution calls, source/license tamper rejection, bounded
partial failure, deterministic binary/text artifact packaging, transport/hash/
size/framing/traversal/bomb rejection, paid0/free-credit/rate/slot guards and
mocked committed preflight. In addition, the actual worker bundler and actual
controller decoder round-tripped all24 synthetic text/binary files exactly on
CPU. No worker main, GPU package, credential or service was invoked by that
integration check. The real committed preflight remains a parent step.

## Reviewed hashes

All paths are relative to this probe.

| Artifact | SHA256 |
|---|---|
| `baseline.py` | `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8` |
| `candidate.py` | `e66f502812f10f070170efcc9a41a7ef566576799d7461d27cd860a10c94f764` |
| `test_source.py` | `69f38480aeb1da05b972f4eec6dcf5ffb07b4f43b15907cda0b143dca3d80862` |
| `SOURCE_AUDIT.md` | `e9117ed4d979c49b5a1b6137f5810a5763bb985832cd1b73e6047740def92bf0` |
| `SEMANTICS_REVIEW.md` | `86cb3e696b6767f138d3b2334d45ec7a79d51762cd61d091e4144463f7ae344b` |
| `EXPERIMENT_CARD.md` | `040237d21771107fdb448e7ab2f27a117694360aa348685fca25caf1e6b18f84` |
| `native/ADMISSION.md` | `cc98e237028eeb5b4fa3ae6f548f8996f1a0042f206a03d82b3e3cbb42ed5b27` |
| `native/worker.py` | `9e0377dc696518225b5091f65fe7f72ac5e96e6c2b539c399f13bf337a80a585` |
| `native/test_worker.py` | `9a07324f07d790c1ffa92cc217a5af89f668927b0dc996c50dd85fab9bfd1956` |
| `native/controller.py` | `2ddfab210d7a589826674990a380d657a078b794ac73e175fb0fc5d27e9978bf` |
| `native/test_controller.py` | `2fbbdf9018bbe5987801380b8409a70ea36b336178e38c518e0858b2a33a8c54` |
| `native/CONTROLLER.md` | `811e39ccf5a021acfad65d34119a74955aa62588e1ab9ab6ae923ecb648dfb5d` |

No material blocker remains for freezing and live admission of this exact
compile-only payload under the parent's serialized slot. Native mechanism,
resource and numerical conclusions remain pending actual artifacts; no
performance or formal-entry action follows from this review.
