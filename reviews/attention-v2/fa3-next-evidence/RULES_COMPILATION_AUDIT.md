# Compilation-route rules audit

Audit date2026-10-26UTC; local TASK/AGENTS read00:03:41.446297UTC.
Public primary rules searched during00:03–00:07UTC. No CUDA compilation, GPU,
install, official job, authentication action, checker edit or rule acceptance.
This report separates documented eligibility from actual server acceptance.

## Concrete eligibility decision

| Route | Rules eligibility | Remaining execution gate |
|---|---|---|
| Native CUDA embedded in the one Python submission, using PyTorch `load_inline`/nvcc | **YES**, explicitly documented by the official rules' indexed text | Current full-body/version and precise compile exception were not independently recovered; actual server checker acceptance and image dependencies remain untested |
| Image-provided CuTe DSL, CUTLASS or TileLang | **YES** as permitted implementation dependencies in global rules | Presence/API/headers in the actual image, track precision, single-file packaging and checker acceptance must still be established; permission to use a library does not prove it is installed |
| Arbitrary explicit `subprocess`/shell/nvcc wrapper in candidate code | **UNCERTAIN** for a tightly scoped compiler invocation; **NO** for general process execution | Do not generalize `load_inline` guidance into a blanket subprocess exemption; exact exception/checker behavior not recovered |
| Processes/network/harness access during the timed attention computation | **NO** under the saved Attention TASK and graph/fresh-input rules | No compile exception should be read as permission to perform unrelated work or modify the harness |
| Untimed weight-dependent `prepare` on Attention | **NO such track interface is documented** | The global exception for `muse_attention_decode` concerns weights/prepare, not a native-CUDA ban |

Thus the previous interpretation that the Attention TASK's short process ban
necessarily prohibits all legitimate extension compilation is overbroad.
Conversely, this audit does not certify an arbitrary custom compiler wrapper,
change a checker, or admit a GPU experiment. Expansion to a new native-CUDA route remains unadmitted by this audit. The
parent separately clarified that existing Triton/Gluon probes remain authorized
and must not be blocked by this new eligibility question.

## Primary public source: what was actually retrieved

Source: [official global rules](https://openkernelchallenge.dev/rules).
The public web search index returned official-domain snippets marked crawled
today, with an approximately18-hour freshness annotation. Direct web open
returned the page title but zero body lines; the public API
`https://openkernelchallenge.dev/api/v1/rules` was inaccessible through that
extractor. Therefore a fresh complete page-byte hash and live rules version
are **unavailable in this audit**, rather than assumed unchanged.

Two short supporting excerpts from the official indexed rules are
`PyTorch's load_inline` and “Compilation time counts toward the run's time
limit but not your score.” The preceding native-CUDA sentence specifically
instructs embedding its compilation inside the file and identifies nvcc as the
compiler behind that PyTorch facility. A separate snippet permits building on
libraries supplied in the image and names CuTe DSL, CUTLASS and TileLang among
them. These establish the supported routes above. They do not expose a complete
static-checker implementation or every condition of its compilation exception.

Another indexed sentence says all problems except Attention have weights and
receive an untimed prepare call. Its exception is about that interface; using
it to prohibit Attention native CUDA would misread its subject. The exact
claimed pre-scoring process-exception sentence was **not independently obtained**.

A browser fallback was attempted once for a separate public rules tab. It was
rejected before creation by automatic approval review: the original no-browser
instruction had only been relaxed for read-only Modal billing. No rules tab
was opened, no existing tab was read or changed, and no browser retry or indirect
browser-control workaround was attempted. A prior direct public urllib read
failed DNS resolution; that was a transport failure, not an approval denial.
The browser action remains **denied**. The parent was notified immediately.
A separate root-agent raw public HTML read returned HTTP403 Forbidden; it was
not retried or routed around. This does not alter the primary indexed evidence
or grant access to the missing full live body.

## Exact local Attention wording and provenance

Saved official track source:
`https://openkernelchallenge.dev/api/v1/problems/muse_attention_decode`.
The snapshot manifest was created2026-10-24T17:19:39.565818UTC. The following
hashes were recomputed from current local bytes; TASK has no independent
semantic rules-version field.

| Local evidence | SHA256 |
|---|---|
| `reviews/attention-v2/official/TASK.md` (9,638bytes) | `fdc57e97579dd8579bb6cc284e454578962d30a65b1b7a526719ce220b0bba76` |
| `reviews/attention-v2/official/AGENTS.md` (2,720bytes) | `c01682737af060546f7f17c480cc37309c28e6c2de1cc13c669f3f66307d3476` |
| Snapshot `reviews/attention-v2/manifest.json` | `b9b32148fee0d62b72f82d3ef3e996d0ffaf7881cbdedeafeac9c4c33488c2f8` |
| Historical `/tmp/composition-official-rules.json` (126bytes) | `3d3c1d12707ff8f1572a86e8bf6de29b9f31e782115cc4bcc36eb4d8fdd0d71c` |

TASK.md:100 says a submission is rejected if it “starts processes or uses the
network.” This is a broad summary alongside bans on harness/timing/secret
access, imported references and module editing. TASK links the full global
rules; it neither implements the checker nor explicitly explains a compiler
exception. TASK.md:96–99 additionally retain CUDA-graph/no-host-sync, fresh
inputs, BF16 activations/FP8-or-finer query encoding and complete attention.
These numerical/computation constraints are independent of language choice.

TASK.md:119 says a run has15 minutes including compiling. AGENTS.md adds graph
replay timing and forbids untimed input-derived computation, output caching,
precision relaxation and harness manipulation; it does not separately ban a
native implementation framework. A compiler's untimed setup is distinct from
moving attention work on live inputs outside `kernel_fn`.

The historical rules receipt contains only version`2026-10-22.1` plus rules and
terms URLs; it contains **no full rule text**. Its file is not a new API read.
The saved workflow's same version field is historical evidence, not proof that
the current online version is unchanged.

## What the local checker actually is

Root `static_check.py` SHA256:
`c2b73f05b37d81be070b0d406e51c6926d5a68726c51ec43077d92ab38715239`.
Line1 explicitly calls it a small sanity check, “not the challenge's submission
validator.” It also asserts the **projection** problem marker at15. Its
lines20–23 whitelist only torch/triton imports and from-typing imports;
lines24–26 reject selected sync/I/O/eval names. It has no official native
compilation-exception implementation and is not an Attention admission oracle.
Indeed its whitelist would reject os and FA3 imports present in the unchanged
Attention source that passed official check5090. That is direct evidence of
its limited relevance, not a reason to edit or disable it. It was not executed
against a new candidate and remains unchanged.

`reviews/attention-v2/official/checks.py` is numerical checking, not the server
static validator. `reviews/attention-v2/check_cycle.py` is our external
quota/receipt/CLI wrapper; its Python `compile(raw, ..., 'exec')` parses syntax,
and its subprocess calls run the CLI/curl outside candidate code. They cannot
establish what the server permits inside a submission. No actual official
server static-checker implementation was found in the inspected local snapshots.

The existing executable reports **okc0.7.1** via read-only `--version`:
`/Users/frank/Documents/Codex/2026-10-23/task/bin/okc`,7,981,538bytes,
SHA256`01a6b091bc827a54403bdb32db721f275489f07c2d3490199f5457e4da27b4e6`.
CLI version and binary identity are not the server checker revision. No CLI
network command, token operation, compile, checker modification or job was run
by this audit.

## Remaining blocker and safe next decision

Global documented framework eligibility is established; blanket rejection of
all native compilation is unsupported. The exact complete live compilation
exception, version and server-checker handling are still **uncertain**. A
parent-owned authorized public read can close that provenance gap without
compiling or spending quota. Until then, do not infer that custom process
wrappers are accepted or that installing missing dependencies is authorized.
The parent's held experiment was not launched by this review, and no existing
precision, freshness, whole-attention, graph or $0-paid guard was relaxed.
