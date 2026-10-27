# Independent result narrative review: PASS; producer branch remains STOP

Reviewer: `/root/compact_review`. This is a local read-only evidence review;
only this report was written. No compiler, container, service, GPU, official
check, authentication, network or Git operation was performed by this reviewer.

The reviewed files are bound to these SHA256 values:

| File | SHA256 |
|---|---|
| `RESULTS.md` | `c491c4a897e4829c93d50a6249a0ca621d67e4170f11cc92fce7b1552ed8743f` |
| `RESOURCE_CLOSURE.json` | `c2eb4d1b8b513d681f6ea1977bdd3174f9666b267d94db90397e1297bd2af0f2` |
| `ARTIFACT_REVIEW.md` | `96c3d2703fad81a46d6cc5f764d3de26a9e25c28028a9d5c2190644b79564838` |
| `evidence/compiler-manifest.json` | `2133c3ccccba2767b3513b5992b91ab342016ed52800b5c3ceb57bbf3f59fa68` |
| `evidence/execution-receipt.json` | `bd8aacd5e2d792f7ba92a7b2364e6d074eba552d41449c4941bdd2ee040b2f99` |

**PASS for accurate reporting, not candidate admission.** The compiler manifest
records four controls and only two candidates, in that order, followed by
`stopped_on_candidate_resource_gate`. D32 full has74 registers, zero stack and
spill bytes, and34944 dynamic shared bytes. D1 sliding has75 registers,32 stack
bytes and zero reported spill bytes. Its96 local-load and64 local-store static
sites in both PTX and SASS match all three96-register sliding controls. The
report correctly separates those observations from LOCAL0 and assembler
spill0; it does not describe this as added register spilling or measured cost.
Static shared1024 B is separately recorded, making the conservative sum35968 B.
Candidate D16 and D32 sliding remain uncompiled and unknown.

The independent artifact review traces the shared sliding local sites to
**inlined RoPE trigonometric range-reduction arrays**, not established
out-of-line calls. RESULTS preserves that qualification and does not claim
the slow paths execute for benchmark inputs. Eight historical M64N8 WGMMA
sites and the separate sixteen N16-tail MMA sites establish generated-code
presence only. The report does not certify complete buffering/publication,
precision, execution counts, achieved residency or a numerical oracle result.
The narrower candidate copy instructions are correctly an unmeasured risk.

The earlier container is separately classified as a cross-Python AST-manifest
failure before Triton import/compiler invocation. `PREPARATION_PROTOCOL.json`
records zero earlier compiler calls and unchanged kernel, candidate, helper,
options and gates. Matching Python3.12 preparation preceded the single actual
compiler session, which made six compile calls. This is not represented as a
native configuration rescue or threshold relaxation. Current local kernel,
candidate and helper hashes match the frozen hashes in RESULTS.

Closure consistently records two own container sessions, both exit1 and
removed, plus a later root exact-name absence query. The shared engine's later
Stopped observation is attributed as an observation, without claiming this
task stopped it or identifying another actor. These are saved receipts/root
observations; this reviewer did not independently query live resources. The
absence of a fresh global Modal inventory is explicitly retained. No GPU or
Modal app, official check, credential action, new download, host installation
or paid resource is reported for this stage; paid spending is$0.

The other artifact reviewer verified the50-member text archive against the
index and originals. RESULTS accurately attributes complete retained text,
local-only cubins, and the six-pattern scan without claiming a comprehensive
security audit. This narrative review does not repeat that archive audit.

The absolute zero-stack/local rule fails behavior already present in the
selected sliding control. RESULTS candidly identifies this limitation while
retaining the predeclared STOP decision: no remaining compile, full-only
rescue, GPU/official run, timing, score gain or promotion is justified by this
run. The selected1.2336330588627955 candidate and A2-11 rollback remain unchanged.
No reporting blocker was found.
