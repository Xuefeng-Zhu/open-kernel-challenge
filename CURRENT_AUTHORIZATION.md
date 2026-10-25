# A2-15 native diagnostic complete; no active GPU work

Modal app `ap-TQMFJknnfH0UZbG8lOBcBe` stopped at 2026-10-25 15:52:43 UTC;
a separate 15:53:12 UTC read confirmed stopped, zero tasks and no active apps.
Four exact-source H200 specializations compiled and disassembled. All 25 raw
files were retrieved and hash-verified. No candidate kernel was executed;
correctness, graph/cache behavior and performance remain untested for A2-15.

See [NATIVE_RESULT.md](reviews/attention-v2/a215/NATIVE_RESULT.md) and the saved
native artifacts. The compiler produces page-load/use separation across part
of PV plus double-buffered K/V, rather than the literal before-QK source order.
Producer shared memory rises 12,544 to 20,480 bytes; registers remain 96, raw
spill field 0. Both versions lower quantized FP8 Q/K to FP16 MMA in this
specialization. Merge SASS is identical. No new score or selection change.

Executed payload commit: 31010ce6b69d54767dc322a53b0cf6894b03d955.
A2-15 candidate hash remains fe09ee9736b3039b7c8edcd02968dc247c4457fe503629930fb41b7ea3c7fb3c.
Post-run local warning hardening is commit aaa801a; it was not uploaded.
Billing at 16:00:27 UTC shows $1.23 gross and $0 paid, an observed $0.03 rounded
increase. Keep the larger $0.061676416 model allowance: cumulative $1.849496218,
remaining $0.150503782 under the $2 stage ceiling. Prior accounting was retained.
No active reservation, uncertain request, new official check or further GPU run.

The next justified experiment is a full correctness/paired-timing check of the
unchanged A2-15 source against A2-13, retaining A2-11 as selected best. This
completed diagnostic does not itself authorize a new launch. The private export
will include reviewed source, artifact hashes and results; verify delivery from
the remote SHA receipt rather than assuming it from this status file.

The earlier checkpoints below are historical and superseded by this result.

---

# A2-15 status: native diagnostic blocked; official capability audit complete

The attempted Modal diagnostic stopped at the existing CLI's authentication
check on 2026-10-25 08:31:05 UTC: `Token has expired`, exit 1. No upload, app,
GPU execution, renewal or uncertain request. Incremental GPU cost is $0;
historical conservative cumulative accounting remains $1.787819802. Current
Modal credits, paid cap and apps could not be reverified. This phase does not
renew credentials or route around that prerequisite.

The subsequent official capability audit succeeded using existing OKC access:
attention 16/20 checks used, projection 3/20, all 37 returned jobs terminal at
08:36:21 UTC. No native artifact API was found in inspected CLI documentation
or saved receipts. No new official check was sent. The predeclared native
scheduling question remains unresolved; selected A2-11 and projection are intact.

Reviewed A2-15 source, tests and sanitized findings are prepared for the
existing private export branch. Actual delivery requires verified remote SHA,
recorded by the push tool and final task response, never inferred from this
file. See reviews/attention-v2/a215/OFFICIAL_CAPABILITY_AUDIT.md. Earlier phase
restrictions below are historical, not a statement of a current pending job.

The earlier local research scope follows as historical context.

---

# Historical phase: A2-15 local research only

The completed A2-14 exact check/private-push permission is exhausted. The current
instruction authorizes bounded local diagnosis, one next candidate or diagnostic,
local tests and a local commit. No new candidate upload, OKC check, Modal launch,
repository push, paid resource, credentials or formal entry is authorized.
Do not route around prior external-action restrictions. Preserve all selected
and failed versions. A2-15 is local-only; it has no reserved or accepted job.

A2-15 source, tests, results and exact native-evidence needs are under
reviews/attention-v2/a215/. Future native execution requires separately scoped
approval and verified existing access; do not renew credentials automatically.
The historical free-check count ceiling removal does not override this scope.

The completed A2-14 record follows as historical context.

---

# Current exact A2-14 authorization

The user approved one official off-board check of the frozen A2-14 candidate
and reviewed source, tests and sanitized results to the existing PRIVATE
Xuefeng-Zhu/open-kernel-challenge repository. This supersedes the preceding
local-only restriction for this exact payload only. No unrelated new upload,
paid resource, Modal launch, new credentials, formal entry, force push or
visibility change is authorized. Preserve A2-11 selection until evidence supports
change. Monitor the accepted job to terminal and never duplicate an uncertain
request. Use the sanitized export branch, not the original attention history.

Exact source/hash, command and status are in
reviews/attention-v2/strategy/approval_request.json and the A2-14 entry of
reviews/attention-v2/attention-cycle2-budget.json. Historical free-check count
limits were removed; actual daily platform limits still apply. This renewed
exact payload permission permits at most one new accepted check.

All historical Modal apps are stopped. Conservative cumulative cost remains
$1.787819802; this approval permits no Modal or credential action.

A2-14 is now terminal and its one-check permission consumed. Performance
regressed; preserve selected A2-11. The reviewed exact-payload private push has also completed and been verified.
Only this completion record is being synchronized. No unrelated future
transfer follows from this exception.
