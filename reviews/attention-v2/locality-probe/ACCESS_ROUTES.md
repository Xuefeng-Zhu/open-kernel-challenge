# Existing execution access: bounded audit

No locality GPU execution has occurred. The diagnostic is input-fixture
construction and measurement, **not a contest candidate**. It must not be
uploaded to an official check to alter judge inputs or obtain synthetic timing.
Selected source remains `3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

| Existing route checked | Actual observation | Scope and decision |
|---|---|---|
| Protected Modal `xuefeng-zhu` profile | Latest recorded authenticated read attempt returned exit1 / expired token | No new or renewed credential operation attempted in this audit; CPU/H200 app not admitted |
| Standard Modal configuration/environment | `~/.modal.toml` absent; no Modal credential environment variables present; protected file contains only the known profile | Names/presence only were reported; no credential value printed, copied or searched elsewhere |
| Browser notebook execution | Current [official Modal documentation](https://modal.com/docs/guide/notebooks) says its notebook editor has shut down and cannot create or run notebooks | No notebook, browser session, token or replacement Jupyter service was created |
| Existing configured SSH alias `hermes-ts` | Local configuration resolves an existing host; one read-only GPU-inventory attempt exited255 after the8-second connection bound, without a usable response | Transport unavailable; authentication/GPU presence unverified. No host-key acceptance, forwarding, credential or resource change; no retry |
| Official challenge runner | Free nonranking checks independently work | Not a general diagnostic shell. Do not submit unchanged code or synthetic input manipulation as a candidate |
| Existing Mac CPU compiler | Pinned cached Triton3.8 wheel and Docker image reused successfully in an isolated CPU-only container | No Modal credential needed. This enables compiler evidence, not GPU correctness, profiling or timing |

Owner-observed access facts and their limits are recorded in
`ACCESS_OBSERVATIONS.json`; raw connection/configuration details are deliberately
not persisted. The notebook finding is documentation evidence, not an
authenticated dashboard inspection. The local credential inspection was intentionally limited to
standard configuration and known task configuration; it is not proof that no
other credential exists anywhere. No browser cookies, secret stores or
unrelated accounts were inspected.

The SSH attempt used BatchMode, strict existing host trust, no host-key updates,
no agent forwarding, no configured port forwarding and an8-second connection
timeout. Its raw connection details were withheld. We did not obtain a GPU
inventory and do not label the host GPU-free or the failure a permission denial.

The known H200 path still needs usable existing authentication. Creating a
replacement Modal credential would be a separate persistent-access action
requiring explicit applicable approval; it is outside this task and has not
been retried. This access limit does not block CPU fixture development or the
new local offline compiler path.

## Prepared diagnostic and remaining gates

The fixture and worker passed **25 CPU tests and default offline preflight**,
plus independent source/protocol review. Full cache capacity is permuted
bijectively, including unused rows; logical FP8 bytes, new-token ownership,
QKV, global positions, lengths and scheduler metadata remain equivalent.
The future worker checks fixed tensor/output addresses, meaningful attended-page
contrast, both original-oracle results and complete inverse cache bytes.
Missing launch metadata is UNKNOWN, not false equality.

Native oracle/replay behavior, timings, activity collection, a remote controller
and hard external admission/watchdog/cleanup are still untested. The default
worker never imports vendor packages or executes a kernel. A later admitted
run may use `--execute`; this document does not admit it.

The separate M64N8 compiler probe already provides an executable, credential-free
next gate: native historical-tile lowering. Its results and strict limitations
are in `../m64n8-transpose-probe/RESULTS.md`. No official quota or paid spend
was used for this access audit, fixture preparation or offline compile.
