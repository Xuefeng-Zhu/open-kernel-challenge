# Independent official composition check review

Official correctness and receipt review pass. The result provides complete
prerequisite/scored coverage for the composition and is consistent with a modest
target-path improvement. It does not prove an aggregate score difference versus
the fusion control or meet the old10% architectural goal. No official request,
CLI/API/network/browser/credential/compute action or source edit was performed
by either reviewer. This report is the only official-review artifact written.

## Job identity, source binding and checks

Job **`5090b83f1d63407b`** is `mode=check`, board `muse_attention_decode`,
terminal `done`, nonduplicate. The result is passed, returncode0, no error,
no global/benchmark flags, checks.ok true and empty flagged/rejected lists.
Exactly16 prerequisite test events pass, with12 benchmark and12 profile events.
Profile events are not extra correctness tests or benchmark samples.

The accepted local request binds the exact committed candidate
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`
at source freeze `c9062363c665e37535d3370c1b6bfb54e04fc7c8`; those bytes
were independently compared with the frozen Git object. This is the same source
validated in the native composition test. The raw terminal response SHA256 is
`0095903e817e17aa8b252c4310a1b6a140aee1cc3690b82a8b04c40f05283cbc`.
The service content identifier is separately
`30f364457bdd36bbf7b7e364feb66620985069bf2ef1d631d16c20877d2b75e5`;
it is not asserted equal to raw local source SHA256. No ranked entry was made.

## Score reconstruction

For each load, independently sum `46×sliding +16×full` for reference and
candidate benchmark rows, take reference/candidate, then the geometric mean
of all six loads. Ignore profile rows in this calculation.

| Load | Weighted reference µs | Weighted candidate µs | Load speedup |
|---|---:|---:|---:|
| Decode1 | 1445.440038 | 960.448082 | 1.504964261 |
| Decode16 | 4426.175935 | 3748.722826 | 1.180715711 |
| Decode32 | 7562.239867 | 5789.276070 | 1.306249655 |
| Verify1 | 1627.200008 | 1254.558905 | 1.297029579 |
| Verify16 | 4598.239907 | 4161.685626 | 1.104898428 |
| Verify32 | 7787.712103 | 7349.602431 | 1.059609982 |

The reconstructed aggregate equals the reported **1.2336330588627955×**.
The official95% interval is **[1.2151285629798536,1.2521375547457374]**.
Canonical reference3766.3µs divided by score3053.014810962µs reproduces the
same speedup. The validated fusion control job `2f5f1fb54b48486d` reported
1.2255554363764571×, interval[1.2071721048308102,1.250062192674554]. The new
aggregate point estimate is **0.659098907% higher** than that historical job.
This is a cross-run point comparison, not a paired candidate-versus-parent
score test. Overlapping marginal intervals are not a significance test of the
difference, and the point difference is smaller than the reported interval
widths. No aggregate statistical-superiority claim follows.

## Primary fusion comparison and unchanged controls

Use the validated preparation-fusion receipt as the primary control, not the
rejected unsplit variant or A2-13. Its published response SHA256 is
`46f9818a029a68995d6bad8b99c79a72f390ab09e464c01eb9a79b64bb160001`.
For each path, normalize calibrated candidate time by its own same-run
reference time. The final column below is
`100×[1−(current_candidate/current_reference)/(fusion_candidate/fusion_reference)]`.
Positive values mean a reference-normalized cross-run latency reduction.

| Path | Fusion candidate µs | Current candidate µs | Current same-run reference µs | Current same-run reference/candidate | Normalized reduction vs fusion |
|---|---:|---:|---:|---:|---:|
| Decode1 sliding | 11.621169 | 11.477114 | 20.576001 | 1.792785274 | −1.832271% |
| Decode1 full | 27.637677 | 27.031303 | 31.184000 | 1.153625506 | +1.190359% |
| Decode16 sliding | 14.462795 | 13.921137 | 26.400001 | 1.896396947 | +2.228452% |
| Decode16 full | 194.360665 | 194.271909 | 200.735994 | 1.033273388 | +0.013793% |
| Decode32 sliding | 19.915057 | 20.103604 | 30.975999 | 1.540818228 | −0.946759% |
| **Decode32 full, changed** | **317.205126** | **304.031893** | **383.583994** | **1.261657090** | **+3.825075%** |
| Verify1 sliding | 17.017043 | 16.988207 | 23.008000 | 1.354351247 | +0.516567% |
| Verify1 full | 30.125822 | 29.568836 | 35.552002 | 1.202347017 | −0.536437% |
| Verify16 sliding | 22.892705 | 21.740250 | 28.687998 | 1.319579973 | +3.021495% |
| Verify16 full | 197.827063 | 197.602134 | 204.911999 | 1.036992849 | +0.137096% |
| Verify32 sliding | 27.459687 | 27.467619 | 34.336002 | 1.250053789 | +0.810137% |
| Verify32 full | 382.479525 | 380.380746 | 388.016002 | 1.020072665 | +0.105831% |

The changed path's current same-run official speedup is1.261657090× against
SGLang. Its **3.825075%** reference-normalized cross-run reduction versus fusion
has the same direction as the native paired benefit but is a different
comparison. Do not call this an official≥4% paired reduction: the predeclared4%
gate applied to the native same-session composition-versus-fusion test, where
4.822642% was observed. The official result is not a paired new-versus-fusion
run and does not supply that same test or a confidence interval for it.

The eleven unchanged paths' normalized reductions range−1.832271% to+3.021495%.
The largest normalized regression is1.832271%, below5%. Their nonzero drift
illustrates why the entire0.659099% aggregate point difference cannot be
attributed solely to lookahead. In particular the unchanged sliding-verify-B16
path improves about3.02% by this cross-run normalization. Preserve every row
rather than selecting favorable controls or subtracting absolute times from
different runs.

## Comparable definitions and remaining uncertainty

The entire evaluator metadata dictionary exactly matches the fusion control,
including code hash
`1d519ebf2014fef91b41ffea9bcd7e9c5066784ef5a2a5e2cf06a8cec554d188`,
all official baseline/reference/check/harness hashes, and judge image
`localhost:30500/kc/judge@sha256:9a7f287b80d772c6f636bbcb57422c619fe326e2aabc53803b85007bca93f68e`.
Both report H200, driver580.159.03, SM clock1980 and memory clock3201. They ran
on different nodes; all12 benchmark seeds and all16 prerequisite seeds differ.
Shared definitions and reference normalization support comparison, but do not
turn different inputs, executions or calibration into a paired score result.

The native source-bound run separately passed32 oracle calls, replay/cache
gates and its new4% replication threshold with4.822642% paired target reduction
and0.153236% unchanged-control slowdown. Together with this complete official
correctness result, that supports retaining the composition as a modest
validated candidate. Neither experiment isolates page-ID scheduling from the
compiler-generated buffering change or meets the historical10% goal. This
review does not promote it over original selected A2-11 or claim an award-level
architecture advance.

The parent reported terminal observation and immediate serialized-slot release
at21:27:58.453455UTC before polishing. No additional check, GPU job or service
query was performed by the reviewers. No receipt, arithmetic or correctness
blocker was found for preserving and privately publishing these results.
