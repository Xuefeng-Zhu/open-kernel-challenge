# Attention A02/A03 — corrected v2 baseline first

Scope: only minimal LOG2Econstexpr fix05922b7/SHA3a86e835..., original algorithm
unchanged. Current public officialAPI files/TASK/AGENTS exactly match snapshots;
16prerequisite fixturecases from own priorfailedreceiptbf4b49bd8bb34dae,12scored
cases/seeds from currentpublicAPI. No competitor software consulted/executed.

Gate corrected v2 against unmodified official collect/check_implementation, which
checks BF16output and newFP8K/V. Output tolerance is clamp(2.5*referenceFP8query
error+0.015,0.03,0.15), perrow5x/nearzeronormalization; cachebound8%. Test actual
page64/default, page1, nonunit1/64scales, TP8/DPheads, edges/soft/coherent, L1/8,
millioncontext and all12scoredshapes. Include realSGLang same-runreference.

All input tensors/configs snapshotted, restored before eachrunner; cachebytes may
change only cache_loc rows, all other rows and noncacheinputs checked unchanged.
Collect newentries before restore; oracle attends candidate's own checkednewentries.
Fresh generatorseeds change qkv/lengths/pages/cache and all fields copied into
fixed graphbuffers; metadata shapechanges fail visibly instead of silentlyrecapture.
No inputderivedresults cached by candidate. SequentialisolatedCUDAgraphs only;
existingglobalticketconcurrencybug remains, no claim of concurrentcall safety.

Eachprerequisite captured/replayed/checked perrunner. Eachscoredshape has2declared
warmupreplays retained/excluded,6steady fresh replays alternatingrunners,3reverse
pairs. All outputs/cachewrites checked eachreplay.256MiB coldL2readoutsideevents.
Raw whole-call CUDAgraphtimes retained. EmptyCUDAgraph measured3before/3aftercase;
separate diagnosticcalibratedvalues subtract its measuredmedian frombothrunners,
no guessed4us/clipping. Calibrationnotavailable/nonpositive is reported as such,
not a formalOKCscore orCI. Setup/compile/wall separate. Resource metadata for
fullverifyB16/B32 via compiledwarmup; no restrictedhardwarecounters.

OneephemeralH200 perrun,CPU2/RAM8GiBrequest+limit,max1/min0/buffer0,scaledown2,
retries0,function240s/startup60s/internal210s/wrapper220s. Reserve0.50USD against
existingconservative1.146411930/2USD; no budgetreset orloweringdelayedreads.
All prior11apps mustbestopped; verifylivecredits/cap0/auth expiry first.
Only12allowlistedsource/licensefiles+manifest, no repository/config/auth contents.
No newcredentials/renewal/terms/payment/push/OKCchecks. Existingauth expires
2026-10-25T06:26:47Z. Baseline mustpass before implementing any new algorithm.

Local gates: syntax,5cache/metadata/one-linepatchtests,7stillapplicable CPUequation/
mask/tickettests. Historicalpreflight's normalization-only assertion failed because
it expects no compilerrepair; retain cpu-constexpr-results.json and oldpreflight
unchanged. Newexactpatchgate replaces thatobsoleteexpectation; CPUmathis not
Tritonexecution. Newcpu-constexpr-gates.json records12tests andequationcases.

```sh
.venv/bin/python reviews/attention-v2/test_modal_worker.py
.venv/bin/python -m py_compile reviews/attention-v2/submission.py reviews/attention-v2/modal_worker.py reviews/attention-v2/modal_control.py
.venv/bin/python reviews/attention-v2/modal_control.py build --mode baseline --bundle local-notes/attention-a03-bundle
.venv/bin/python reviews/attention-v2/modal_control.py run --mode baseline --experiment A03-baseline --bundle local-notes/attention-a03-bundle
.venv/bin/python reviews/attention-v2/modal_control.py finalize --experiment A03-baseline
```

If correctedbaseline fails, preserveexactfailure andstopbefore independentmerge.
If correct andbudgetpermits, onlyfull-attentionfinalmergewillbeimplemented in a
separatesourcecommit, preservingpartition/math/newtokenownership. No further
fixedscheduling/prefetch/FP8Q/doublehead experiments in this cycle.
