# Cycle2 P2-01 — pooler layout, sole accepted job before workflow pause

Source candidates/projection_pooler_layout.py, commit
0229137d871df93f09c83f86dacc15a95d1b6722, SHA256
b3346d5359602c098735e80dda5b38ab22f069fb19abd777fda5dc988f6a7571.
Hypothesis/CPU gates in experiments/015-pooler-layout-local.md.

```sh
python3 cycle2_check.py send P2-01-pooler-layout candidates/projection_pooler_layout.py
python3 cycle2_check.py poll P2-01-pooler-layout
python3 summarize_check.py P2-01-pooler-layout --ledger experiments/projection-cycle2-budget.json
.venv/bin/python compare_results.py R10-final-composition P2-01-pooler-layout --layer pooler
```

Accepted job **eeec05d897ef4946** is terminal done/pass,9/9 prerequisite +30/30
scored-shape checks, no checker flags/rejections. Official speedup **1.1290207051×**
SGLang,95%interval **[1.1120853945,1.1459560157]**. This highest observed point score
has an interval overlapping R10[1.1039211,1.1375430]; no statistically resolved whole-
workload win is asserted. Reference-normalized1.2439541361× original starter is a
cross-run comparison, not paired same-input trials.

Pooler m8/16/32/128/256 times20.2619/20.7076/20.6395/23.0184/27.9043µs, versus
R10 20.8359/20.7819/21.6892/24.7756/28.7850µs. Within-run SGLang-normalized ratios
improve1.0275/1.0277/1.0407/1.0791/1.0356×. Untouched m1 is18.4982 versus18.1286µs;
do not attribute variation of untouched paths to this layout. Full samples/profiles
and evaluator metadata in results/remote-P2-01-pooler-layout.json and CSV.

The user changed workflow to Modal before official submission while this accepted
job was in flight. It was monitored to terminal, not canceled or duplicated. All
NEW official admissions are now guarded/paused. Cycle2 budget1/10used,9remaining,
0pending,0uncertain;$0,no scored entry. Latest service quota12/20projection checks
used,8left; separate from user budget. No new work was pushed or published.
