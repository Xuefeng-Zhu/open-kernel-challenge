# R05-wide-layout vs R02-router-wide — saved H200 records only

R02-router-wide: job 89fe00c5034a4198, 1.0476946x SGLang, interval [1.031979203716431, 1.063410042408302]
R05-wide-layout: job 1631810d19af4af4, 1.0509508x SGLang, interval [1.03518654398615, 1.0667150681684692]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 10.5051 | 10.0242 | 1.00497 |
| 8 | 26.0400 | 25.9675 | 1.00588 |
| 16 | 9.6031 | 9.3782 | 1.03935 |
| 32 | 9.8180 | 9.7966 | 1.01582 |
| 128 | 10.0582 | 11.6574 | 0.87822 |
| 256 | 11.3033 | 12.6248 | 0.88725 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.007337 | 1.002337 |
| decode 16 | 1.081297 | 1.090872 |
| decode 32 | 1.066095 | 1.076199 |
| verify 1 | 0.995165 | 1.004127 |
| verify 16 | 1.071094 | 1.067510 |
| verify 32 | 1.068488 | 1.068201 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
