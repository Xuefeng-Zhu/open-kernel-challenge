# R10-final-composition vs R07-router-native-wide — saved H200 records only

R07-router-native-wide: job 26ea99ee0869482a, 1.0991970x SGLang, interval [1.0827090458817419, 1.1156849559085968]
R10-final-composition: job 02e099c3724e4a05, 1.1207320x SGLang, interval [1.1039210682982272, 1.137543029769239]

Independent runs, not paired same-input trials. The normalized ratio below divides
each candidate/SGLang ratio by the control/SGLang ratio; it does not remove
all run-to-run noise or establish statistical significance. Higher is better.

| m | control us | candidate us | candidate/control within-run SGLang speedup ratio |
|---|---|---|---|
| 1 | 9.2852 | 7.1789 | 1.28936 |
| 8 | 9.1417 | 7.4584 | 1.24533 |
| 16 | 9.2756 | 9.1754 | 1.01755 |
| 32 | 9.6759 | 9.4573 | 0.99761 |
| 128 | 10.3692 | 10.2562 | 1.01300 |
| 256 | 10.9153 | 11.0143 | 0.98556 |

| load | control speedup vs SGLang | candidate speedup vs SGLang |
|---|---|---|
| decode 1 | 1.019832 | 1.108670 |
| decode 16 | 1.073936 | 1.080303 |
| decode 32 | 1.083296 | 1.070531 |
| verify 1 | 1.298175 | 1.339258 |
| verify 16 | 1.070246 | 1.078765 |
| verify 32 | 1.069997 | 1.069726 |

Untouched layers can vary too. Inspect all shape records rather than attributing
a whole-workload change to the targeted kernel. No new performance is predicted.
