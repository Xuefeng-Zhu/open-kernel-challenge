# 021 — QKV/unpooler m1 composition on frozen R10

Compose the independent019 unpooler and016/018/020 QKV paths. Only the m1 dispatch
set grows to include qkv_gate/unpooler beside R10 pooler; original optional weights
are BF16-contiguous. Kernel body, router, larger-M and all other paths unchanged.
Keep this layout-free candidate as an independent control for pooler-layout effects.

Source candidates/projection_native_composition.py, SHA256
7b571e48c02240da9e464aee5a8bda8c532628562e06540187205b7acfa5cc27.

```sh
.venv/bin/python check.py --submission candidates/projection_native_composition.py --device cpu --suite all --output results/router-cycle2-native-compose-cpu.json
PROJECTION_CANDIDATE=projection_native_composition.py .venv/bin/python -m unittest discover -s tests -p test_projection_cycle2.py -v
.venv/bin/python static_check.py --submission candidates/projection_native_composition.py
.venv/bin/python -m py_compile candidates/projection_native_composition.py
```

Actual local result:39/39CPU oracle cases and9focused tests passed (three modified/
preserved layers, fresh inputs, zero/probe rows, N/K tails, real widths and control
paths). Syntax/static pass. M2/macOS26.6.2, Python3.12.13, PyTorch2.14.1, no CUDA.
This combined source still needs its own GPU check; component results alone are
not combined correctness/performance evidence. Planned Modal factorial stability
comparison with full composition, pooler-layout control, R10 and SGLang; no OKC
dispatch or push, no new credential, within the existing$2credit/$0paid stage.

## Subsequent native validation, experiment023

Exact source bytes preserved. App ap-XxzsrPfRejsEpPVQCxIqzJ passed39CUDA cases for
this runner and all30shapes x6declared warmups +30steady fresh-input repetitions,
then verified stopped/tasks0. Comparison and limitations: MODAL_COMPOSITION_RESULTS.md;
all raw samples in results/remote-modal-M23-composition.json. No official score/CI.
