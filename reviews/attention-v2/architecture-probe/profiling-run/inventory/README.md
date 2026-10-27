# CPU-only existing-tool inventory

Prepared locally; **not launched** by this agent. Parent reported that fresh
billing UI verification was denied, so the controller remains inadmissible
until that prerequisite is resolved within authorization. No workaround is
implemented. No GPU, CPU service, browser, network, credential or install action
was performed during this preparation subtask.

`worker.py` imports no Torch/CUDA library and requests no context/kernel/metrics.
It inventories PATH and bounded existing CUDA/NVIDIA directories for ncu/nsys,
selected CUPTI libraries and distribution metadata. Only executable --version,
--help and ncu --list-sections commands run, max4s each; report output and file
hash bytes are bounded. Missing tools or failed metadata commands are recorded,
not interpreted as hardware-counter permission or a workload bottleneck.

`controller.py` uploads worker.py only to the existing immutable image. Resources
are explicitly gpu=None,2CPU,8GiB,max1container,no idle buffer and no retries.
Worker20s / child25s / function30s / startup15s / appwatchdog50s deadlines apply.
The app watchdog starts before app creation. Existing protected config is used
in place through MODAL_CONFIG_PATH, never copied or printed. Fresh credit/cap,
zero-paid live billing, idle apps, exact committed bytes and exclusive request
ledger are required. Planning reserve is$0.02; no artificial stage ceiling.

The parent must review/freeze worker and controller in one exact HEAD before any
invocation. The billing receipt argument must be a regular file in sibling
`profiling-run/control/`, using the existing receipt schema. Stop is explicit;
up to three bounded read-only status observations follow. A remaining stopping
or uncertain state must be independently reconciled; never relaunch automatically.

Six local safety/transport tests and Python compilation passed. Tests execute
no inventory command or Modal operation. Their receipt and full source hashes
are in local-verification.json and manifest.json. Counter installation/permission
remains a separate GPU-stage question regardless of this inventory's outcome.
