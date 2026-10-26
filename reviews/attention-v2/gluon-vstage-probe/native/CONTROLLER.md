# Revised common-K / V-stage compile harness

This is a local adaptation of the reviewed `gluon-overlap-probe/native` harness.
Old source, artifacts and results remain unchanged. The new labels are exactly
`v_after_wait` and `v_before_wait`; the single selector is `V_BEFORE_WAIT`.
`config.compile_kwargs` accepts those labels, not booleans. Both arms share
raw-K prefetch, identical work/math/buffers and the same fixed B32/HKV2/HQ16,
M64/N64/D64, page1/stride86008, four-warps/one-stage configuration.

Read the source-owned `../EXPERIMENT_CARD.md` for the prospective mechanism
gate. Native review must identify an actual remaining V-gather/conversion/
shared-publication contrast around the corresponding QK wait, with common-K
prefetch, matched work and safe ownership. A fully serialized control is not
required or manufactured. If relevant native schedules are identical, do not
execute or benchmark. Resource gates remain zero spills/local access, shared
at most64KiB and no reduction in legal maximum CTA/SM versus `v_after_wait`.
Missing exact ISA, limits or ownership proof is inconclusive, not admission.

The installed Triton3.8.0/Gluon/Hopper/async-source pins, source/hash guards,
bounded gzip artifact transport, exclusive ledger and shutdown logic are
unchanged. Artifact names are `v_after_wait.core.<format>` and
`v_before_wait.core.<format>`. Required outputs are TTGIR/PTX/CUBIN; TTIR/LLIR
are optional. Readable exact SASS is required for subsequent mechanism review.

Worker default is CPU source preflight only. Native compile requires explicit
`--compile --output PATH --deadline 90`. The controller defaults to offline
frozen-Git preflight; explicit `--launch` requires fresh parent slot and billing
receipts. It adds the worker's `--compile` flag, compiles two units and never
invokes their deliberately undersized one-element pointers.

The fixed existing image/H200/2CPU/8GiB envelope, one app, zero retries,
90s worker/100s child/115s function/20s startup/140s watchdog and $0.40 free-
credit reserve remain unchanged. Actual paid limit is $0. Existing protected
credentials are selected in place; there is no renewal, install or billing
change. No GPU action follows merely from local preparation. The parent must
review, freeze and run actual preflight before the separately coordinated slot.

No GPU, service, network, credential or push action was performed by this
harness author. Compilation is neither numerical validation nor performance
evidence, and no automatic official check or production selection is made.
