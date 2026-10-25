# A2-08 — single-token full attention uses the verified full prologue

Parent A2-07 sourcef1b3e8c/jobba9e02c1d6584cc4,1.1505332420x. Its fullverify1
fusion passed, target15.77%normalized cut and million-context/page64 correctness.
Now only fullB1L1 joins the same host adapter/kernel. Every function's AST is
unchanged except length guard8→(1,8). No arithmetic/precision/layout/stage/warp/
chunk change. For L1 the existing shape formulas produce5CTAs rather than40;
this is not a new tuning parameter. Supplied FA3 decode scheduler is forwarded.

Parent full decode1 adjusted31.874us. Its separate profile has norm, threecasts
andcachewrite; proven fusion removes four launch boundaries. Profile costs cannot
be subtracted from graph latency. Predeclared goal>=10%normalizedtargetcut,
inspect all11controls>5%regression, and qualify aggregate CIoverlap. Full paths
are attention-dominated, so modeled aggregate gain is modest; scenarios saved
in A2-08-cpu-preflight.json are not results. Reject if the target goal is missed.

SevenCPU/static methods pass, including exact helper preservation and complete
entrypoint; the component suite has16modelcalls covering TP8/DP,page1/64,scales,
live0,emptyhistory,finite oracle/cache,unchanged oldrows and same-addressABA with
changed lengths/metadata. Synthetic CPU scheduler is only a forwarding sentinel,
not a valid native encoding. No actualTriton/FA3/CUDA execution in local tests.

Official new native target is fullB1L1,page1,scale1,context80000 with supplied
scheduler. Preserved fullB1L8 still covers million-context/page64. This does not
prove L1 page64/nonunit/DP/explicitABA/concurrent-stress on GPU. No local GPU or
Modal launch is authorized. Source hash,commands and actual outcomes are retained.

```
.venv/bin/python -B reviews/attention-v2/test_full_fused_small_dispatch.py
.venv/bin/python -B reviews/attention-v2/test_full_fused_decode_model.py
.venv/bin/python reviews/attention-v2/check_cycle.py send A2-08 --cycle2
.venv/bin/python reviews/attention-v2/check_cycle.py poll A2-08 --cycle2
.venv/bin/python reviews/attention-v2/assess_cycle2.py A2-08
```

One serialized off-board check from3remaining; two remain after acceptance,
including final reserve. No ranked/paid/newcredentials/Modal. Reviewed automatic
private push remains authorized. Do not repeat identical-source checks for a
ceremonial final: the actual complete file is checked here.
