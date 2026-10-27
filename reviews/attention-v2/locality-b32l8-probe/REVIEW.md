# Independent prelaunch review

**PASS for one bounded diagnostic. No implementation blocker found.**
Independently ran all **50 CPU tests: PASS**, plus the default source preflight:
seven frozen bindings verified, zero vendor imports/GPU launches/service calls.
This reviewer used no services, GPU, authentication, browser or Git operations.

| File | SHA256 |
| --- | --- |
| `worker.py` | `7bfe1237a81094b217b665dcdd325c9857a1180f66d917b7ecb1e0414f21315d` |
| `controller.py` | `cc11d5c4fe2d7bc1df4525a2fc08fa71ecd74aacf77709aff5724c4252e5f4a6` |
| `fixture.py` | `203240b0f4de073b4c6222c62daeeaff20ce0b3de32b519f8d9f59c5528e0e28` |
| `trace_helpers.py` | `d89e2f6dac4d8afd9105b713ce255407df8e9253c330a1e727d8193a220c5ec8` |
| `test_worker.py` | `12dfffc2e30fcea761d90da73d17847354d8ce31dc90f81ab21ff3e09b9e687c` |
| `test_controller.py` | `141e41c560616142113d526740a79c3d3d6e55c4ce199c4ecc293d478da1b216` |
| `test_fixture.py` | `24f11ea7c9f412104af10d0f069e24f5426176179e9fa22a51264204b46de821` |
| `test_b32_semantics.py` | `f1e29ea647eaf31423a7beda9c20dc89c2c7438a46a54acf4118ab7c90614906` |

Scope is exactly B32/L8 full attention, context 80000, page1, TP8, unit scale,
agentic/coherent inputs, seeds 135831 and 135832. Selected source remains
`3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8`.

The complete physical-cache permutation is bijective, preserves every encoded
K/V byte including unused rows, and remaps page tables/new-token locations
together. CPU tests include unique row IDs, every B32/L8 causal key range and
post-write inverse equality. Runtime verifies unchanged nonlocality inputs,
full inverse cache equality, new-write ownership and original oracle results.
Cross-layout bitwise output equality remains observational; each arm keeps the
original tolerances and exact replay against its own oracle-checked output.

One graph, input/output addresses and layouts persist across both fixtures and
arms. First-seed fixtures and full-cache recovery temporaries are released.
There are four original-oracle calls, 16 warmup replays, 48 measured replays
and two activity replays: **70 checked replays**, with capture calls excluded.
Timing contains only graph replay/events; restoration and 256 MiB eviction
are outside it. No replay-cost subtraction is used.

Activity attribution uses the frozen host-launch correlation plus matching GPU
annotation/external ID helper, tested against displaced CPU/GPU clocks and a
real saved trace. Complete equal native symbol/node/launch metadata, including
one FA3 mainloop per arm, is mandatory. Missing or different identity fails.
The controller verifies its local worker hash before recomputing completion
and the decision, and independently requires full-capacity permutation and
post-call full-cache inverse proof.

For valid completed measurements, **KEEP** requires at least a 15% latency cut
on each fixture and all six reversed-order pairs per fixture favorable (12
pairs total). Otherwise the valid result is **DROP**. Invalid correctness,
trace or completion evidence fails without a causal performance conclusion.
KEEP retains a locality-sensitivity hypothesis only. Layout construction is
outside timing; no production candidate, official score, repacking benefit or
hardware bottleneck is established.

The flat source-only payload and partial-result/trace transport are bounded.
One H200 invocation uses worker/child/function/watchdog limits of
160/175/190/220 seconds, no retries, and a $0.70 free-credit reserve. Launch
still requires the root's live credit/$0-cap/authentication/slot checks and
independent stopped/tasks-0 verification afterward. No additional experiment,
official check, ranked entry or candidate change is part of this review.
