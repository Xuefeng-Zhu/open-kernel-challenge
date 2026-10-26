# Independent revised V-stage SASS review

Offline SASS-only review of compile result
`258c907f54f03c9d301f25724cdf1d393b652cba167957de55cc8819a0a41240`,
source `6a5b1cf73eb2ab20b1f1b22b3666bec92affd452ceee1625b1a194f73adef83e`.
Both SASS hashes were checked against `artifacts/manifest.json`:

- `v_after_wait.core.sass`: `19341152847cfea894358c0c59c5ac5b202128bb4c9c08664e9ab5a7bfd70c90`.
- `v_before_wait.core.sass`: `4e32ac70a6b8fe76e28322eb5a889e22169edabc442daa05b017908b43fc58d6`.

**The narrow prospective remaining-stage placement gate passes.** Complete
next-tile K/V shared publication precedes the corresponding QK drain only in
`v_before_wait`; it follows that drain in `v_after_wait`. The schedules have
not collapsed. Both arms already hoist V loads, and some conversions in the
after arm, ahead of the QK wait. Consequently this is a surviving publication/
conversion-placement contrast, not proof of a V-load-only contrast or literal
source-order execution. The revised card expressly permits partial hoisting;
this finding does not amend or pass the older fully serialized-control gate.

## Exact boundaries and value chains

Lines include encoding lines in the saved complete SASS; PCs are byte offsets.

| Native event | `v_after_wait` line / PC | `v_before_wait` line / PC |
|---|---|---|
| Current BF16 QK group |1361/`0x2a60`,1409/`0x2be0`,1421/`0x2c40`,1423/`0x2c50` |1447/`0x2d10`,1461/`0x2d80`,1465/`0x2da0`,1489/`0x2e60` |
| Next K first payload load |1283/`0x27f0`, before QK |1307/`0x28b0`, before QK |
| Next V first payload load |1291/`0x2830`, before QK |1313/`0x28e0`, before QK |
| First next K shared store |1677/`0x3440`, after drain |1565/`0x30c0`, before drain |
| First next V shared store |2289/`0x4760`, after drain |1705/`0x3520`, before drain |
| Final next V shared store |2331/`0x48b0`, after drain |1801/`0x3820`, before drain |
| Shared publication fence |2335/`0x48d0`, after drain |1809/`0x3860`, before drain |
| QK drain |1531/`0x2fb0` |1813/`0x3880` |

The loads are identified by pointer dataflow, not mnemonic totals. In both
arms, `c[0x0][0x218]` supplies K base `UR14` and `c[0x0][0x220]` supplies V
base `UR16` immediately before the loop (after PCs `0x2580/0x2590`; before
`0x2570/0x2580`). The page1 int32 entry is clamped nonnegative, multiplied by
`0x80` for HKV2×D64 bytes, then combined with head/lane offsets. Addresses add
K or V base separately. This confirms common live next-K gathering, not a
conversion masquerading as a new global load.

For a representative V word in **after**, address formation adds `UR16` at
`0x2800`; `0x2830` loads `R83`; `0x2cd0` unpacks E4M3 into `R85`;
`0x2d20/0x2d30` widen its halves into BF16 `R80/R81`. These already precede
the drain. Their values remain live until `0x4740` packs them into `R24`,
which `0x4760` stores to the next V bank, after the drain. Other V words
continue conversion after the drain, e.g. `R143` unpack at `0x35c0`.

For the corresponding word in **before**, V address formation adds `UR16`
at `0x2890`; `0x28e0` loads `R84`; `0x2de0` unpacks into `R81`;
`0x2ec0/0x2f80` widen to `R140/R139`; `0x34f0` packs them into `R72`;
`0x3520` stores the next V bank. This whole value chain and the remaining
publication finish before drain `0x3880`. Thus native evidence supports the
revised remaining-stage contrast despite V-load hoisting in both arms.

## Shared-slot and PV ordering

Both initialize slot `UR4=0` (after1197/`0x2540`, before1191/`0x2510`).
Current K base is formed from `UR8+(UR4<<13)` into `UR9` (after1329/`0x2960`,
before1279/`0x27d0`). The slot bit flips before next publication (after1605/
`0x3200`, before1445/`0x2d00`); next base `UR7` is then formed by the same
shift13 formula (after1641/`0x3320`, before1455/`0x2d50`). Hence current and
next bases differ8192 bytes. Next K stores use `UR7`; next V stores use the
same base plus `0x4000`, in a separate16384-byte bank. Parent IR/PTX review
must bind printed `gdesc[...]` MMA operands to these source descriptors;
this SASS-only review does not claim a complete independent descriptor decode.

After PV issues at `0x4db0/0x4e00/0x4e70/0x5000`, it drains at2607/`0x5150`;
first numerator use follows at2609/`0x5160`, and backedge2675/`0x5370` returns
to `0x25a0`. Before PV issues at `0x4f80/0x5080/0x50c0/0x50d0`, drains at
2593/`0x50e0`, first numerator use follows at2595/`0x50f0`, and backedge2667/
`0x5330` returns to `0x2590`. In both static loop paths, slot reuse follows
the previous PV drain; the V publication moved around QK does not move a store
ahead of the preceding iteration's PV completion.

This admits only the SASS placement portion of the prospective compile gate.
It proves neither runtime correctness/race freedom nor achieved overlap,
occupancy, latency or superiority to FA3. No kernel, service, counter or new
compile was executed during this review; old artifacts were not edited.
