# Independent SASS scheduling review

Offline review of actual compile-only artifacts; no kernel execution, services,
source edit or additional compilation. Result SHA256
`8b5f8717c56de7a6ea6971840bd4f2252aced16c86e1fb3aca26ed6af1a859ea`.
Both SASS hashes match the artifact manifest:

- Serial: `63ef626f4e871b35d49d57bfa43740237383c850f1d87f3952fa903c1f2baadf`.
- Overlap: `d334f93fc988fcb5f6fc657cdffaa470e6502efef66b779c2f7dd7961156de06`.

**Finding: a real native widening/shared-store placement difference survives,
but the literal serial-before-all-gather gate does not.** The compiler hoists
page-table and some next-K payload loads in serial ahead of its QK wait. This
is not a fully serialized native control; do not silently relabel the original
gate as passed. Neither static placement nor the reported167/168 registers,
32768 shared bytes and zero spills establishes dynamic overlap or performance.

## Actual producer dataflow and wait boundary

Line numbers refer to the complete `artifacts/*.core.sass` files, including
the alternating encoding lines; PCs are the disassembler's byte offsets.

| Stage | Serial line / PC | Overlap line / PC |
|---|---|---|
| Current QK BF16 HGMMA group |1299/`0x2870`,1307/`0x28b0`,1313/`0x28e0`,1387/`0x2b30` |1301/`0x2880`,1309/`0x28c0`,1321/`0x2920`,1323/`0x2930` |
| Next page-table loads |1243/`0x26b0` onward, already before QK |1253/`0x2700` onward, already before QK |
| Example next-K payload load |1333/`0x2980`, **before** QK wait |1351/`0x2a10`, before QK wait |
| Example next-V payload load |1467/`0x2db0`, **after** QK wait |1461/`0x2d80`, before QK wait |
| First next-K BF16 shared store |1703/`0x3510`, after QK wait |1641/`0x3320`, before QK wait |
| Last next-V shared store |2365/`0x49c0`, after QK wait |1803/`0x3830`, before QK wait |
| Publication fence |2369/`0x49e0` |1807/`0x3850` |
| Corresponding QK drain |1455/`0x2d50` `WARPGROUP.DEPBAR.LE ...0` |1833/`0x3920` `WARPGROUP.DEPBAR.LE ...0` |

This is dataflow evidence, rather than mnemonic counts. In overlap, the
next-K value loaded into`R132` at`0x2a10` is unpacked from E4M3 at`0x2f30`
into`R73`;`0x2f40/0x2f60` widen its halves to BF16`R133/R76`;
`0x32d0` packs these into`R72`, which`0x3320` stores to the next shared
slot. For V,`0x2d80` loads`R80`;`0x3170` unpacks it into`R72`;
`0x3210/0x3230` widen to`R139/R140`;`0x35a0` packs these and`0x35d0`
stores them at the V-bank offset`+0x4000`. Both chains complete before
`0x3920`. In serial, the example K load at`0x2980` is not unpacked until
`0x2e40`, after its`0x2d50` wait; its BF16 packing/store is
`0x34d0`→`0x3510`. Thus serial already hides some K-load work, but not the
same completed K/V preparation.

## Shared slots and PV lifetime

Overlap initializes slot`UR4=0` at line1213/`0x25c0`. Current shared base is
formed at1283/`0x27f0` by`ULEA UR10,UR4,UR8,0xd`. After current QK issues,
1511/`0x2f10` flips slot bit0, and1513/`0x2f20` forms the other base
`UR7=UR8+(flipped_slot<<13)`. Next K stores at`0x3320..0x3570` use this
`UR7` base; next V stores at`0x35d0..0x3830` add`0x4000`. The slot bases
differ by8192 bytes; V occupies the separate16384-byte bank. Serial has the
same construction at`0x27e0`, slot flip`0x3330`, and other base`0x33e0`.

The printed HGMMA`gdesc[UR20]/[UR24]` operand notation does not yield a fully
independent descriptor-register binding from this SASS-only read. The parent's
IR/PTX review must bind those operands to the current K/V descriptors. The
base flip and shared-store destinations above are directly visible; they are
not a substitute for that binding or a runtime race proof.

Overlap PV issues at`0x4ec0/0x4f10/0x4f60/0x5110`, then drains at
2639/`0x5250`; first use of its result in numerator FFMA is2641/`0x5260`.
The backedge at2713/`0x54a0` returns to`0x2640`, after that drain and
the common numerator updates. Serial PV issues at`0x4ef0/0x4f40/0x4fb0/
0x5150`, drains2649/`0x52a0`, first numerator FFMA2651/`0x52b0`, and
loops at2717/`0x54c0`→`0x2630`. No later next-slot overwrite in either
iteration precedes the prior PV drain in this static control-flow path.

No correctness, race freedom, achieved overlap, occupancy, latency, or FA3
replacement is admitted by this review. The strict original serial-control
gate needs an explicit disposition by the parent; it is not fully met here.
