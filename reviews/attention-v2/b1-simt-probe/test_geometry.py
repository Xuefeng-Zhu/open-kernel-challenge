"""CPU-only contract tests for the isolated B1/L1/G8 sliding SIMT geometry.

These tests never import or execute candidate.py (or torch/Triton).  The model
enumerates token/address ownership independently of attention arithmetic; a few
AST assertions bind its critical formulas to the actual source.  Passing is not
evidence of GPU compilation, numerical accuracy, or performance.
"""

import ast
from collections import Counter
from pathlib import Path
import random
import unittest


SOURCE = Path(__file__).with_name("candidate.py")
SEQUENCES = (0, 1, 10, 63, 64, 127, 128, 2047, 2048, 2049, 80000)
WINDOW, SPLITS, TILE = 2048, 16, 64
BATCH, LENGTH, GROUP, KV_HEADS, Q_HEADS, DIM, PARTIAL_ROWS = 1, 1, 8, 2, 16, 64, 16


def ceil_div(n, d):
    return (n + d - 1) // d


def split_tiles(seq):
    """Model physical tiles; the specification below independently supplies keys."""
    lo = max(seq - WINDOW + 1, 0)
    per = ceil_div(ceil_div(seq - lo, SPLITS), TILE) * TILE
    result = []
    for split in range(SPLITS):
        start, stop = lo + split * per, min(lo + (split + 1) * per, seq)
        tiles = [tuple(t for t in range(base, base + TILE) if t < stop)
                 for base in range(start, stop, TILE)]
        result.append(tiles)
    return result


def visible_history(seq):
    # Independent direct causal/window predicate, not the partition expression.
    return tuple(t for t in range(max(0, seq - WINDOW - 2), seq)
                 if seq - t < WINDOW)


def shuffled_pages(seq, page, seed):
    count = seq // page + 1  # Include the new token's logical page.
    pages = list(range(count))
    random.Random(seed).shuffle(pages)
    return pages


def physical_rows(pages, page):
    # Reference representation: concatenate whole physical pages in logical order.
    return tuple(row for physical_page in pages
                 for row in range(physical_page * page, (physical_page + 1) * page))


def tiled_address_reads(seq, page, pages):
    """Only active lanes may index the table; record logical-page reads as well."""
    reads, rows = [], []
    for tiles in split_tiles(seq):
        for tile in tiles:
            for token in tile:
                logical_page = token // page
                if not 0 <= logical_page < len(pages):
                    raise AssertionError("active token indexed outside page table")
                reads.append(logical_page)
                rows.append(pages[logical_page] * page + token % page)
    return reads, rows


def partial_row(kvh, split, query_row):
    return (kvh * SPLITS + split) * PARTIAL_ROWS + query_row


def materialize_partials():
    poison = object()
    rows = BATCH * KV_HEADS * SPLITS * PARTIAL_ROWS
    numer = [poison] * (rows * DIM)
    maxima, denom = [poison] * rows, [poison] * rows
    writes = Counter()
    for kvh in range(KV_HEADS):
        for split in range(SPLITS):
            for g in range(GROUP):
                row = partial_row(kvh, split, g)
                maxima[row], denom[row] = ("m", kvh, split, g), ("l", kvh, split, g)
                for d in range(DIM):
                    off = row * DIM + d
                    numer[off] = (kvh, split, g, d)
                    writes[off] += 1
    return poison, numer, maxima, denom, writes


def normalized_expr(text):
    return ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)


def assigned_expressions(function, name):
    values = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            values.append(ast.dump(node.value, include_attributes=False))
    return values


class SlidingPartitionTests(unittest.TestCase):
    def test_direct_window_includes_new_token_and_exact_history(self):
        for seq in SEQUENCES:
            with self.subTest(seq=seq):
                history = visible_history(seq)
                expected_lo = max(0, seq + 1 - WINDOW)
                self.assertEqual(history, tuple(range(expected_lo, seq)))
                self.assertEqual(len(history) + 1, min(seq + 1, WINDOW))
                self.assertEqual(history + (seq,), tuple(range(expected_lo, seq + 1)))
                self.assertNotIn(expected_lo - 1, history)

    def test_every_historical_token_has_exactly_one_split_owner(self):
        for seq in SEQUENCES:
            with self.subTest(seq=seq):
                ownership = Counter(token for tiles in split_tiles(seq)
                                    for tile in tiles for token in tile)
                self.assertEqual(ownership, Counter(visible_history(seq)))
                self.assertTrue(all(count == 1 for count in ownership.values()))
                self.assertNotIn(seq, ownership)

    def test_tile_alignment_capacity_and_tail_lengths(self):
        # Independent hard boundaries catch off-by-one/rounding errors at 64/128.
        expected_per = {0: 0, 1: 64, 10: 64, 63: 64, 64: 64, 127: 64,
                        128: 64, 2047: 128, 2048: 128, 2049: 128, 80000: 128}
        for seq, per in expected_per.items():
            with self.subTest(seq=seq):
                tiles = split_tiles(seq)
                counts = [sum(map(len, split)) for split in tiles]
                span = len(visible_history(seq))
                self.assertLessEqual(span, SPLITS * per)
                self.assertTrue(all(count <= per for count in counts))
                self.assertTrue(all(0 < len(tile) <= TILE for split in tiles for tile in split))
                self.assertEqual(sum(counts), span)
                if span:
                    last = max(i for i, count in enumerate(counts) if count)
                    self.assertEqual(counts[:last], [per] * last)
                    self.assertEqual(counts[last], span - last * per)

    def test_last_split_new_token_ownership_is_independent_of_history(self):
        for seq in SEQUENCES:
            with self.subTest(seq=seq):
                writers = [(split, kvh) for split in range(SPLITS)
                           for kvh in range(KV_HEADS) if split == SPLITS - 1]
                self.assertEqual(writers, [(15, 0), (15, 1)])
                for kvh in range(KV_HEADS):
                    self.assertEqual(sum(head == kvh for _, head in writers), 1)
                all_keys = [token for tiles in split_tiles(seq) for tile in tiles for token in tile]
                all_keys.append(seq)
                self.assertEqual(Counter(all_keys), Counter(visible_history(seq) + (seq,)))
        for seq in (0, 1, 10, 63, 64, 127, 128):
            self.assertEqual(split_tiles(seq)[15], [])

    def test_masked_tile_padding_does_not_read_page_table(self):
        # seq=1 has a 64-lane tile, but only logical page 0 is safe for page1.
        reads, rows = tiled_address_reads(1, 1, [0])
        self.assertEqual(reads, [0])
        self.assertEqual(rows, [0])
        self.assertEqual(tiled_address_reads(0, 1, []), ([], []))


class PageAndHeadAddressTests(unittest.TestCase):
    def test_shuffled_page1_history_addresses_match_flattened_pages(self):
        for seq in SEQUENCES:
            with self.subTest(seq=seq):
                pages = shuffled_pages(seq, 1, 2701 + seq)
                _, rows = tiled_address_reads(seq, 1, pages)
                reference = physical_rows(pages, 1)
                self.assertEqual(rows, [reference[t] for t in visible_history(seq)])
                self.assertEqual(len(rows), len(set(rows)))
                self.assertIn(0, pages)

    def test_shuffled_page64_addresses_preserve_intra_page_offsets(self):
        for seq in SEQUENCES:
            with self.subTest(seq=seq):
                pages = shuffled_pages(seq, 64, 6803 + seq)
                _, rows = tiled_address_reads(seq, 64, pages)
                reference = physical_rows(pages, 64)
                self.assertEqual(rows, [reference[t] for t in visible_history(seq)])
                self.assertEqual([row % 64 for row in rows], [t % 64 for t in visible_history(seq)])
                self.assertEqual(len(rows), len(set(rows)))
                self.assertIn(0, pages)

    def test_slot_zero_is_a_valid_historical_row_and_new_cache_destination(self):
        for page in (1, 64):
            # Explicitly place the first visible logical page at physical page 0.
            seq = 128
            pages = shuffled_pages(seq, page, 11)
            first_page = visible_history(seq)[0] // page
            zero_at = pages.index(0)
            pages[first_page], pages[zero_at] = pages[zero_at], pages[first_page]
            _, rows = tiled_address_reads(seq, page, pages)
            self.assertIn(0, rows)
            # New token lies at offset zero here; it is not a skip/sentinel row.
            logical_new = seq // page
            zero_at = pages.index(0)
            pages[logical_new], pages[zero_at] = pages[zero_at], pages[logical_new]
            loc = pages[logical_new] * page + seq % page
            self.assertEqual(loc, 0)
            writers = Counter(loc * KV_HEADS * DIM + kvh * DIM + d
                              for split in range(SPLITS) if split == SPLITS - 1
                              for kvh in range(KV_HEADS) for d in range(DIM))
            self.assertEqual(writers, Counter(range(KV_HEADS * DIM)))

    def test_new_cache_writers_are_unique_and_do_not_overwrite_history(self):
        for page in (1, 64):
            for seq in SEQUENCES:
                with self.subTest(page=page, seq=seq):
                    pages = shuffled_pages(seq, page, 13003 + seq)
                    flat = physical_rows(pages, page)
                    loc = flat[seq]
                    writes = Counter((loc * KV_HEADS + kvh) * DIM + d
                                     for split in range(SPLITS) if split == SPLITS - 1
                                     for kvh in range(KV_HEADS) for d in range(DIM))
                    self.assertEqual(len(writes), KV_HEADS * DIM)
                    self.assertEqual(set(writes.values()), {1})
                    historical = {flat[t] for t in visible_history(seq)}
                    self.assertNotIn(loc, historical)
                    self.assertTrue(all(0 <= off < len(pages) * page * KV_HEADS * DIM
                                        for off in writes))

    def test_l1_g8_query_mapping_covers_all_sixteen_heads_once(self):
        mapping = [(kvh, row, row // GROUP, kvh * GROUP + row % GROUP)
                   for kvh in range(KV_HEADS) for row in range(GROUP)]
        self.assertEqual(Counter(head for _, _, _, head in mapping), Counter(range(Q_HEADS)))
        self.assertEqual({j for _, _, j, _ in mapping}, {0})
        offsets = [head * DIM + d for _, _, _, head in mapping for d in range(DIM)]
        self.assertEqual(sorted(offsets), list(range(Q_HEADS * DIM)))


class PartialLayoutTests(unittest.TestCase):
    def test_producer_valid_rows_are_disjoint_with_original_stride_sixteen(self):
        _, _, _, _, writes = materialize_partials()
        self.assertEqual(len(writes), BATCH * KV_HEADS * SPLITS * GROUP * DIM)
        self.assertEqual(set(writes.values()), {1})
        valid = {partial_row(kvh, split, g) for kvh in range(KV_HEADS)
                 for split in range(SPLITS) for g in range(GROUP)}
        padded = {partial_row(kvh, split, g) for kvh in range(KV_HEADS)
                  for split in range(SPLITS) for g in range(GROUP, PARTIAL_ROWS)}
        self.assertTrue(valid.isdisjoint(padded))
        self.assertEqual(len(valid), 256)
        self.assertEqual(len(padded), 256)

    def test_merge_reads_every_valid_partial_and_never_padded_poison(self):
        poison, numer, maxima, denom, writes = materialize_partials()
        reads = Counter()
        for head in range(Q_HEADS):
            kvh, g = divmod(head, GROUP)
            for split in range(SPLITS):
                row = ((kvh * SPLITS + split) * PARTIAL_ROWS) + g
                self.assertEqual(maxima[row], ("m", kvh, split, g))
                self.assertEqual(denom[row], ("l", kvh, split, g))
                for d in range(DIM):
                    off = row * DIM + d
                    self.assertIsNot(numer[off], poison)
                    self.assertEqual(numer[off], (kvh, split, g, d))
                    reads[off] += 1
        self.assertEqual(reads, writes)
        for kvh in range(KV_HEADS):
            for split in range(SPLITS):
                for g in range(GROUP, PARTIAL_ROWS):
                    row = partial_row(kvh, split, g)
                    self.assertIs(maxima[row], poison)
                    self.assertIs(denom[row], poison)
                    self.assertTrue(all(value is poison for value in numer[row * DIM:(row + 1) * DIM]))

    def test_stride_eight_regression_would_read_poison(self):
        # Sensitivity check: a compact stride is incompatible with old merge.
        poison, numer, _, _, _ = materialize_partials()
        wrong_reads = [numer[((kvh * SPLITS + split) * GROUP + g) * DIM]
                       for kvh in range(KV_HEADS) for split in range(SPLITS)
                       for g in range(GROUP)]
        self.assertTrue(any(value is poison for value in wrong_reads))

    def test_partial_allocation_and_valid_traffic_bytes(self):
        allocated_rows = BATCH * KV_HEADS * SPLITS * PARTIAL_ROWS
        active_rows = BATCH * KV_HEADS * SPLITS * GROUP
        self.assertEqual(allocated_rows * (DIM + 2) * 4, 135168)
        self.assertEqual(active_rows * (DIM + 2) * 4, 67584)
        poison, numer, maxima, denom, writes = materialize_partials()
        self.assertEqual((len(numer) + len(maxima) + len(denom)) * 4, 135168)
        actual_scalars = len(writes) + sum(x is not poison for x in maxima) + sum(x is not poison for x in denom)
        self.assertEqual(actual_scalars * 4, 67584)  # Producer writes; same valid merge reads.


class CandidateFormulaBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.functions = {node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)}

    def assert_assignment(self, function_name, name, expression):
        self.assertIn(normalized_expr(expression), assigned_expressions(self.functions[function_name], name))

    def test_window_partition_and_last_split_formulas_match_original_producer(self):
        formulas = {
            "lo": ("tl.maximum(seq - window + 1, 0)", "tl.minimum(lo, seq)"),
            "span": ("seq - lo",),
            "per": ("((span + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N",),
            "s_lo": ("lo + split * per",),
            "s_hi": ("tl.minimum(s_lo + per, seq)",),
            "t_ok": ("t < s_hi",),
            "vis": ("vis & (t[None, :] > qpos[:, None] - window)",),
            "row": ("pg * page + t % page",),
            "kv_off": ("row[:, None] * (HKV * D) + kvh * D + offs_d[None, :]",),
            "dst": ("loc[:, None] * (HKV * D) + kvh * D + offs_d[None, :]",),
        }
        for name, expressions in formulas.items():
            for expression in expressions:
                for function in ("_fused", "_fused_b1_simt"):
                    with self.subTest(function=function, variable=name):
                        self.assert_assignment(function, name, expression)
        for function in ("_fused", "_fused_b1_simt"):
            owners = [node for node in ast.walk(self.functions[function]) if isinstance(node, ast.If)
                      and ast.dump(node.test, include_attributes=False) == normalized_expr("split == SPLITS - 1")]
            self.assertEqual(len(owners), 1)
            # The writer guard is not nested inside the historical loop/nonempty test.
            self.assertIn(owners[0], self.functions[function].body)
            cache_stores = [call for call in ast.walk(owners[0]) if isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute) and call.func.attr == "store"]
            self.assertEqual(len(cache_stores), 2)

    def test_query_and_partial_producer_consumer_address_formulas(self):
        for name, expression in {
            "row_ok": "offs_m < L * G", "j": "tl.where(row_ok, offs_m // G, 0)",
            "g": "offs_m % G", "head": "kvh * G + g",
            "pbase": "((b * HKV + kvh) * SPLITS + split) * PARTIAL_M + offs_m",
        }.items():
            self.assert_assignment("_fused_b1_simt", name, expression)
        for name, expression in {"kvh": "head // G", "g": "head % G",
                                 "qb": "((b * HKV + kvh) * SPLITS + offs_s) * BLOCK_M + g"}.items():
            self.assert_assignment("_merge", name, expression)


if __name__ == "__main__":
    unittest.main()
