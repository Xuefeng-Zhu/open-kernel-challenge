"""Independent byte/address and causal-window models; no candidate import/GPU.

These CPU tests establish the proposed compaction geometry, not native lowering,
FA3 API acceptance, numerical accuracy or performance. Cache bytes are opaque:
scaling and FP8 interpretation are deliberately outside the copy operation.
"""
import hashlib
import random
import struct
import unittest


LENGTH = 8
WINDOW = 2048
CAPACITY = WINDOW - 1 + LENGTH
FEATURES = 2 * 64
BLOCK_TOKENS = 32


def compact_bounds(total):
    if total < LENGTH:
        raise ValueError('The live cache length includes the verify block')
    count = min(total, CAPACITY)
    return total - count, count


def original_allowed(history, query):
    """Absolute-position contract, independent of compaction formulas."""
    end = history + query
    return set(range(max(0, end - WINDOW + 1), end + 1))


def packed_allowed(count, query):
    """FA3 bottom-right causal/window alignment in compact coordinates."""
    aligned_query = query + count - LENGTH
    return set(range(max(0, aligned_query - WINDOW + 1), aligned_query + 1))


def fixture(histories, seed):
    """Disjoint shuffled pages, including a live physical row zero.

    Only rows actually read plus untouched sentinels need byte payloads. Page
    maps still include all historical positions, including the 80K case.
    """
    rng = random.Random(seed)
    totals = [h + LENGTH for h in histories]
    mapping = list(range(sum(totals)))
    rng.shuffle(mapping)
    # Force slot0 to a new-token row in request0 rather than leaving it unused.
    target = totals[0] - 1
    where_zero = mapping.index(0)
    mapping[target], mapping[where_zero] = mapping[where_zero], mapping[target]
    tables = []
    offset = 0
    for total in totals:
        tables.append(mapping[offset:offset + total])
        offset += total
    touched = set()
    for total, table in zip(totals, tables):
        start, count = compact_bounds(total)
        touched.update(table[start:start + count])
    sentinels = set(mapping) - touched
    # A few non-window rows are enough to check that original cache is read-only.
    touched.update(sorted(sentinels)[:9])
    k_cache = {}
    v_cache = {}
    for physical in touched:
        prefix = struct.pack('<I', physical)
        k_cache[physical] = prefix + rng.randbytes(FEATURES - len(prefix))
        v_cache[physical] = prefix[::-1] + rng.randbytes(FEATURES - len(prefix))
    return totals, tables, k_cache, v_cache


def copy_model(totals, tables, k_cache, v_cache, scratch=None):
    """Model a complete fixed-capacity copy, including zeroed padding.

    The byte loop intentionally differs from a TL pointer-expression model.
    It follows the logical page mapping first, then copies opaque whole rows.
    """
    if scratch is None:
        scratch = [[bytearray([0xA5]) * (CAPACITY * FEATURES),
                    bytearray([0x5A]) * (CAPACITY * FEATURES)] for _ in totals]
    lengths = []
    for request, (total, table) in enumerate(zip(totals, tables)):
        first, count = compact_bounds(total)
        lengths.append(count)
        for packed_position in range(CAPACITY):
            begin = packed_position * FEATURES
            end = begin + FEATURES
            if packed_position < count:
                physical = table[first + packed_position]
                scratch[request][0][begin:end] = k_cache[physical]
                scratch[request][1][begin:end] = v_cache[physical]
            else:
                scratch[request][0][begin:end] = bytes(FEATURES)
                scratch[request][1][begin:end] = bytes(FEATURES)
    return lengths, scratch


class WindowGeometry(unittest.TestCase):
    def test_every_query_matches_absolute_contract_at_boundaries(self):
        histories = [0, 1, 10, 31, 32, 33, 63, 64, 65,
                     2046, 2047, 2048, 2049, 2054, 2055, 80000, 1048576]
        for history in histories:
            first, count = compact_bounds(history + LENGTH)
            for query in range(LENGTH):
                with self.subTest(history=history, query=query):
                    actual = {first + i for i in packed_allowed(count, query)}
                    self.assertEqual(actual, original_allowed(history, query))
                    self.assertNotIn(history + query + 1, actual)

    def test_union_is_exact_not_merely_a_superset(self):
        for history in [0, 1, 2046, 2047, 2048, 2049, 80000]:
            wanted = set().union(*(original_allowed(history, q) for q in range(LENGTH)))
            first, count = compact_bounds(history + LENGTH)
            self.assertEqual(wanted, set(range(first, first + count)))
            self.assertEqual(count, min(history, WINDOW - 1) + LENGTH)

    def test_window_must_remain_enabled_after_compaction(self):
        history = 80000
        first, count = compact_bounds(history + LENGTH)
        # Plain causal attention would retain the first seven obsolete keys
        # for the final query. This is a regression detector, not an alternative.
        causal_only = set(range(count - 1 + 1))
        actual = packed_allowed(count, LENGTH - 1)
        self.assertEqual(len(causal_only - actual), LENGTH - 1)
        self.assertNotEqual({first + i for i in causal_only},
                            original_allowed(history, LENGTH - 1))

    def test_empty_history_has_only_live_causal_new_tokens(self):
        first, count = compact_bounds(LENGTH)
        self.assertEqual((first, count), (0, LENGTH))
        for query in range(LENGTH):
            self.assertEqual(packed_allowed(count, query), set(range(query + 1)))

    def test_final_partial_block_has_seven_live_capacity_rows(self):
        self.assertEqual(CAPACITY, 2055)
        self.assertEqual(CAPACITY % BLOCK_TOKENS, 7)
        rows = [block * BLOCK_TOKENS + lane
                for block in range((CAPACITY + BLOCK_TOKENS - 1) // BLOCK_TOKENS)
                for lane in range(BLOCK_TOKENS)
                if block * BLOCK_TOKENS + lane < CAPACITY]
        self.assertEqual(rows, list(range(CAPACITY)))

    def test_metadata_contract_rejects_missing_verify_tokens(self):
        for total in [-1, 0, 1, 7]:
            with self.assertRaises(ValueError):
                compact_bounds(total)


class ByteAndAddressGeometry(unittest.TestCase):
    def test_shuffled_page_copy_and_live_zero_are_byte_exact(self):
        histories = [0, 1, 2047, 2048, 2049, 80000]
        totals, tables, kc, vc = fixture(histories, 127301)
        original_k, original_v = dict(kc), dict(vc)
        lengths, scratch = copy_model(totals, tables, kc, vc)
        self.assertEqual(tables[0][-1], 0)
        for b, total in enumerate(totals):
            first, count = compact_bounds(total)
            self.assertEqual(lengths[b], count)
            self.assertEqual(struct.unpack('<i', struct.pack('<i', count))[0], count)
            for offset in range(count):
                begin, end = offset * FEATURES, (offset + 1) * FEATURES
                physical = tables[b][first + offset]
                self.assertEqual(bytes(scratch[b][0][begin:end]), kc[physical])
                self.assertEqual(bytes(scratch[b][1][begin:end]), vc[physical])
            self.assertEqual(set(scratch[b][0][count * FEATURES:]), {0} if count < CAPACITY else set())
            self.assertEqual(set(scratch[b][1][count * FEATURES:]), {0} if count < CAPACITY else set())
        self.assertEqual(kc, original_k)
        self.assertEqual(vc, original_v)

    def test_opaque_fp8_encodings_and_scales_are_never_requantized(self):
        totals, tables, kc, vc = fixture([2049], 127302)
        for physical in kc:
            kc[physical] = bytes(range(128))
            vc[physical] = bytes(range(128, 256))
        outputs = []
        for scale in [1.0, 1.0 / 64]:
            # Scale belongs to FA3 descale, not this opaque byte copy.
            lengths, scratch = copy_model(totals, tables, kc, vc)
            outputs.append((lengths, bytes(scratch[0][0]), bytes(scratch[0][1])))
            self.assertEqual(scratch[0][0][:FEATURES], bytes(range(128)))
            self.assertEqual(scratch[0][1][:FEATURES], bytes(range(128, 256)))
        self.assertEqual(outputs[0], outputs[1])

    def test_allocation_write_addresses_are_unique_and_complete(self):
        batch = 5
        scalar_addresses = [((b * CAPACITY + position) * FEATURES + feature)
                            for b in range(batch)
                            for position in range(CAPACITY)
                            for feature in range(FEATURES)]
        self.assertEqual(len(set(scalar_addresses)), batch * CAPACITY * FEATURES)
        self.assertEqual(min(scalar_addresses), 0)
        self.assertEqual(max(scalar_addresses), batch * CAPACITY * FEATURES - 1)
        self.assertEqual(len(scalar_addresses), max(scalar_addresses) + 1)

    def test_return_A_metadata_mutation_rewrites_same_scratch(self):
        a = fixture([80000, 2047, 0], 127303)
        b = fixture([1, 2049, 10], 127304)
        la, scratch = copy_model(*a)
        handles = [(id(pair[0]), id(pair[1])) for pair in scratch]
        digest_a = [(hashlib.sha256(pair[0]).hexdigest(), hashlib.sha256(pair[1]).hexdigest())
                    for pair in scratch]
        lb, scratch = copy_model(*b, scratch=scratch)
        self.assertNotEqual(la, lb)
        # The long A content must not survive the shorter B metadata.
        for count, pair in zip(lb, scratch):
            self.assertEqual(pair[0][count * FEATURES:], bytes((CAPACITY - count) * FEATURES))
            self.assertEqual(pair[1][count * FEATURES:], bytes((CAPACITY - count) * FEATURES))
        la2, scratch = copy_model(*a, scratch=scratch)
        self.assertEqual(la2, la)
        self.assertEqual(handles, [(id(pair[0]), id(pair[1])) for pair in scratch])
        self.assertEqual(digest_a, [(hashlib.sha256(pair[0]).hexdigest(), hashlib.sha256(pair[1]).hexdigest())
                                    for pair in scratch])

    def test_batch_requests_do_not_share_page_mapping_or_scratch(self):
        totals, tables, kc, vc = fixture([10, 2048, 80000], 127305)
        self.assertTrue(all(not (set(tables[i]) & set(tables[j]))
                            for i in range(len(tables)) for j in range(i)))
        lengths, scratch = copy_model(totals, tables, kc, vc)
        self.assertEqual(len({id(part) for pair in scratch for part in pair}), 6)
        snapshot = bytes(scratch[1][0])
        scratch[0][0][0] ^= 0xFF
        self.assertEqual(bytes(scratch[1][0]), snapshot)


if __name__ == '__main__':
    unittest.main()
