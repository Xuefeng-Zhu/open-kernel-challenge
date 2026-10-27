"""CPU representation tests only; no native decoder/throughput evidence."""
import unittest
import torch

from projection_weight_codec import encode_weights, decode_weights, storage_report, BLOCK_K


def from_bits(values, shape):
    return torch.tensor(values, dtype=torch.int32).to(torch.int16).reshape(shape).view(torch.bfloat16)


def independent_native_tile_bits(encoded, n_start, k_start):
    """CPU address model, independent of decode_weights; not Triton execution.

    Mirrors BN64/BK64 W tiles for a BM16/SPLIT8 producer. BM indexes X/output,
    so it deliberately does not enter the W address calculation. Metadata has
    already been transposed to [NB,N], unlike the canonical codec serialization.
    """
    n, k = encoded['original_shape']
    nb = (k+255)//256
    sign = encoded['sign_mantissa'].reshape(n,nb*256).flatten()
    delta = encoded['exponent_delta'].reshape(n,nb*128).flatten()
    bases = encoded['exponent_base'].t().contiguous().flatten()
    indices = encoded['escape_index'].t().contiguous().flatten()
    escape = encoded['escape_exponents']
    # Native preparation needs a valid pointer even when every masked load is off.
    if escape.shape[0] == 0:
        escape = torch.zeros((1,256),dtype=torch.uint8)
    escape = escape.flatten()
    ns = n_start+torch.arange(64)
    ks = k_start+torch.arange(64)
    valid = (ns[:,None] < n) & (ks[None,:] < k)
    safe_n = torch.where(ns < n,ns,0)[:,None]
    safe_k = torch.where(ks < k,ks,0)[None,:]
    block, within = safe_k//256, safe_k%256
    sm = sign[safe_n*(nb*256)+safe_k].to(torch.int32)
    packed = delta[safe_n*(nb*128)+block*128+within//2].to(torch.int32)
    metadata_offset = block*n+safe_n
    base = bases[metadata_offset].to(torch.int32)
    escape_id = indices[metadata_offset].long()
    low = (packed >> ((within%2)*4)) & 15
    full = escape[escape_id.clamp_min(0)*256+within].to(torch.int32)
    exponent = torch.where(escape_id >= 0,full,base+low)
    bits = (sm & 127) | ((sm & 128)<<8) | (exponent<<7)
    return torch.where(valid,bits,0), valid


class WeightCodecTests(unittest.TestCase):
    def roundtrip(self, weights):
        original = weights.contiguous().view(torch.int16).clone()
        encoded = encode_weights(weights)
        restored = decode_weights(encoded)
        self.assertEqual(restored.shape, weights.shape)
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(restored.view(torch.int16), original))
        self.assertTrue(torch.equal(weights.contiguous().view(torch.int16), original))
        return encoded

    def test_all_bf16_bit_patterns(self):
        weights = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256,256)
        normal = self.roundtrip(weights)
        self.assertEqual(normal['escape_exponents'].shape[0], 0)
        escaped = self.roundtrip(weights.t())
        self.assertEqual(escaped['escape_exponents'].shape[0], 256)

    def test_special_patterns_and_wide_exponent_escape(self):
        # Includes both signed zeros, subnormals, infinities, quiet/signaling NaN payloads.
        bits = [0x0000,0x8000,0x0001,0x007f,0x8001,0x807f,0x7f80,0xff80,
                0x7fc1,0x7f81,0xffa3,0x3f80,0xbf80,0x7f7f,0xff7f]
        encoded = self.roundtrip(from_bits(bits, (3,5)))
        self.assertGreater(encoded['escape_exponents'].shape[0], 0)

    def test_exact_span_boundary(self):
        for span, expected_escapes in [(15,0),(16,1)]:
            values = [(100<<7)|3, ((100+span)<<7)|77]
            encoded = self.roundtrip(from_bits(values,(1,2)))
            self.assertEqual(encoded['escape_exponents'].shape[0], expected_escapes)

    def test_pair_order_and_sign_fraction_encoding(self):
        encoded = self.roundtrip(from_bits([(100<<7)|5, (103<<7)|0x8000|7],(1,2)))
        self.assertEqual(encoded['exponent_delta'][0,0,0].item(), 0x30)
        self.assertEqual(encoded['sign_mantissa'][0,0,:2].tolist(), [5,135])

    def test_odd_dimensions_padding_and_noncontiguous_view(self):
        for n,k in [(1,1),(3,255),(5,257),(3,513)]:
            gen=torch.Generator().manual_seed(n*1000+k)
            values=torch.randn((n,k*2),generator=gen).bfloat16()[:,::2]
            encoded=self.roundtrip(values)
            self.assertEqual(encoded['sign_mantissa'].shape, (n,(k+255)//256,256))

    def test_fresh_random_weights_and_repeat(self):
        previous = None
        for seed in (5151,5152,7171):
            gen=torch.Generator().manual_seed(seed)
            weights=torch.randn((9,1025),generator=gen).bfloat16()
            encoded=self.roundtrip(weights)
            repeat=encode_weights(weights)
            for key in ('sign_mantissa','exponent_delta','exponent_base','escape_index','escape_exponents'):
                self.assertTrue(torch.equal(encoded[key],repeat[key]))
            if previous is not None:
                self.assertFalse(torch.equal(previous,encoded['sign_mantissa']))
            previous=encoded['sign_mantissa']

    def test_all_streams_and_header_count(self):
        encoded=self.roundtrip(torch.ones((2,BLOCK_K),dtype=torch.bfloat16))
        result=storage_report(encoded)
        expected=2*256 + 2*128 + 2*1 + 2*4
        self.assertEqual(result['tensor_bytes'], expected)
        self.assertEqual(result['encoded_bytes'],expected+32)
        self.assertEqual(result['original_bytes'],2*256*2)
        self.assertEqual(result['escape_blocks'],0)

    def test_wrong_type_empty_and_corrupt_stream_refused(self):
        for weight in (torch.ones((1,3)),torch.ones((2,),dtype=torch.bfloat16),
                       torch.empty((1,0),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError): encode_weights(weight)
        encoded=encode_weights(torch.ones((2,257),dtype=torch.bfloat16))
        encoded['escape_index'][0,0]=0
        with self.assertRaisesRegex(ValueError,'Escape index'):
            decode_weights(encoded)

    def test_independent_flattened_native_tile_addresses(self):
        all_bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256,256)
        gen=torch.Generator().manual_seed(43043)
        odd=torch.randint(0,65536,(67,513),generator=gen,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        # Normal and escape representations each cover every possible BF16 pattern.
        # Odd shapes exercise metadata strides, partial N/K tiles, and empty splits.
        for weights in (all_bits,all_bits.t(),odd,torch.ones((3,1),dtype=torch.bfloat16)):
            with self.subTest(shape=tuple(weights.shape),transpose=not weights.is_contiguous()):
                encoded=encode_weights(weights)
                n,k=weights.shape
                expected=weights.contiguous().view(torch.int16).to(torch.int32)&65535
                reconstructed=torch.zeros((n,k),dtype=torch.int32)
                visits=torch.zeros((n,k),dtype=torch.int32)
                part_k=((k+8*64-1)//(8*64))*64
                for split in range(8):
                    for step in range(0,part_k,64):
                        start_k=split*part_k+step
                        for start_n in range(0,n,64):
                            bits,valid=independent_native_tile_bits(encoded,start_n,start_k)
                            self.assertTrue(torch.all(bits[~valid]==0))
                            width=max(0,min(64,k-start_k))
                            height=min(64,n-start_n)
                            if width:
                                reconstructed[start_n:start_n+height,start_k:start_k+width]=bits[:height,:width]
                                visits[start_n:start_n+height,start_k:start_k+width]+=1
                self.assertTrue(torch.equal(reconstructed,expected))
                self.assertTrue(torch.all(visits==1))


if __name__ == '__main__':
    unittest.main()
