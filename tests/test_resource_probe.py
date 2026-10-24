"""CPU checks for diagnostic interpretation, no GPU measurements."""
import unittest
from projection_resource_probe import operand_requests,ptx_features


class ResourceInterpretation(unittest.TestCase):
    def test_logical_traffic_tradeoff(self):
        wide=operand_requests(128,4096,8192,32,128)
        narrow=operand_requests(128,4096,8192,32,32)
        self.assertEqual(wide['grid_ctas'],128)
        self.assertEqual(narrow['grid_ctas'],512)
        self.assertEqual(wide['w_logical_bytes'],256*1024**2)
        self.assertEqual(wide['x_logical_bytes'],64*1024**2)
        self.assertEqual(narrow['w_logical_bytes'],wide['w_logical_bytes'])
        self.assertEqual(narrow['x_logical_bytes'],4*wide['x_logical_bytes'])

    def test_comments_symbols_and_predicates(self):
        text='''// mma.sync.aligned.f32.bf16 ignored
            .visible .entry mma_sync_symbol()
            @%p1 ld.global.v4.b32 {%r1}, [%rd1];
            cp.async.ca.shared.global [%r1], [%rd1], 16;
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%r1}, {%r2};
            wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r1}, %rd1;
            @!%p2 st.local.b32 [%rd1], %r1;
        '''
        counts=ptx_features(text)['static_instruction_counts']
        for key in ['ld.global','cp.async','mma.sync','wgmma.mma_async','st.local']:
            self.assertEqual(counts[key],1,key)
        self.assertEqual(counts['ld.local'],0)
