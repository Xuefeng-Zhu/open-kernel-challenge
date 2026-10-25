import unittest
from projection_p053_analysis import analyze, instruction_sites, logical_work, ptx_body


class AnalysisTests(unittest.TestCase):
    def test_valid_masked_work_and_no_hbm_claim(self):
        a,b=logical_work(8),logical_work(7)
        self.assertEqual(a["logical_weight_read_bytes"],64*1024**2)
        self.assertEqual(a["logical_activation_read_bytes"],8*1024**2)
        self.assertEqual(b["logical_activation_read_bytes"],7*1024**2)
        self.assertEqual(a["partial_write_bytes"],1024**2)
        self.assertEqual(b["partial_write_bytes"],7*1024**2//8)
        self.assertEqual(a["partial_epilogue_shared_read_plus_write_bytes"],4*1024**2)
        self.assertEqual(a["padded_tensor_flops"],b["padded_tensor_flops"])
        self.assertIn("not HBM",a["qualification"])

    def test_model_does_not_extend_to_unimplemented_tiles(self):
        for args in ((32,), (0,), (True,), (8,65,131)):
            with self.assertRaises(ValueError): logical_work(*args)

    def test_parser_distinguishes_data_copies_from_waits_and_debug(self):
        text="\ncp.async.cg.shared.global [a], [b], 16;\ncp.async.commit_group;\ncp.async.wait_group 0;\n@%p1 st.global.v4.b32 [a], {b};\n.loc 1 4 0\n\t.file 1 \"secret-not-copied\"\n.b8 4\n"
        r=instruction_sites(text)
        self.assertEqual(r["async_data_copy_sites"],1)
        self.assertEqual(r["async_commit_sites"],1)
        self.assertEqual(r["async_wait_sites"],1)
        self.assertEqual(r["global_vector4_stores"],1)
        self.assertNotIn("secret-not-copied",ptx_body(text))

    def test_bound_receipt_and_real_static_control(self):
        r=analyze()
        self.assertEqual(r["gpu_runs_added"],0)
        self.assertEqual([c["work"]["m"] for c in r["cases"]],[8,7])
        a=r["cases"][0]
        self.assertAlmostEqual(a["candidate_cut_to_match_p046_under_same_paired_statistic"],.1757178693614808)
        self.assertTrue(a["ptx"]["control"]["same_executable_text_as_p051"])
        self.assertTrue(a["ptx"]["weight_first"]["same_executable_text_as_p051"])
        self.assertEqual(a["ptx"]["control"]["async_data_copy_sites"],10)
        self.assertEqual(a["ptx"]["weight_first"]["async_data_copy_sites"],10)
        self.assertEqual(a["ptx"]["control"]["shared_vector4_stores"],2)
        self.assertEqual(a["ptx"]["weight_first"]["shared_vector2_stores"],4)


if __name__=="__main__": unittest.main()
