"""Local upload-boundary and immutable-source tests; never imports Modal or CUDA."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_preflight as subject


class ModalPreparationTests(unittest.TestCase):
    def test_rejects_paths_outside_candidate_allowlist(self):
        for name in ["../submission.py", "/tmp/candidates/projection_x.py", "submission.py",
                     "candidates/router_pooler_hybrid.py", "candidates/projection_x.txt"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.describe(name)

    def test_rejects_uncommitted_bytes(self):
        with patch.object(subject.subprocess, "check_output", return_value=b"different bytes"):
            with self.assertRaisesRegex(ValueError, "Commit exact bytes"):
                subject.describe("candidates/projection_qkv_m1_simt.py")

    def test_bundle_is_explicit_and_hash_verified(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.py").write_text("# harmless source\n")
            raw = (root / "candidate.py").read_bytes()
            plan = {"files": {"candidate.py": {"sha256": hashlib.sha256(raw).hexdigest()}}}
            destination = root / "local-notes" / "bundle"
            with patch.object(subject, "ROOT", root):
                subject.build(plan, destination)
                self.assertEqual((destination / "candidate.py").read_bytes(), raw)
                self.assertEqual(json.loads((destination / "bundle-manifest.json").read_text()), plan)
                with self.assertRaises(FileExistsError):
                    subject.build(plan, destination)
                with self.assertRaisesRegex(ValueError, "NEW directory"):
                    subject.build(plan, root / "outside")
                (root / "candidate.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    subject.build(plan, root / "local-notes" / "changed")


if __name__ == "__main__":
    unittest.main()
