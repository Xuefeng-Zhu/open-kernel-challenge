"""Local-only transport, budget, source and launch-boundary tests."""
import ast
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import projection_p044_modal as p


class LauncherTests(unittest.TestCase):
    def preflight(self, stamp=None):
        return dict(observed_utc=(stamp or datetime.now(timezone.utc)).isoformat(),
                    workspace=p.PROFILE, gpu_slot_granted=True,
                    verified_credit_balance_usd="28.00", verified_net_spend_cap_usd="0",
                    total_spend_usd="0", total_usage_usd="2.00")

    def transport(self, files=None):
        files = files or {"summary.json": b'{"complete":true}', "timing.json": b'{}'}
        raw = json.dumps({n: base64.b64encode(v).decode() for n, v in files.items()}, sort_keys=True).encode()
        packed = zlib.compress(raw)
        chunks = [packed[i:i+p.CHUNK] for i in range(0, len(packed), p.CHUNK)]
        manifest = dict(chunks=len(chunks), packed_bytes=len(packed), encoded_bytes=len(raw),
                        artifact_bytes=sum(map(len, files.values())),
                        sha256=hashlib.sha256(packed).hexdigest(),
                        artifact_sha256={n: hashlib.sha256(v).hexdigest() for n, v in files.items()})
        return manifest, chunks

    def test_reserve_includes_headroom_and_current_higher_rate(self):
        self.assertEqual(p.reserve(), Decimal(".709018880"))
        self.assertEqual(p.reserve(Decimal(".002")), Decimal("1.0760"))
        self.assertEqual(p.reserve(Decimal(".0001")), p.reserve())

    def test_rate_and_gross_never_lower_accounting(self):
        rates = dict(gpu_hour_cost_h200="4.5", cpu_hour_cost=".05", mem_gib_hour_cost=".01")
        self.assertEqual(p.resource_rate(rates), Decimal("4.68") / 3600)
        self.assertEqual(p.gross(dict(metered_cost="1.20", metered_cost_breakdown={"apps":"1.24"})), Decimal("1.24"))

    def test_budget_requires_credits_after_delayed_usage_and_no_paid_spend(self):
        rates = dict(gpu_hour_cost_h200="4.5", cpu_hour_cost=".05", mem_gib_hour_cost=".01")
        billing = dict(billed_cost="0", metered_cost="2.01", metered_cost_breakdown={"apps":"2.20"})
        self.assertEqual(p.budget_guard(self.preflight(), billing, rates)[2], Decimal("27.80"))
        for pre, bill in (({**self.preflight(), "verified_credit_balance_usd":".90"}, billing),
                          (self.preflight(), {**billing, "billed_cost":".01"})):
            with self.assertRaises(ValueError): p.budget_guard(pre, bill, rates)

    def test_fresh_preflight_and_missing_permission_fail_closed(self):
        now = datetime.now(timezone.utc)
        p.validate_preflight(self.preflight(now), now)
        for key, value in (("gpu_slot_granted", False), ("workspace", "other"),
                           ("verified_net_spend_cap_usd", "1"), ("total_spend_usd", ".01"),
                           ("verified_credit_balance_usd", "0")):
            data = self.preflight(now); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_preflight(data, now)
        for delta in (-900, 6):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                p.validate_preflight(self.preflight(now + timedelta(seconds=delta)), now)

    def test_auth_file_only_metadata_is_inspected(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "config"
            f.write_text("dummy")
            f.chmod(0o600)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("secret bytes read")), \
                 patch.object(Path, "read_text", side_effect=AssertionError("secret text read")):
                p.check_owned_file(f)
            f.chmod(0o644)
            with self.assertRaises(ValueError): p.check_owned_file(f)
            link = Path(temp) / "link"; link.symlink_to(f)
            with self.assertRaises(ValueError): p.check_owned_file(link)

    def test_expired_auth_metadata_refused_without_reading_auth_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); notes = root / "local-notes"; notes.mkdir()
            auth = notes / "modal-auth.toml"; auth.write_text("not a credential"); auth.chmod(0o600)
            (notes / "modal-login-20261025-1536.local.json").write_text(json.dumps(dict(
                state="verified", verification=dict(expires_utc="2000-01-01 00:00:00 UTC"))))
            read = Path.read_text
            def guarded(path, *a, **kw):
                if path.name == "modal-auth.toml": raise AssertionError("credential contents read")
                return read(path, *a, **kw)
            with patch.object(p.subprocess, "run"), patch.object(p.subprocess, "check_output", return_value=b""), \
                 patch.object(Path, "read_text", guarded), patch.dict(p.os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "expiry"):
                    p.configure_existing_auth(root)

    def test_payload_is_exact_and_excludes_control_data(self):
        self.assertEqual(len(p.PAYLOAD), 13)
        self.assertIn(p.SOURCE, p.PAYLOAD)
        self.assertIn("official/baseline.py", p.PAYLOAD)
        self.assertIn("licenses/June-9-Researcher-Reciprocity.txt", p.PAYLOAD)
        self.assertFalse(any("local-notes" in n or "auth" in n or "config" in n for n in p.PAYLOAD))
        self.assertEqual(hashlib.sha256((p.ROOT/p.SOURCE).read_bytes()).hexdigest(), p.SOURCE_SHA256)

    def test_changed_source_or_head_refused(self):
        with self.assertRaises(ValueError): p.frozen_payload("main")
        with patch.object(p.subprocess, "check_output", return_value="0" * 40 + "\n"):
            with self.assertRaises(ValueError): p.frozen_payload("1" * 40)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (*p.PAYLOAD, "projection_p044_modal.py"):
                file = root/name; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b"changed")
            def fake_git(cmd, **kwargs):
                return "1" * 40 + "\n" if cmd[1] == "rev-parse" else b"changed"
            with patch.object(p.subprocess, "check_output", side_effect=fake_git), \
                 self.assertRaisesRegex(ValueError, "P039 source mismatch"):
                p.frozen_payload("1" * 40, root)

    def test_existing_record_prevents_all_preflight_or_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"experiments").mkdir()
            (root/"experiments/projection-p044-modal.json").write_text('{}')
            with patch.object(p, "ROOT", root), patch.object(p, "frozen_payload") as freeze, \
                 patch.object(p.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "reconcile"):
                p.launch("1" * 40, root)
            freeze.assert_not_called(); run.assert_not_called()

    def test_transport_roundtrip_and_no_replacement(self):
        manifest, chunks = self.transport()
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "artifacts"
            self.assertEqual(p.unpack_artifacts(manifest, chunks, dest), {"complete": True})
            self.assertEqual((dest/"timing.json").read_bytes(), b'{}')
            with self.assertRaises(ValueError): p.unpack_artifacts(manifest, chunks, dest)

    def test_transport_hash_truncation_and_size_guards(self):
        manifest, chunks = self.transport()
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "artifacts"
            for changes, sample in (({"sha256":"0"*64}, chunks), ({}, []),
                                    ({"packed_bytes":p.MAX_PACKED+1}, chunks),
                                    ({"artifact_bytes":p.MAX_RAW+1}, chunks)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.unpack_artifacts({**manifest, **changes}, sample, dest)
            self.assertFalse(dest.exists())
            bad = {**manifest, "artifact_sha256": {**manifest["artifact_sha256"], "timing.json":"0"*64}}
            with self.assertRaisesRegex(ValueError, "Artifact hash"):
                p.unpack_artifacts(bad, chunks, dest)

    def test_artifact_path_and_unexpected_file_guards(self):
        for name in ("../secret", "/absolute", "modal-auth.toml", "extra.py",
                     "activity.case05.pooler.m128.candidate.trace.json"):
            with self.subTest(name=name):
                self.assertFalse(p.artifact_name_ok(name))
                manifest, chunks = self.transport({"summary.json":b'{}', name:b"test"})
                with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                    p.unpack_artifacts(manifest, chunks, Path(temp)/"artifacts")
        self.assertTrue(p.artifact_name_ok("activity.case04.unpooler.m16.baseline.trace.json"))

    def test_cli_cannot_create_credentials_or_launch(self):
        for args in (("token", "new"), ("run", "payload.py"), ("deploy", "payload.py")):
            with self.subTest(args=args), patch.object(p.subprocess, "run") as run, self.assertRaises(ValueError):
                p.cli(Path("/unused"), *args)
            run.assert_not_called()

    def test_stopped_state_requires_zero_tasks(self):
        self.assertTrue(p.all_stopped([{"state":"stopped", "tasks":"0"}]))
        self.assertFalse(p.all_stopped([{"state":"stopped", "tasks":"1"}]))
        self.assertFalse(p.all_stopped([{"state":"stopping...", "tasks":"0"}]))

    def test_source_has_one_invocation_and_bounded_supervisors(self):
        tree = ast.parse(Path(p.__file__).read_text())
        invocations = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr in ("remote", "remote_gen", "spawn")]
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].func.attr, "remote_gen")
        entry = p.make_remote_entry()
        self.assertIsNone(entry.__closure__)
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "function"]
        settings = {k.arg: ast.literal_eval(k.value) for k in functions[0].keywords if k.arg != "image"}
        self.assertEqual(settings["gpu"], "H200")
        self.assertEqual((settings["timeout"], settings["startup_timeout"]), (390,45))
        self.assertIsNone(settings["retries"])
        self.assertEqual((settings["min_containers"], settings["max_containers"], settings["buffer_containers"]), (0,1,0))
        self.assertTrue(settings["block_network"] and settings["restrict_modal_access"])
        self.assertFalse(settings["include_source"])
        self.assertIn('threading.Timer(420, watchdog)', Path(p.__file__).read_text())

    def test_local_self_test_never_imports_modal_or_calls_subprocess(self):
        with patch.object(p.subprocess, "run", side_effect=AssertionError("external action")), \
             patch.object(p.subprocess, "check_output", side_effect=AssertionError("external action")):
            self.assertEqual(p.main(["self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
