"""Local launcher-boundary review; no Modal import, upload or GPU launch."""
import ast
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import projection_p044_modal as shared
import projection_p046_confirmation_modal as p


class ConfirmationLauncherTests(unittest.TestCase):
    def preflight(self, stamp=None):
        return dict(observed_utc=(stamp or datetime.now(timezone.utc)).isoformat(),
                    workspace=p.PROFILE, gpu_slot_granted=True,
                    verified_credit_balance_usd="28", verified_net_spend_cap_usd="0",
                    total_spend_usd="0", total_usage_usd="2")

    def transport(self, files=None):
        files = files or {"summary.json": b'{"complete":true,"stages":[{"mode":"timing","exit_code":0}],"worker_wall_seconds":1}',
                          "timing.json": b'{"passed":true}', "timing.log": b'bounded diagnostic log'}
        encoded = json.dumps({name: base64.b64encode(value).decode() for name, value in files.items()}, sort_keys=True).encode()
        packed = zlib.compress(encoded)
        chunks = [packed[i:i+p.CHUNK] for i in range(0, len(packed), p.CHUNK)]
        manifest = dict(chunks=len(chunks), packed_bytes=len(packed), encoded_bytes=len(encoded),
                        artifact_bytes=sum(map(len, files.values())), sha256=hashlib.sha256(packed).hexdigest(),
                        artifact_sha256={name: hashlib.sha256(value).hexdigest() for name, value in files.items()})
        return manifest, chunks

    def test_270_second_reserve_and_higher_actual_rate(self):
        self.assertEqual(p.reserve(), Decimal(".4076345000990"))
        self.assertLess(p.reserve(), Decimal(".408"))
        self.assertEqual(p.reserve(Decimal(".002")), Decimal(".6140"))
        self.assertEqual(p.reserve(Decimal(".0001")), p.reserve())
        self.assertGreater(270, 45 + 195 + 15)

    def test_budget_accounts_for_delayed_gross_and_strict_credit_boundary(self):
        rates = dict(gpu_hour_cost_h200="4.5", cpu_hour_cost=".05", mem_gib_hour_cost=".01")
        billing = dict(billed_cost="0", metered_cost="2.01", metered_cost_breakdown={"apps": "2.20"})
        rate, reserve, floor = p.budget_guard(self.preflight(), billing, rates)
        self.assertEqual(floor, Decimal("27.80"))
        self.assertEqual(rate, p.RATE_FLOOR)
        pre = self.preflight()
        pre["verified_credit_balance_usd"] = str(reserve + Decimal(".2"))
        with self.assertRaisesRegex(ValueError, "credits"):
            p.budget_guard(pre, billing, rates)
        with self.assertRaisesRegex(ValueError, "paid"):
            p.budget_guard(self.preflight(), {**billing, "billed_cost": ".01"}, rates)
        higher = {**rates, "gpu_hour_cost_h200": "7.2"}
        self.assertGreater(p.budget_guard(self.preflight(), billing, higher)[1], reserve)

    def test_inherited_preflight_auth_and_transport_guards_are_exact(self):
        for name in ("validate_preflight", "configure_existing_auth", "cli", "all_stopped", "unpack_artifacts"):
            self.assertIs(getattr(p, name), getattr(shared, name))
        now = datetime.now(timezone.utc)
        p.validate_preflight(self.preflight(now), now)
        for data in (self.preflight(now - timedelta(seconds=900)),
                     {**self.preflight(now), "gpu_slot_granted": False},
                     {**self.preflight(now), "verified_net_spend_cap_usd": "1"}):
            with self.assertRaises(ValueError):
                p.validate_preflight(data, now)
        auth_source = inspect.getsource(p.configure_existing_auth)
        self.assertIn("total_seconds() > 600", auth_source)
        self.assertIn("check_owned_file(auth)", auth_source)
        self.assertNotIn("auth.read_", auth_source)

    def test_exact_twelve_file_payload_excludes_secrets_and_controllers(self):
        expected = {p.SOURCE, p.PARENT, "projection_p046_confirmation.py", "benchmark.py", "harness.py",
                    "official/reference.py", "official/checks.py", "official/muse_bf16_proj.json",
                    "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt",
                    "licenses/June-9-Researcher-Reciprocity.txt"}
        self.assertEqual(len(p.PAYLOAD), 12)
        self.assertEqual(set(p.PAYLOAD), expected)
        self.assertFalse(any("local-notes" in name or "auth" in name or "modal" in name for name in p.PAYLOAD))
        self.assertEqual(hashlib.sha256((p.ROOT / p.SOURCE).read_bytes()).hexdigest(), p.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256((p.ROOT / p.PARENT).read_bytes()).hexdigest(), p.PARENT_SHA256)

    def test_frozen_payload_checks_head_commit_bytes_and_both_source_hashes(self):
        with patch.object(p.subprocess, "check_output") as command, self.assertRaises(ValueError):
            p.frozen_payload("main")
        command.assert_not_called()
        with patch.object(p.subprocess, "check_output", return_value="0" * 40 + "\n"), self.assertRaisesRegex(ValueError, "HEAD"):
            p.frozen_payload("1" * 40)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (*p.PAYLOAD, "projection_p046_confirmation_modal.py", "projection_p044_modal.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not the frozen source")
            def fake_git(command, **kwargs):
                return "1" * 40 + "\n" if command[1] == "rev-parse" else (root / command[2].split(":", 1)[1]).read_bytes()
            with patch.object(p.subprocess, "check_output", side_effect=fake_git), self.assertRaisesRegex(ValueError, "P039/P046 source mismatch"):
                p.frozen_payload("1" * 40, root)
            with patch.object(p.subprocess, "check_output", side_effect=["1" * 40 + "\n", b"different committed content"]), \
                 self.assertRaisesRegex(ValueError, "Uncommitted or changed payload"):
                p.frozen_payload("1" * 40, root)

    def test_existing_record_or_artifacts_refuse_before_any_external_action(self):
        for existing in ("experiments/projection-p046-confirmation-modal.json", "local-notes/projection-p046-confirmation-artifacts"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / existing
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("existing uncertain state")
                with patch.object(p, "ROOT", root), patch.object(p, "frozen_payload") as freeze, \
                     patch.object(p.subprocess, "run") as command, patch.object(p, "cli") as cli, \
                     self.assertRaisesRegex(ValueError, "reconcile"):
                    p.launch("1" * 40, root)
                freeze.assert_not_called()
                command.assert_not_called()
                cli.assert_not_called()

    def test_remote_entry_has_no_secret_closure_or_controller_globals(self):
        entry = p.make_remote_entry()
        self.assertIsNone(entry.__closure__)
        closure = inspect.getclosurevars(entry)
        self.assertEqual(closure.nonlocals, {})
        self.assertEqual(closure.globals, {})
        text = inspect.getsource(entry)
        for forbidden in ("control_root", "modal-auth", "MODAL_TOKEN", "local-notes", "cli(", "os.environ"):
            self.assertNotIn(forbidden, text)
        self.assertIn('for mode in ("timing",)', text)
        self.assertIn('"--deadline-seconds", "120"', text)
        self.assertIn('"--outer-timeout-seconds", "180"', text)
        self.assertIn("communicate(timeout=180)", text)
        self.assertIn("os.killpg(child.pid, signal.SIGKILL)", text)
        self.assertIn("output[-4000:]", text)

    def test_single_generator_bounded_gpu_resources_and_watchdog(self):
        text = Path(p.__file__).read_text()
        tree = ast.parse(text)
        remote = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr in ("remote", "remote_gen", "spawn")]
        self.assertEqual(len(remote), 1)
        self.assertEqual(remote[0].func.attr, "remote_gen")
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "function")
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in function.keywords if kw.arg != "image"}
        self.assertEqual((settings["gpu"], settings["cpu"], settings["memory"]), ("H200", (2, 2), (8192, 8192)))
        self.assertEqual((settings["timeout"], settings["startup_timeout"], settings["scaledown_window"]), (195, 45, 2))
        self.assertEqual((settings["min_containers"], settings["max_containers"], settings["buffer_containers"]), (0, 1, 0))
        self.assertIsNone(settings["retries"])
        self.assertTrue(settings["serialized"] and settings["block_network"] and settings["restrict_modal_access"])
        self.assertFalse(settings["include_source"])
        self.assertIn("threading.Timer(240, watchdog)", text)
        self.assertIn('record_path.open("x")', text)
        self.assertLess(text.index('record_path.open("x")'), text.index("with app.run("))
        self.assertIn("app.run(detach=False", text)

    def test_exporter_output_allowlist_and_chunk_bounds(self):
        tree = ast.parse(inspect.getsource(p.make_remote_entry))
        allowed = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "allowed")
        literal = next(node for node in ast.walk(allowed) if isinstance(node, ast.Set))
        self.assertEqual(ast.literal_eval(literal), {"summary.json", "timing.json", "timing.log"})
        text = inspect.getsource(p.make_remote_entry)
        self.assertIn("raw_bytes > 8 * 1024 * 1024", text)
        self.assertIn("len(packed) > 2 * 1024 * 1024", text)
        self.assertIn("i * 16384:(i + 1) * 16384", text)
        self.assertEqual(p.CHUNK, 16384)

    def test_three_artifact_stream_roundtrip_without_overwrite(self):
        manifest, chunks = self.transport()
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "artifacts"
            summary = p.unpack_artifacts(manifest, chunks, dest)
            self.assertTrue(summary["complete"])
            self.assertEqual({x.name for x in dest.iterdir()}, {"summary.json", "timing.json", "timing.log"})
            with self.assertRaises(ValueError):
                p.unpack_artifacts(manifest, chunks, dest)

    def test_corrupt_incomplete_and_unsafe_streams_fail_before_writes(self):
        manifest, chunks = self.transport()
        for altered, data in (({**manifest, "sha256": "0" * 64}, chunks), (manifest, []),
                              ({**manifest, "packed_bytes": p.MAX_PACKED + 1}, chunks)):
            with tempfile.TemporaryDirectory() as temp:
                dest = Path(temp) / "artifacts"
                with self.assertRaises(ValueError):
                    p.unpack_artifacts(altered, data, dest)
                self.assertFalse(dest.exists())
        for name in ("../secret", "/absolute", "modal-auth.toml", "extra.py"):
            manifest, chunks = self.transport({"summary.json": b"{}", name: b"fixture"})
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                p.unpack_artifacts(manifest, chunks, Path(temp) / "artifacts")

    def test_shutdown_requires_exact_app_stopped_zero_tasks_and_zero_paid(self):
        self.assertTrue(p.all_stopped([dict(state="stopped", tasks="0")]))
        self.assertFalse(p.all_stopped([dict(state="stopped", tasks="1")]))
        self.assertFalse(p.all_stopped([dict(state="stopping...", tasks="0")]))
        text = inspect.getsource(p.launch)
        self.assertIn('a["app_id"] == app.app_id', text)
        self.assertIn('bool(exact and all_stopped([exact]))', text)
        self.assertIn('record.get("shutdown_verified") and record.get("all_apps_stopped_tasks_zero") and record.get("paid_after_zero")', text)
        self.assertIn('state="terminal" if terminal else "uncertain"', text)
        self.assertIn('"app", "stop", app.app_id', text)

    def test_self_test_has_no_sdk_import_subprocess_or_network(self):
        with patch.object(p.subprocess, "run", side_effect=AssertionError("external action")), \
             patch.object(p.subprocess, "check_output", side_effect=AssertionError("external action")), \
             patch.dict(sys.modules, {"modal": None}):
            self.assertEqual(p.main(["self-test", "--control-root", "/unused"]), 0)


if __name__ == "__main__":
    unittest.main()
