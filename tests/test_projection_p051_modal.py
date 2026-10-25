"""P051 launcher fixture/static tests only; no SDK, external calls or GPU."""
import ast
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import projection_p051_modal as p


def artifact_fixture(complete=True):
    compiled, content = {}, {}
    for orientation in ("control", "weight_first", "reduce"):
        name = "timing.pooler_m8." + orientation + ".ptx"
        raw = ("// fixture " + orientation + "\n.version 8.0\n").encode()
        content[name] = raw
        compiled[orientation] = dict(ptx_file=name, ptx_sha256=hashlib.sha256(raw).hexdigest())
    report = dict(complete=complete, compile_only=True, cases=[dict(
        name="pooler_m8", m=8, n=4096, k=8192, complete=complete,
        generated_kernel_executed=False, correctness_status="not run", correctness=[], compiled=compiled)])
    content["timing.json"] = json.dumps(report).encode()
    content["summary.json"] = json.dumps(dict(complete=complete, stages=[dict(mode="compile-only", exit_code=0 if complete else 1)])).encode()
    content["compile-only.log"] = b"fixture\n"
    return content


def encode(content):
    raw = json.dumps({name: base64.b64encode(value).decode() for name, value in content.items()}, sort_keys=True).encode()
    packed = zlib.compress(raw)
    chunks = [packed[i:i + p.CHUNK] for i in range(0, len(packed), p.CHUNK)]
    manifest = dict(chunks=len(chunks), packed_bytes=len(packed), encoded_bytes=len(raw),
        artifact_bytes=sum(map(len, content.values())), sha256=hashlib.sha256(packed).hexdigest(),
        artifact_sha256={name: hashlib.sha256(value).hexdigest() for name, value in content.items()})
    return manifest, chunks


class P051LauncherTests(unittest.TestCase):
    def test_minimal_exact_payload_and_no_secret_controller_or_oracle_upload(self):
        self.assertEqual(p.PAYLOAD, ("projection_p051_orientation.py", "candidates/pooler_triton_split8.py",
            "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/Triton-MIT.txt", "licenses/June-9-Researcher-Reciprocity.txt"))
        self.assertEqual(len(set(p.PAYLOAD)), 6)
        self.assertFalse(any("local-notes" in x or "auth" in x or "official/" in x or x.endswith("_modal.py") for x in p.PAYLOAD))
        self.assertEqual(len(p.ARTIFACTS), 6)
        self.assertFalse(any(".." in name or "/" in name for name in p.ARTIFACTS))

    def test_exact_compile_only_command_and_timeout_boundaries(self):
        tree = ast.parse(Path(p.__file__).read_text())
        command = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
        literals = [node.value for node in command.elts if isinstance(node, ast.Constant)]
        self.assertEqual(literals, ["-B", "--output", "--compile-only", "--deadline-seconds", "75", "--outer-timeout-seconds", "90"])
        self.assertIn("projection_p051_orientation.py", [node.value for node in ast.walk(command)
                      if isinstance(node, ast.Constant) and isinstance(node.value, str)])
        source = Path(p.__file__).read_text()
        for required in ("child.communicate(timeout=90)", "threading.Timer(150, watchdog)", "os.killpg(child.pid, signal.SIGKILL)"):
            self.assertIn(required, source)
        self.assertNotIn('mode in ("timing",)', source)

    def test_one_invocation_resources_no_retry_network_or_closure(self):
        tree = ast.parse(Path(p.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        remote = [node for node in calls if node.func.attr in ("remote", "remote_gen", "spawn")]
        self.assertEqual([node.func.attr for node in remote], ["remote_gen"])
        settings = {kw.arg: ast.literal_eval(kw.value) for node in calls if node.func.attr == "function"
                    for kw in node.keywords if kw.arg != "image"}
        self.assertEqual((settings["gpu"], settings["cpu"], settings["memory"]), ("H200", (2, 2), (8192, 8192)))
        self.assertEqual((settings["max_containers"], settings["min_containers"], settings["buffer_containers"]), (1, 0, 0))
        self.assertEqual((settings["timeout"], settings["startup_timeout"], settings["scaledown_window"]), (105, 45, 2))
        self.assertIsNone(settings["retries"])
        self.assertTrue(settings["block_network"] and settings["restrict_modal_access"])
        self.assertFalse(settings["include_source"])
        self.assertIsNone(p.make_remote_entry().__closure__)
        app_runs = [node for node in calls if node.func.attr == "run" and isinstance(node.func.value, ast.Name) and node.func.value.id == "app"]
        self.assertEqual(len(app_runs), 1)
        self.assertFalse(next(kw.value.value for kw in app_runs[0].keywords if kw.arg == "detach"))

    def test_frozen_commit_verifies_every_payload_byte_and_dynamic_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            head = "a" * 40
            blobs = {}
            for name in (*p.PAYLOAD, "projection_p051_modal.py", "projection_p044_modal.py"):
                blobs[name] = ("fixture " + name).encode()
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blobs[name])
            def git(command, **kwargs):
                return head + "\n" if command[1] == "rev-parse" else blobs[command[2].split(":", 1)[1]]
            with patch.object(p.subprocess, "check_output", side_effect=git):
                paths, hashes = p.frozen_payload(head, root)
                self.assertEqual(set(paths), set(p.PAYLOAD))
                self.assertEqual(hashes[p.WORKER], hashlib.sha256(blobs[p.WORKER]).hexdigest())
                (root / p.WORKER).write_text("uncommitted change")
                with self.assertRaisesRegex(ValueError, "Uncommitted"):
                    p.frozen_payload(head, root)

    def test_wrong_head_and_symlink_payload_refused(self):
        with patch.object(p.subprocess, "check_output", return_value="b" * 40 + "\n"):
            with self.assertRaisesRegex(ValueError, "HEAD differs"):
                p.frozen_payload("a" * 40)
        with patch.object(p.subprocess, "check_output") as call:
            with self.assertRaisesRegex(ValueError, "reviewed commit"):
                p.frozen_payload("HEAD")
            call.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").write_text("fixture")
            (root / p.WORKER).symlink_to(root / "target")
            with patch.object(p.subprocess, "check_output", return_value="a" * 40 + "\n"):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    p.frozen_payload("a" * 40, root)

    def test_existing_ledger_prevents_any_auth_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments/projection-p051-compile-modal.json").write_text("{}")
            with patch.object(p, "ROOT", root), patch.object(p, "configure_existing_auth") as auth, patch.object(p, "frozen_payload") as freeze:
                with self.assertRaisesRegex(ValueError, "reconcile"):
                    p.launch("a" * 40, root)
                auth.assert_not_called(); freeze.assert_not_called()
        self.assertIn('with record_path.open("x")', Path(p.__file__).read_text())

    def test_reserve_paid_credit_and_delayed_billing_guard(self):
        self.assertEqual(p.reserve(), Decimal(180) * p.RATE_FLOOR * Decimal("1.1") + Decimal(".02"))
        self.assertLess(p.reserve(), Decimal(".279"))
        rates = dict(gpu_hour_cost_h200="4.5396", cpu_hour_cost=".04728", mem_gib_hour_cost=".00807")
        billing = dict(billed_cost="0", metered_cost="2.00", metered_cost_breakdown={"app": "2.10"})
        pre = dict(verified_credit_balance_usd=".37", total_usage_usd="2.00")
        with self.assertRaisesRegex(ValueError, "reserve"):
            p.budget_guard(pre, billing, rates)
        pre["verified_credit_balance_usd"] = "28"
        rate, reserve, floor = p.budget_guard(pre, billing, rates)
        self.assertEqual(floor, Decimal("27.90"))
        self.assertGreaterEqual(rate, p.RATE_FLOOR)
        with self.assertRaisesRegex(ValueError, "zero"):
            p.budget_guard(pre, dict(billing, billed_cost=".01"), rates)

    def test_inherited_fresh_cap_and_explicit_slot_guard(self):
        stamp = datetime.now(timezone.utc)
        good = dict(observed_utc=stamp.isoformat(), workspace="xuefeng-zhu", gpu_slot_granted=True,
            verified_net_spend_cap_usd="0", total_spend_usd="0", verified_credit_balance_usd="28", total_usage_usd="2")
        p.validate_preflight(good, stamp)
        for changed in (dict(observed_utc=(stamp-timedelta(minutes=16)).isoformat()),
                        dict(gpu_slot_granted=False), dict(verified_net_spend_cap_usd="1")):
            with self.assertRaises(ValueError):
                p.validate_preflight(dict(good, **changed), stamp)

    def test_complete_transport_exact_ptx_hashes_no_execution_claim(self):
        content = artifact_fixture()
        manifest, chunks = encode(content)
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out"
            self.assertTrue(p.unpack_artifacts(manifest, chunks, dest)["complete"])
            self.assertEqual({f.name for f in dest.iterdir()}, set(content))
            for name, raw in content.items(): self.assertEqual((dest / name).read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, "destination exists"):
                p.unpack_artifacts(manifest, chunks, dest)

    def test_transport_rejects_traversal_tampering_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out"
            content = artifact_fixture(); content["../bad.py"] = b"bad"
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                p.unpack_artifacts(*encode(content), dest)
            self.assertFalse(dest.exists())
            manifest, chunks = encode(artifact_fixture())
            with self.assertRaisesRegex(ValueError, "Packed hash"):
                p.unpack_artifacts(dict(manifest, sha256="0"*64), chunks, dest)
            with self.assertRaisesRegex(ValueError, "chunk count"):
                p.unpack_artifacts(manifest, [], dest)
            self.assertFalse(dest.exists())

    def test_complete_report_requires_exact_case_no_execution_and_three_ptx(self):
        mutations = (
            lambda r: r.update(compile_only=False),
            lambda r: r.update(complete=False),
            lambda r: r["cases"].append(dict(r["cases"][0])),
            lambda r: r["cases"][0].update(m=7),
            lambda r: r["cases"][0].update(generated_kernel_executed=True),
            lambda r: r["cases"][0].update(correctness_status="passed"),
            lambda r: r["cases"][0]["compiled"].pop("reduce"),
            lambda r: r["cases"][0]["compiled"]["control"].update(ptx_sha256="0"*64),
        )
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out"
            for mutate in mutations:
                content = artifact_fixture()
                report = json.loads(content["timing.json"])
                mutate(report)
                content["timing.json"] = json.dumps(report).encode()
                with self.assertRaises(ValueError): p.unpack_artifacts(*encode(content), dest)
                self.assertFalse(dest.exists())
            content = artifact_fixture(); del content["timing.pooler_m8.reduce.ptx"]
            with self.assertRaisesRegex(ValueError, "PTX evidence"):
                p.unpack_artifacts(*encode(content), dest)

    def test_incomplete_failure_preserved_without_fabricated_success(self):
        content = {"summary.json": b'{"complete":false}', "timing.json": b'{"compile_only":true,"complete":false,"error":"fixture"}'}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out"
            self.assertFalse(p.unpack_artifacts(*encode(content), dest)["complete"])
            self.assertEqual((dest / "timing.json").read_bytes(), content["timing.json"])

    def test_timeout_atomic_temp_omitted_but_unknown_or_symlink_rejected(self):
        for extra in ("regular_temp", "symlink_temp", "unknown"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                remote_root, remote_dest = root / "source", root / "remote"
                remote_root.mkdir()
                original_path_type = type(root)

                mappings = {"/opt/projection-p051-compile": str(remote_root),
                            "/tmp/projection-p051-compile": str(remote_dest)}

                def mapped_path(path):
                    return original_path_type(mappings.get(str(path), path))

                child = Mock(pid=1234567)
                child.communicate.side_effect = [subprocess.TimeoutExpired("fixture", 90), (b"timeout fixture", None)]

                def fake_popen(command, **kwargs):
                    self.assertIn("--compile-only", command)
                    (remote_dest / "timing.json").write_bytes(b'{"compile_only":true,"complete":false}')
                    (remote_dest / "timing.pooler_m8.control.ptx").write_bytes(b"// partial compiled evidence\n")
                    if extra == "regular_temp":
                        (remote_dest / "timing.json.tmp").write_bytes(b"incomplete JSON")
                    elif extra == "symlink_temp":
                        (remote_dest / "timing.json.tmp").symlink_to(remote_dest / "timing.json")
                    else:
                        (remote_dest / "unexpected.txt").write_text("not allowed")
                    return child

                with patch("pathlib.Path", side_effect=mapped_path), patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("os.killpg") as kill:
                    items = list(p.make_remote_entry()())
                kill.assert_called_once()
                if extra != "regular_temp":
                    self.assertEqual(items, [{"kind": "failure", "reason": "artifact_path_guard"}])
                    continue
                manifest = items[0]
                self.assertNotIn("timing.json.tmp", manifest["artifact_sha256"])
                self.assertEqual(items[-1]["kind"], "complete")
                chunks = [item["data"] for item in items if item["kind"] == "chunk"]
                output = root / "retrieved"
                summary = p.unpack_artifacts(manifest, chunks, output)
                self.assertFalse(summary["complete"])
                self.assertEqual(summary["stages"][0]["exit_code"], 124)
                self.assertEqual((output / "compile-only.log").read_bytes(), b"timeout fixture")
                self.assertEqual((output / "timing.pooler_m8.control.ptx").read_bytes(), b"// partial compiled evidence\n")
                self.assertEqual((output / "timing.json").read_bytes(), b'{"compile_only":true,"complete":false}')

    def test_shutdown_reconciles_exact_app_and_paid_zero(self):
        source = Path(p.__file__).read_text()
        for required in ('stop()\n        timer.cancel()', 'exact and all_stopped([exact])',
                         'record["all_apps_stopped_tasks_zero"] = all_stopped(apps)',
                         'record["paid_after_zero"] = Decimal(after["billed_cost"]) == 0',
                         'record["state"] = "terminal" if terminal else "needs-reconciliation"'):
            self.assertIn(required, source)
        self.assertIn('"app", "stop", app.app_id, "--yes"', source)
        self.assertEqual(p.all_stopped([dict(state="stopped", tasks="0")]), True)
        self.assertFalse(p.all_stopped([dict(state="stopped", tasks="1")]))

    def test_self_test_never_imports_sdk_or_calls_external(self):
        with patch.object(p.subprocess, "run", side_effect=AssertionError("external")), \
             patch.object(p.subprocess, "check_output", side_effect=AssertionError("external")), patch("builtins.print"):
            self.assertEqual(p.main(["self-test", "--control-root", "/unused"]), 0)


if __name__ == "__main__":
    unittest.main()
