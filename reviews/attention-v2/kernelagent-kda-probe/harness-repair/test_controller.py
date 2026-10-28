"""Offline lifecycle, source-scope, and bounded artifact transport tests."""
import base64
from contextlib import nullcontext, redirect_stdout
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("kda_h200_controller", HERE / "controller.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    def hashes(self):
        return {name: c.EXPECTED.get(name, "a" * 64) for name in c.payload()}

    def artifact(self, name, raw):
        packed = gzip.compress(raw, mtime=0)
        return {"name": name, "raw_bytes": len(raw), "raw_sha256": c.digest(raw),
            "gzip_bytes": len(packed), "gzip_sha256": c.digest(packed),
            "gzip_base64": base64.b64encode(packed).decode()}

    def reply(self, status="complete", exit_code=0):
        hashes = self.hashes()
        checked = {"passed": True, "ownership": True, "fixed_addresses": True}
        baseline = [{"case": {"batch": batch, "q_len": length, "layer": layer, "page": 1},
            "oracle": copy.deepcopy(checked), "post_oracle": copy.deepcopy(checked)}
            for batch in (1, 16, 32) for length in (1, 8) for layer in ("sliding", "full")]
        paired = [{"case": {"batch": batch, "q_len": 1, "layer": "full", "page": 1},
            "fixtures": [{"seed": seed,
                "oracle": {arm: copy.deepcopy(checked) for arm in ("champion", "v4")},
                "post_oracle": {arm: copy.deepcopy(checked) for arm in ("champion", "v4")},
                "mutation_oracle": {arm: {**checked, "timed": False} for arm in ("champion", "v4")},
                "samples": [{"champion": 100.0, "v4": 95.0} for _ in range(12)]}
                for seed in (135871, 135872)]} for batch in (16, 32)]
        report = {"status": status, "worker_sha256": hashes["worker.py"],
            "verified_hashes": {key: hashes[key] for key in ("selected.py", "v4.py", "reference.py", "checks.py")},
            "correctness_complete": True, "timing_complete": True, "counters_complete": True,
            "candidate_changed": False, "official_check": False, "ranked_submission": False,
            "baseline_cases": baseline, "paired_cases": paired, "counter_reports": []}
        artifacts = []
        for arm in ("champion", "v4"):
            for batch in (16, 32):
                stem = f"ncu-{arm}-b{batch}"
                raw = (stem + " fake report fixture, not native evidence").encode()
                artifacts += [self.artifact(stem + ".ncu-rep", raw), self.artifact(stem + ".csv", b"capture\n"),
                    self.artifact(stem + "-imported.csv", b"import\n")]
                report["counter_reports"].append({"arm": arm, "batch": batch,
                    "source_sha256": hashes["selected.py" if arm == "champion" else "v4.py"],
                    "artifact_name": stem + ".ncu-rep", "artifact_sha256": c.digest(raw), "artifact_bytes": len(raw),
                    "capture_csv": stem + ".csv", "import_csv": stem + "-imported.csv",
                    "roundtrip_passed": True, "child_checks": {"pre": True, "post": True},
                    "identity": {"kernel_name": "_full2h fixture" if arm == "v4" else "FlashAttnFwdSm90 fixture" if batch == 16 else "_fused fixture",
                        "grid": [132, 1, 1], "block": [256, 1, 1]}})
        artifacts.append(self.artifact("summary.json", json.dumps(report).encode()))
        return {"exit_code": exit_code, "mounted_sha256": hashes, "artifacts": artifacts, "artifact_errors": []}, report

    def test_flat_exact_allowlist_and_fixed_original_sources(self):
        self.assertEqual(set(c.payload()), {"worker.py", "selected.py", "v4.py", "reference.py", "checks.py", "LICENSE", "THIRD_PARTY_NOTICES.md"})
        self.assertFalse(any("/" in name or "local-notes" in str(path) for name, path in c.payload().items()))
        for name, expected in c.EXPECTED.items():
            self.assertEqual(c.digest(c.regular_bytes(c.payload()[name])), expected)

    def test_remote_function_has_no_local_closure_or_auth_references(self):
        fn = c.make_remote_entry()
        self.assertIsNone(fn.__closure__)
        self.assertFalse({"AUTH", "cli", "ORIGINAL", "PRIOR_RECEIPT", "configure_existing_auth"} & set(fn.__code__.co_names))
        source = (HERE / "controller.py").read_text().split("def make_remote_entry():", 1)[1].split("def unpack_artifacts", 1)[0]
        self.assertNotIn("local-notes", source)
        self.assertIn('block_network=True', (HERE / "controller.py").read_text())
        self.assertIn('restrict_modal_access=True', (HERE / "controller.py").read_text())

    def test_reused_accounting_is_not_a_fresh_balance_or_charge(self):
        record = c.historical_accounting()
        self.assertEqual(record["historical_verified_free_balance_usd"], "27.13")
        self.assertEqual(record["modeled_balance_after_allowances_usd"], "15.93")
        self.assertEqual(record["new_job_reserve_usd"], "1.20")
        self.assertIsNone(record["actual_charge_usd"])
        self.assertFalse(record["fresh_billing_queried"])

    def test_cli_rejects_every_non_app_operation_without_execution(self):
        for args in (("billing", "summary", "--json"), ("token", "info"), ("token", "new"), ("app", "stop", "bad", "--yes", "--env", "main")):
            with self.subTest(args=args), patch.object(c.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    c.cli(*args)
                run.assert_not_called()

    def test_cli_failure_never_forwards_raw_auth_output(self):
        completed = subprocess.CompletedProcess([], 1, "SENSITIVE_FIXTURE", "SENSITIVE_FIXTURE")
        with patch.object(c.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "^Supported Modal app operation failed; raw auth output withheld$"):
                c.cli("app", "list", "--env", "main", "--json")

    def test_stopped_requires_observed_zero_tasks(self):
        self.assertTrue(c.stopped([]))
        for count in (0, "0", "00"):
            self.assertTrue(c.stopped([{"state": "stopped", "tasks": count}]))
        for row in ({"state": "stopped", "tasks": 1}, {"state": "running", "tasks": "0"}, {"state": "stopped"}):
            self.assertFalse(c.stopped([row]))
        for count in (False, True, None, 0.0, 1, -1, "1", "001", "", " ", " 0", "0 ", "+0", "-0", "0.0", "0\n", "٠", "０", "0x0", "0e0", [], {}):
            with self.subTest(count=count):
                self.assertFalse(c.stopped([{"state": "stopped", "tasks": count}]))

    def test_original_native_shutdown_observations_are_correctly_reconciled(self):
        original = json.loads((HERE.with_name("kda-h200-trial") / "run.local.json").read_text())
        self.assertEqual(original["app_id"], "ap-NIeb23lk3RtcAOgTimYjQM")
        self.assertFalse(original["shutdown_verified"])  # immutable historical false-negative
        observations = original["shutdown_observations"]
        self.assertEqual(observations[0]["app"]["tasks"], "1")
        self.assertFalse(c.stopped([observations[0]["app"]]))
        self.assertTrue(all(c.stopped([row["app"]]) for row in observations[1:]))
        with patch.object(c, "cli", side_effect=[[row["app"]] for row in observations]) as cli, patch.object(c.time, "sleep"):
            _, exact, repaired = c.observe_shutdown(original["app_id"])
        self.assertEqual(cli.call_count, 2)
        self.assertEqual(len(repaired), 2)
        self.assertEqual(exact["tasks"], "0")
        self.assertTrue(c.stopped([exact]))

    def test_stop_ack_needs_independent_app_observation(self):
        sequence = [[{"app_id": "ap-test", "state": "stopped", "tasks": 1}], [{"app_id": "ap-test", "state": "stopped", "tasks": 0}]]
        with patch.object(c, "cli", side_effect=sequence) as cli, patch.object(c.time, "sleep"):
            rows, exact, observations = c.observe_shutdown("ap-test")
        self.assertEqual(cli.call_count, 2)
        self.assertEqual(len(observations), 2)
        self.assertTrue(c.stopped([exact]))
        with patch.object(c, "cli", return_value=[]):
            _, exact, _ = c.observe_shutdown("ap-missing", attempts=1)
        self.assertIsNone(exact)

    def test_complete_transport_keeps_exact_nonempty_reports(self):
        reply, _ = self.reply()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifacts"
            result = c.materialize_result(reply, path, self.hashes())
            self.assertTrue(result["completion_verified"])
            self.assertEqual((path / "ncu-champion-b16.ncu-rep").read_bytes(), b"ncu-champion-b16 fake report fixture, not native evidence")
            self.assertTrue((path / "summary.json.transport.gz").is_file())

    def test_failed_worker_and_nonzero_exit_preserve_partial_evidence(self):
        for status, code in (("failed", 124), ("complete", 1)):
            reply, _ = self.reply(status, code)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "artifacts"
                self.assertFalse(c.materialize_result(reply, path, self.hashes())["completion_verified"])
                self.assertTrue((path / "summary.json").is_file())
                self.assertTrue((path / "ncu-v4-b32.ncu-rep").is_file())

    def test_unknown_traversal_duplicate_and_existing_destination_rejected(self):
        reply, _ = self.reply()
        for name in ("../escape", "/tmp/escape", "secret.txt"):
            bad = copy.deepcopy(reply)
            bad["artifacts"][0]["name"] = name
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                c.unpack_artifacts(bad, Path(tmp) / "output")
        bad = copy.deepcopy(reply)
        bad["artifacts"].append(copy.deepcopy(bad["artifacts"][0]))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            c.unpack_artifacts(bad, Path(tmp) / "output")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            c.unpack_artifacts(reply, Path(tmp))

    def test_transport_rejects_corruption_and_decompression_bombs(self):
        for field in ("raw_sha256", "gzip_sha256"):
            reply, _ = self.reply()
            reply["artifacts"][0][field] = "0" * 64
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                c.unpack_artifacts(reply, Path(tmp) / "output")
        reply, _ = self.reply()
        reply["artifacts"] = [self.artifact("supervisor.log", b"a" * 4096)]
        with patch.object(c, "MAX_FILE", 128), tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "Raw artifact"):
            c.unpack_artifacts(reply, Path(tmp) / "output")

    def test_native_report_digest_roundtrip_correctness_and_sources_required(self):
        reply, report = self.reply()
        artifacts = {r["name"]: r for r in reply["artifacts"]}
        mutations = [lambda r: r.update(worker_sha256="0" * 64), lambda r: r.update(timing_complete=False),
            lambda r: r["counter_reports"][0].update(artifact_sha256="0" * 64),
            lambda r: r["counter_reports"][0].update(roundtrip_passed=False),
            lambda r: r["counter_reports"][0].update(child_checks={"pre": True, "post": False}),
            lambda r: r["counter_reports"][0].update(source_sha256="0" * 64),
            lambda r: r["counter_reports"][0]["identity"].update(kernel_name="unrelated"),
            lambda r: r["counter_reports"][0]["identity"].update(grid="UNKNOWN"),
            lambda r: r["counter_reports"][0]["identity"].update(block=[256, 1, False]),
            lambda r: r["counter_reports"][0]["identity"].update(grid=[132, 0, 1]),
            lambda r: r["baseline_cases"][0]["case"].update(q_len=9),
            lambda r: r["paired_cases"][0]["fixtures"][0]["mutation_oracle"]["v4"].update(passed=False),
            lambda r: r["paired_cases"][0]["fixtures"][0]["mutation_oracle"]["v4"].update(timed=True),
            lambda r: r["paired_cases"][0]["fixtures"].pop(),
            lambda r: r["paired_cases"][0]["fixtures"][0]["samples"].pop(),
            lambda r: r.update(ranked_submission=True)]
        for mutation in mutations:
            altered = copy.deepcopy(report)
            mutation(altered)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.validate_report(altered, self.hashes(), artifacts)

    def test_total_transport_bound_and_regular_source_guard(self):
        reply, _ = self.reply()
        reply["artifacts"] = [self.artifact("supervisor.log", b"a" * 80), self.artifact("ncu-help.txt", b"b" * 80)]
        with patch.object(c, "MAX_TOTAL", 128), tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "Raw artifact"):
            c.unpack_artifacts(reply, Path(tmp) / "output")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_bytes(b"fixture")
            alias = Path(tmp) / "alias"
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "Regular bounded file"):
                c.regular_bytes(alias)

    def test_source_freeze_rejects_changes_and_duplicate_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "manifest.json"
            c.save(file, {"fixed": True}, exclusive=True)
            with patch.object(c, "make_manifest", return_value={"fixed": True}):
                self.assertEqual(c.verify_manifest(file), {"fixed": True})
            with patch.object(c, "make_manifest", return_value={"fixed": False}), self.assertRaises(ValueError):
                c.verify_manifest(file)
            with self.assertRaises(FileExistsError):
                c.save(file, {}, exclusive=True)
            (Path(tmp) / "run.local.json").write_text("{}")
            with patch.object(c, "HERE", Path(tmp)), patch.object(c, "configure_existing_auth") as auth, self.assertRaisesRegex(ValueError, "never duplicate"):
                c.launch(file)
            auth.assert_not_called()

    def test_mocked_native_lifecycle_stops_and_observes_success_and_uncertain_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = root / "manifest.json"
                manifest.write_text("{}")
                reply, _ = self.reply()
                sources, invoked, resources = [], [], {}
                modal = ModuleType("modal")
                exception = ModuleType("modal.exception")
                exception.ServerWarning = type("ServerWarning", (Warning,), {})
                modal.config = SimpleNamespace(user_config_path=str(root / "protected.toml"))

                class Image:
                    @classmethod
                    def from_id(cls, identifier):
                        self.assertEqual(identifier, c.IMAGE)
                        return cls()

                    def add_local_file(self, path, target, **kwargs):
                        sources.append(target)
                        return self

                class App:
                    app_id = "ap-fixture"

                    def __init__(self, name, **kwargs):
                        resources["app"] = kwargs

                    def function(self, **kwargs):
                        resources["function"] = kwargs

                        def wrap(entry):
                            def remote():
                                invoked.append(True)
                                self_record = json.loads((root / "run.local.json").read_text())
                                if not self_record["invocation_attempted"]:
                                    raise AssertionError("Invocation ledger must precede request")
                                if fail:
                                    raise RuntimeError("Fixture uncertain remote outcome")
                                return reply
                            return SimpleNamespace(remote=remote)
                        return wrap

                    def run(self, **kwargs):
                        return nullcontext()

                modal.Image, modal.App = Image, App
                timer = SimpleNamespace(start=lambda: None, cancel=lambda: None)
                after = [{"app_id": "ap-fixture", "state": "stopped", "tasks": 0}]
                with patch.dict(sys.modules, {"modal": modal, "modal.exception": exception}), \
                        patch.object(c, "HERE", root), patch.object(c, "AUTH", root / "protected.toml"), \
                        patch.object(c, "verify_manifest", return_value={"payload_sha256": self.hashes()}), \
                        patch.object(c, "configure_existing_auth"), patch.object(c.threading, "Timer", return_value=timer), \
                        patch.object(c, "cli", side_effect=[[], {"exit_code": 0}, after]) as cli, \
                        redirect_stdout(io.StringIO()):
                    rc = c.launch(manifest)
                record = json.loads((root / "run.local.json").read_text())
                self.assertEqual(rc, 1 if fail else 0)
                self.assertEqual(record["state"], "needs-reconciliation" if fail else "function-terminal")
                self.assertTrue(record["shutdown_verified"])
                self.assertTrue(record["invocation_attempted"])
                self.assertEqual(len(invoked), 1)
                self.assertEqual(len(sources), 7)
                self.assertEqual(cli.call_args_list[1].args, ("app", "stop", "ap-fixture", "--yes", "--env", "main"))
                self.assertEqual(cli.call_args_list[2].args, ("app", "list", "--env", "main", "--json"))
                self.assertEqual(resources["function"]["max_containers"], 1)
                self.assertEqual(resources["function"]["retries"], 0)
                self.assertTrue(resources["function"]["block_network"])
                self.assertTrue(resources["function"]["restrict_modal_access"])


if __name__ == "__main__":
    unittest.main()
