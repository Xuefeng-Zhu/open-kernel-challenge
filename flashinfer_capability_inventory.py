"""Bounded stdlib-only inventory; never imports packages, loads binaries or runs JIT.

This program is preparation for an authorized inspection inside an exact image.
Source text markers establish presence only, not callable or JIT-free APIs.
"""
import csv
import configparser
import hashlib
import importlib.metadata as metadata
import io
import json
import re
import sys
from pathlib import Path, PurePosixPath

DISTRIBUTIONS = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
                 "torch", "sglang", "nvidia-cublas-cu12", "nvidia-cublas-cu13",
                 "flashinfer-jit-cache-sm90", "flashinfer-jit-cache-sm90a")
SOURCE_SUFFIXES = ("gemm_base.py", "jit/gemm/core.py", "jit/core.py", "jit/cpp_ext.py",
                   "jit/env.py", "mm_bf16_cublaslt.cu", "mm_bf16_cublaslt.cuh")
MANIFEST_NAMES = ("aot_manifest.json", "aot_cache_manifest.json", "jit_cache_manifest.json",
                  "aot_config.json", "aot_kernels.json", "build_meta.json", "manifest.json",
                  "provider_manifest.json")
MARKERS = ("get_mm_bf16_cublaslt_module", "cublaslt_bf16_gemm_runner", "get_valid_tactics",
           "build_and_load", "load_cuda_ops", "FLASHINFER_JIT", "jit_cache", "nvcc", "AOT", "aot",
           "CUBLAS_COMPUTE_32F", "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",
           "CUBLASLT_ALGO_CONFIG_SPLITK_NUM", "get_jit_cache_providers",
           "get_aot_artifacts", "cuda_architectures", "FLASHINFER_DISABLE_JIT")
MAX_FILES = 64
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_RECORD_ROWS = 10000
MAX_CONTEXTS = 8
MAX_CONTEXT_CHARS = 240
MAX_PROVIDER_DISTRIBUTIONS = 8
PROVIDER_GROUP = "flashinfer.jit_cache.providers"
PROVIDER_NAME = re.compile(r"flashinfer-jit-cache-sm\d{2,3}[af]?")
PINNED_VERSION = "0.7.0.post1"


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts):
        return None
    return path.as_posix()


def _selected_source(relative):
    path = PurePosixPath(relative)
    parts = path.parts
    # Only files actually owned by a FlashInfer package, never another package
    # or a shared cache directory named in arbitrary source/environment content.
    if not parts or not (parts[0] == "flashinfer" or parts[0].startswith("flashinfer_")):
        return False
    return (any(relative == suffix or relative.endswith("/" + suffix) for suffix in SOURCE_SUFFIXES)
            or path.name in MANIFEST_NAMES
            or ("mm_bf16_cublaslt" in relative and path.suffix in (".so", ".cubin")))


class Reader:
    def __init__(self, max_files=MAX_FILES, max_file_bytes=MAX_FILE_BYTES, max_total_bytes=MAX_TOTAL_BYTES):
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (max_files, max_file_bytes, max_total_bytes)):
            raise ValueError("Read limits must be positive integers")
        if max_files > MAX_FILES or max_file_bytes > MAX_FILE_BYTES or max_total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("Read limits may only be reduced")
        self.max_files, self.max_file_bytes, self.max_total_bytes = max_files, max_file_bytes, max_total_bytes
        self.files_attempted = 0
        self.bytes_read = 0

    def read(self, root, relative):
        record = {"path": relative, "status": "unread", "size": None, "sha256": None}
        relative = _safe_relative(relative)
        if relative is None:
            record.update(path="<unsafe-relative-path>", status="unsafe_path")
            return record, None
        if self.files_attempted >= self.max_files:
            record["status"] = "file_count_limit"
            return record, None
        self.files_attempted += 1
        path = root
        try:
            for part in PurePosixPath(relative).parts:
                path = path / part
                if path.is_symlink():
                    record["status"] = "symlink_rejected"
                    return record, None
            if not path.is_file():
                record["status"] = "missing_or_not_regular"
                return record, None
            stat = path.stat()
            record["size"] = stat.st_size
            if stat.st_size > self.max_file_bytes:
                record["status"] = "file_byte_limit"
                return record, None
            if stat.st_size > self.max_total_bytes - self.bytes_read:
                record["status"] = "total_byte_limit"
                return record, None
            with path.open("rb") as stream:
                data = stream.read(stat.st_size)
            self.bytes_read += len(data)
            after = path.stat()
            if len(data) != stat.st_size or after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
                record["status"] = "changed_during_read"
                return record, None
            record.update(status="read", sha256=hashlib.sha256(data).hexdigest())
            return record, data
        except OSError as exc:
            record.update(status="read_error", error_type=type(exc).__name__)
            return record, None


def _headers(data):
    # Only Name/Version headers; neither package metadata code nor requirements run.
    found = {"provider_dependencies": []}
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line:
            break
        if line.startswith("Name: "):
            found["name"] = line[6:].strip()
        elif line.startswith("Version: "):
            found["version"] = line[9:].strip()
        elif line.startswith("Requires-Dist: "):
            match = re.match(r"Requires-Dist: ([A-Za-z0-9_.-]+)(?:\s|\[|\(|;|<|>|=|!|~|$)", line)
            name = _canonical(match.group(1)) if match else ""
            if PROVIDER_NAME.fullmatch(name) and name not in found["provider_dependencies"]:
                if len(found["provider_dependencies"]) < MAX_PROVIDER_DISTRIBUTIONS:
                    found["provider_dependencies"].append(name)
                else:
                    found["provider_dependency_limit_reached"] = True
    return found


def _entry_point_evidence(data):
    """Parse declarations only; never call EntryPoint.load or its factory."""
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(data.decode("utf-8"))
    except (UnicodeError, configparser.Error):
        return dict(status="invalid_entry_points", entries=[], factory_executed=False)
    items = list(parser.items(PROVIDER_GROUP)) if parser.has_section(PROVIDER_GROUP) else []
    entries = []
    for name, target in items[:MAX_PROVIDER_DISTRIBUTIONS]:
        # Reject arbitrary text/paths, and do not follow extras or imports.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name) or not re.fullmatch(
                r"flashinfer_[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*:[A-Za-z_][A-Za-z0-9_]*", target):
            entries.append(dict(status="unsupported_declaration"))
            continue
        module, function = target.split(":")
        entries.append(dict(status="declared", name=name, module=module, function=function))
    return dict(status="read", entries=entries, entries_truncated=len(items) > len(entries),
                factory_executed=False)


def _manifest_evidence(data):
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError, RecursionError):
        return dict(status="invalid_json", sm90_compatibility="unknown")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return dict(status="unknown_manifest_schema", sm90_compatibility="unknown")
    archs, modules = value.get("cuda_architectures"), value.get("modules")
    valid_archs = isinstance(archs, list) and 0 < len(archs) <= 32 and all(
        isinstance(a, str) and re.fullmatch(r"(?:sm_?|compute_)?\d{2,3}[af]?", a.lower()) for a in archs)
    normalized = sorted({re.sub(r"^(?:compute_|sm_?)", "", a.lower()) for a in archs}) if valid_archs else []
    distribution, version = value.get("distribution"), value.get("version")
    return dict(status="recognized_provider_fields",
        distribution=distribution if isinstance(distribution, str) and PROVIDER_NAME.fullmatch(_canonical(distribution)) else None,
        version=version if isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9.+_-]{1,100}", version) else None,
        declared_cuda_architectures=["sm" + a for a in normalized],
        sm90_compatibility=("declares_sm90_family" if set(normalized) & {"90", "90a"} else "no_declared_sm90_family") if valid_archs else "unknown",
        module_declared=("mm_bf16_cublaslt" in modules) if isinstance(modules, list) and all(isinstance(m, str) for m in modules) else None,
        binary_loadability_verified=False,
        qualification="Static declarations only; no factory, binary, CUDA target or fallback execution. Cache path is neither emitted nor followed.")


def _version_evidence(reports):
    versions = {r["distribution"]: r["version"] for r in reports if r.get("status") == "metadata_read"}
    base, shim = versions.get("flashinfer-python"), versions.get("flashinfer-jit-cache")
    for report in reports:
        name, version = report["distribution"], report.get("version")
        if name.startswith("flashinfer-"):
            report["pinned_version_relation"] = ("matches_release_or_cuda_local_suffix" if version == PINNED_VERSION or
                isinstance(version, str) and version.startswith(PINNED_VERSION + "+") else "different" if version else "unknown")
        if PROVIDER_NAME.fullmatch(name):
            report["provider_version_evidence"] = dict(
                matches_installed_shim=(version == shim) if version and shim else None,
                matches_installed_python=(version == base or version.startswith(base + "+")) if version and base else None,
                manifest_version_matches_metadata=[m.get("version") == version if m.get("version") and version else None
                    for m in report.get("provider_manifests", [])],
                qualification="Version equality is metadata evidence, not ABI, architecture, or binary-load validation")


def _source_markers(data):
    lines = data.decode("utf-8", errors="replace").splitlines()
    markers = {marker: [] for marker in MARKERS}
    contexts = []
    for index, line in enumerate(lines, 1):
        matches = [marker for marker in MARKERS if marker in line]
        for marker in matches:
            if len(markers[marker]) < MAX_CONTEXTS:
                markers[marker].append(index)
        if matches and len(contexts) < MAX_CONTEXTS:
            contexts.append(dict(line=index, markers=matches, text=line[:MAX_CONTEXT_CHARS]))
    return {key: value for key, value in markers.items() if value}, contexts


def inventory(distribution_lookup=metadata.distribution, reader=None):
    reader = reader or Reader()
    output = dict(kind="Static installed-package inventory; no package imports or execution",
                  python=sys.version.split()[0], distributions=[],
                  limitations=["Static source markers and binary/cache manifests do not prove callable or JIT-free APIs.",
                               "Only allowlisted distribution RECORD entries are inspected; no cache tree or environment is read.",
                               "Missing pattern lists mean not successfully inspected; per-file statuses distinguish absence from read limits.",
                               "Filesystem PathDistribution metadata paths are required; zip/custom distributions are unsupported.",
                               "The caller must independently bind this result to the exact container/image digest.",
                               "Provider discovery is limited to fixed SM90 names and bounded shim Requires-Dist declarations; undeclared providers remain uninspected.",
                               "Entry-point factories are never executed; dynamic manifests remain unknown."])
    queue = list(DISTRIBUTIONS)
    provider_names = {name for name in queue if PROVIDER_NAME.fullmatch(name)}
    for name in queue:
        report = dict(distribution=name, status="missing_distribution", version=None, files=[],
                      missing_source_suffixes=list(SOURCE_SUFFIXES), missing_manifest_names=list(MANIFEST_NAMES),
                      provider_manifests=[])
        output["distributions"].append(report)
        try:
            dist = distribution_lookup(name)
        except metadata.PackageNotFoundError:
            continue
        # Avoid Distribution.files/version/read_text: they read RECORD/METADATA
        # without our byte limits. _path is a guarded stdlib implementation detail.
        meta_path = getattr(dist, "_path", None)
        if not isinstance(dist, metadata.PathDistribution) or not isinstance(meta_path, Path):
            report["status"] = "unsupported_metadata_layout"
            continue
        lexical_root = Path(dist.locate_file("")).absolute()
        try:
            relative_meta = meta_path.absolute().relative_to(lexical_root).as_posix()
        except ValueError:
            report["status"] = "metadata_outside_distribution_root"
            continue
        if not relative_meta.endswith(".dist-info") or "/" in relative_meta:
            report["status"] = "unsupported_metadata_layout"
            continue
        # Canonicalize the shared root only after deriving a lexical relative
        # path; /tmp versus /private/tmp is a valid system-level alias on macOS.
        root = lexical_root.resolve()
        metadata_record, metadata_bytes = reader.read(root, relative_meta + "/METADATA")
        report["files"].append(metadata_record)
        if metadata_bytes is None:
            report["status"] = "metadata_unavailable"
            continue
        headers = _headers(metadata_bytes)
        if _canonical(headers.get("name", "")) != name or not headers.get("version"):
            report["status"] = "metadata_identity_mismatch"
            continue
        report.update(status="metadata_read", version=headers["version"])
        if name == "flashinfer-jit-cache":
            report["declared_provider_dependencies"] = headers["provider_dependencies"]
            for dependency in headers["provider_dependencies"]:
                if dependency not in provider_names:
                    if len(provider_names) >= MAX_PROVIDER_DISTRIBUTIONS:
                        report["provider_distribution_limit_reached"] = True
                        break
                    provider_names.add(dependency)
                    queue.append(dependency)
            if headers.get("provider_dependency_limit_reached"):
                report["provider_distribution_limit_reached"] = True
        record_info, record_bytes = reader.read(root, relative_meta + "/RECORD")
        report["files"].append(record_info)
        if record_bytes is None:
            report["inventory_status"] = "record_unavailable"
            continue
        selected, listed, rejected, count = set(), set(), 0, 0
        for row in csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="replace"))):
            count += 1
            if count > MAX_RECORD_ROWS:
                report["record_row_limit_reached"] = True
                break
            relative = _safe_relative(row[0]) if row else None
            if relative is None:
                rejected += 1
                continue
            listed.add(relative)
            if name.startswith("flashinfer-") and _selected_source(relative):
                selected.add(relative)
        report.update(record_rows_inspected=min(count, MAX_RECORD_ROWS), unsafe_record_rows=rejected,
                      inventory_status="bounded_record_inspected")
        factory_sources = set()
        entry_path = relative_meta + "/entry_points.txt"
        if name.startswith("flashinfer-") and entry_path in listed:
            entry_info, entry_bytes = reader.read(root, entry_path)
            report["files"].append(entry_info)
            if entry_bytes is not None:
                report["provider_entry_points"] = _entry_point_evidence(entry_bytes)
                for entry in report["provider_entry_points"]["entries"]:
                    if entry.get("status") != "declared":
                        continue
                    module_path = entry["module"].replace(".", "/")
                    # Only exact module files named in this distribution's RECORD.
                    # Merely read/hash source; never interpret factory results.
                    for relative in (module_path + ".py", module_path + "/__init__.py"):
                        if relative in listed and _safe_relative(relative):
                            selected.add(relative)
                            factory_sources.add(relative)
        elif name.startswith("flashinfer-"):
            report["provider_entry_points"] = dict(status="not_in_inspected_record", entries=[], factory_executed=False)
        found_suffixes, found_manifests = set(), set()
        for relative in sorted(selected):
            if reader.files_attempted >= reader.max_files:
                report["file_count_limit_reached"] = True
                break
            info, data = reader.read(root, relative)
            report["files"].append(info)
            if PurePosixPath(relative).suffix in (".so", ".cubin"):
                # Stat/read/hash only. Never pass binary bytes to text decoding,
                # source-marker scanning, a module loader or a dynamic loader.
                info.update(kind="prebuilt_mm_bf16_cublaslt_artifact",
                            binary_bytes_not_loaded=True, binary_bytes_not_decoded=True)
                continue
            if data is None:
                continue
            found_suffixes.update(suffix for suffix in SOURCE_SUFFIXES if relative.endswith("/" + suffix))
            if PurePosixPath(relative).name in MANIFEST_NAMES:
                found_manifests.add(PurePosixPath(relative).name)
                evidence = _manifest_evidence(data)
                info["provider_manifest_evidence"] = evidence
                report["provider_manifests"].append(dict(path=relative, sha256=info["sha256"], **evidence))
            else:
                info["literal_markers"], contexts = _source_markers(data)
                # Provider source may contain resolved local paths. Retain marker
                # locations/hash, never source excerpts or dynamic return values.
                info["literal_context"] = [] if relative in factory_sources else contexts
                if relative in factory_sources:
                    info["kind"] = "unexecuted_provider_factory_source"
        report["missing_source_suffixes"] = sorted(set(SOURCE_SUFFIXES) - found_suffixes)
        report["missing_manifest_names"] = sorted(set(MANIFEST_NAMES) - found_manifests)
    _version_evidence(output["distributions"])
    output["read_limits"] = dict(max_files=reader.max_files, max_file_bytes=reader.max_file_bytes,
                                 max_total_bytes=reader.max_total_bytes, max_record_rows=MAX_RECORD_ROWS,
                                 max_provider_distributions=MAX_PROVIDER_DISTRIBUTIONS)
    output["reads"] = dict(files_attempted=reader.files_attempted, bytes_read=reader.bytes_read)
    output["package_imports_performed"] = False
    output["api_callability_verified"] = False
    output["jit_free_verified"] = False
    output["provider_factories_executed"] = False
    output["binary_loadability_verified"] = False
    return output


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, allow_nan=False))
