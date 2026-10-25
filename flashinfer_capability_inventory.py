"""Bounded stdlib-only inventory; never imports packages, loads binaries or runs JIT.

This program is preparation for an authorized inspection inside an exact image.
Source text markers establish presence only, not callable or JIT-free APIs.
"""
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import re
import sys
from pathlib import Path, PurePosixPath

DISTRIBUTIONS = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
                 "torch", "sglang", "nvidia-cublas-cu12", "nvidia-cublas-cu13")
SOURCE_SUFFIXES = ("gemm_base.py", "jit/gemm/core.py", "jit/core.py", "jit/cpp_ext.py",
                   "mm_bf16_cublaslt.cu", "mm_bf16_cublaslt.cuh")
MANIFEST_NAMES = ("aot_manifest.json", "aot_cache_manifest.json", "jit_cache_manifest.json",
                  "aot_config.json", "aot_kernels.json", "build_meta.json", "manifest.json")
MARKERS = ("get_mm_bf16_cublaslt_module", "cublaslt_bf16_gemm_runner", "get_valid_tactics",
           "build_and_load", "load_cuda_ops", "FLASHINFER_JIT", "jit_cache", "nvcc", "AOT", "aot",
           "CUBLAS_COMPUTE_32F", "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",
           "CUBLASLT_ALGO_CONFIG_SPLITK_NUM")
MAX_FILES = 64
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_RECORD_ROWS = 10000
MAX_CONTEXTS = 8
MAX_CONTEXT_CHARS = 240


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
    found = {}
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line:
            break
        if line.startswith("Name: "):
            found["name"] = line[6:].strip()
        elif line.startswith("Version: "):
            found["version"] = line[9:].strip()
    return found


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
                               "The caller must independently bind this result to the exact container/image digest."])
    for name in DISTRIBUTIONS:
        report = dict(distribution=name, status="missing_distribution", version=None, files=[],
                      missing_source_suffixes=list(SOURCE_SUFFIXES), missing_manifest_names=list(MANIFEST_NAMES))
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
        record_info, record_bytes = reader.read(root, relative_meta + "/RECORD")
        report["files"].append(record_info)
        if record_bytes is None:
            report["inventory_status"] = "record_unavailable"
            continue
        selected, rejected, count = set(), 0, 0
        for row in csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="replace"))):
            count += 1
            if count > MAX_RECORD_ROWS:
                report["record_row_limit_reached"] = True
                break
            relative = _safe_relative(row[0]) if row else None
            if relative is None:
                rejected += 1
                continue
            if name.startswith("flashinfer-") and _selected_source(relative):
                selected.add(relative)
        report.update(record_rows_inspected=min(count, MAX_RECORD_ROWS), unsafe_record_rows=rejected,
                      inventory_status="bounded_record_inspected")
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
            info["literal_markers"], info["literal_context"] = _source_markers(data)
            found_suffixes.update(suffix for suffix in SOURCE_SUFFIXES if relative.endswith("/" + suffix))
            if PurePosixPath(relative).name in MANIFEST_NAMES:
                found_manifests.add(PurePosixPath(relative).name)
        report["missing_source_suffixes"] = sorted(set(SOURCE_SUFFIXES) - found_suffixes)
        report["missing_manifest_names"] = sorted(set(MANIFEST_NAMES) - found_manifests)
    output["read_limits"] = dict(max_files=reader.max_files, max_file_bytes=reader.max_file_bytes,
                                 max_total_bytes=reader.max_total_bytes, max_record_rows=MAX_RECORD_ROWS)
    output["reads"] = dict(files_attempted=reader.files_attempted, bytes_read=reader.bytes_read)
    output["package_imports_performed"] = False
    output["api_callability_verified"] = False
    output["jit_free_verified"] = False
    return output


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, allow_nan=False))
