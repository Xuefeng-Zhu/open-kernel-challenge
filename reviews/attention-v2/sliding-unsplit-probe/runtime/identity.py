"""Bounded, read-only installed-FA3 identity capture; no GPU/package imports."""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import sys

MAX_SOURCE_TEXT_BYTES = 64 * 1024
MAX_INSPECT_FILE_BYTES = 1024 * 1024
MAX_HASH_FILE_BYTES = 4 * 1024 * 1024 * 1024
MAX_HASH_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_LOADED_BINARIES = 16
PACKAGE_NAMES = ('torch', 'triton', 'sglang', 'sgl-kernel')
MODULE_PREFIXES = ('sgl_kernel', 'flash_attn', 'sgl_flash_attn')


def _relevant_library(path):
    lower = str(path).lower()
    return any(part in lower for part in
               ('sgl_kernel', 'sgl-kernel', 'flash_attn', 'flash-attn', 'sgl_flash_attn'))


def _file_record(path, budget):
    """Hash at most the declared byte budgets; never follow a directory walk."""
    record = {'path': str(path), 'sha256': None}
    try:
        file = Path(path)
        record['resolved_path'] = str(file.resolve())
        stat = file.stat()
        record.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if not file.is_file():
            record['hash_omitted_reason'] = 'not_regular_file'
            return record
        if stat.st_size > MAX_HASH_FILE_BYTES:
            record['hash_omitted_reason'] = 'per_file_hash_byte_limit'
            return record
        if stat.st_size > budget['remaining']:
            record['hash_omitted_reason'] = 'total_hash_byte_limit'
            return record
        digest = hashlib.sha256()
        consumed = 0
        with file.open('rb') as stream:
            while consumed < stat.st_size:
                chunk = stream.read(min(1024 * 1024, stat.st_size - consumed))
                if not chunk:
                    break
                digest.update(chunk)
                consumed += len(chunk)
            # Detect growth, without allowing the original budget to expand.
            grew = bool(stream.read(1))
        budget['remaining'] -= consumed
        budget['hashed_bytes'] += consumed
        if consumed != stat.st_size or grew or file.stat().st_size != stat.st_size:
            record['hash_omitted_reason'] = 'file_changed_while_reading'
        else:
            record['sha256'] = digest.hexdigest()
        return record
    except TimeoutError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        record['read_error_type'] = type(exc).__name__
        return record


def _source_record(fn, budget):
    try:
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
    except TimeoutError:
        raise
    except (OSError, TypeError) as exc:
        return {'available': False, 'reason': 'source_path_unavailable',
                'error_type': type(exc).__name__}
    if not path:
        return {'available': False, 'reason': 'source_path_unavailable'}
    record = _file_record(path, budget)
    record['available'] = record.get('sha256') is not None
    file_size = record.get('bytes')
    if file_size is None:
        return record
    try:
        if file_size <= MAX_SOURCE_TEXT_BYTES:
            raw = Path(path).read_bytes()
            if len(raw) <= MAX_SOURCE_TEXT_BYTES:
                record['file_text'] = raw.decode('utf-8')
                record['file_text_sha256'] = hashlib.sha256(raw).hexdigest()
        else:
            record['file_text_omitted_reason'] = 'source_text_byte_limit'
        if file_size <= MAX_INSPECT_FILE_BYTES:
            text = inspect.getsource(fn)
            raw = text.encode('utf-8')
            if len(raw) <= MAX_SOURCE_TEXT_BYTES:
                record['function_source'] = text
                record['function_source_sha256'] = hashlib.sha256(raw).hexdigest()
            else:
                record['function_source_omitted_reason'] = 'source_text_byte_limit'
        else:
            record['function_source_omitted_reason'] = 'inspect_file_byte_limit'
    except TimeoutError:
        raise
    except (OSError, UnicodeError, TypeError) as exc:
        record['source_text_error_type'] = type(exc).__name__
    return record


def inspect_fa3(fn):
    """Return an admission report; unsupported explicit APIs fail closed.

    This function never calls ``fn``, imports a GPU package, requests CUDA, or
    traverses the filesystem. A true API admission is not proof of native
    split/combine behavior; a later activity diagnostic must establish that.
    """
    report = {'admitted': False, 'reason': None,
              'exact_observed_extensions_verified': False,
              'callable_module': getattr(fn, '__module__', None),
              'callable_qualname': getattr(fn, '__qualname__', type(fn).__name__),
              'follow_wrapped': False,
              'limits': {'source_text_bytes': MAX_SOURCE_TEXT_BYTES,
                         'inspect_file_bytes': MAX_INSPECT_FILE_BYTES,
                         'per_file_hash_bytes': MAX_HASH_FILE_BYTES,
                         'total_hash_bytes': MAX_HASH_TOTAL_BYTES,
                         'loaded_binaries': MAX_LOADED_BINARIES},
              'qualification': 'Installed API/provenance only; not upstream identity or executed split/combine proof'}
    try:
        signature = inspect.signature(fn, follow_wrapped=False)
        report['signature'] = str(signature)
    except (TypeError, ValueError) as exc:
        report.update(reason='signature_unavailable', error_type=type(exc).__name__)
        return report
    parameter = signature.parameters.get('num_splits')
    if parameter is None:
        report['reason'] = 'explicit_num_splits_parameter_missing'
        return report
    report['num_splits_kind'] = parameter.kind.name
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY):
        report['reason'] = 'num_splits_not_keyword_capable'
        return report
    report['num_splits_default'] = None if parameter.default is inspect.Parameter.empty else repr(parameter.default)
    report.update(admitted=True, reason='explicit_keyword_capable_num_splits')
    budget = {'remaining': MAX_HASH_TOTAL_BYTES, 'hashed_bytes': 0}
    report['source'] = _source_record(fn, budget)
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = {'status': 'found', 'version': importlib.metadata.version(name)}
        except TimeoutError:
            raise
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {'status': 'not_installed'}
        except Exception as exc:
            packages[name] = {'status': 'unavailable', 'error_type': type(exc).__name__}
    report['packages'] = packages
    paths = {}

    def add_path(path, discovered_by):
        try:
            canonical = str(Path(path).resolve())
        except TimeoutError:
            raise
        except (OSError, ValueError):
            canonical = str(path)
        entry = paths.setdefault(canonical, {'observed_paths': [], 'discovered_by': []})
        if str(path) not in entry['observed_paths']:
            entry['observed_paths'].append(str(path))
        if discovered_by not in entry['discovered_by']:
            entry['discovered_by'].append(discovered_by)

    for name, module in sorted(list(sys.modules.items())):
        if not name.startswith(MODULE_PREFIXES):
            continue
        path = getattr(module, '__file__', None)
        if not isinstance(path, str) or not path.endswith(('.so', '.pyd', '.dll', '.dylib')):
            continue
        add_path(path, 'module:'+name)
    torch_module = sys.modules.get('torch')
    library_discovery = {'status': 'torch_not_loaded', 'relevant_paths': 0}
    if torch_module is not None:
        try:
            libraries = getattr(getattr(torch_module, 'ops'), 'loaded_libraries')
            if not isinstance(libraries, (set, list, tuple, frozenset)):
                raise TypeError('Loaded-library inventory is not a supported collection')
            library_discovery['status'] = 'observed'
            for value in libraries:
                if isinstance(value, (str, Path)) and _relevant_library(value):
                    add_path(str(value), 'torch.ops.loaded_libraries')
                    library_discovery['relevant_paths'] += 1
        except TimeoutError:
            raise
        except (AttributeError, TypeError, OSError) as exc:
            library_discovery.update(status='unavailable', error_type=type(exc).__name__)
    binaries = []
    omitted = max(0, len(paths)-MAX_LOADED_BINARIES)
    for path, discovery in sorted(paths.items())[:MAX_LOADED_BINARIES]:
        discoveries = discovery['discovered_by']
        record = _file_record(path, budget)
        record['observed_paths'] = discovery['observed_paths']
        record['discovered_by'] = discoveries
        record['module'] = next((entry.removeprefix('module:') for entry in discoveries
                                 if entry.startswith('module:')), None)
        binaries.append(record)
    report.update(loaded_binaries=binaries, loaded_binary_paths_omitted=omitted,
                  loaded_library_discovery=library_discovery,
                  hashed_file_bytes=budget['hashed_bytes'])
    report['exact_observed_extensions_verified'] = bool(binaries and not omitted
        and library_discovery['status'] != 'unavailable'
        and all(row['sha256'] is not None for row in binaries))
    if report['exact_observed_extensions_verified']:
        report['extension_identity_reason'] = 'all_observed_relevant_extensions_fully_hashed'
    elif not binaries:
        report['extension_identity_reason'] = 'no_relevant_loaded_extension_path_observed'
    elif omitted:
        report['extension_identity_reason'] = 'relevant_loaded_extension_paths_omitted'
    elif library_discovery['status'] == 'unavailable':
        report['extension_identity_reason'] = 'loaded_library_inventory_unavailable'
    else:
        report['extension_identity_reason'] = 'observed_extension_hash_unavailable'
    if not binaries:
        report['binary_identity_qualification'] = 'No relevant loaded extension path observed; native binary identity remains unavailable'
    elif any(row['sha256'] is None for row in binaries):
        report['binary_identity_qualification'] = 'One or more extension hashes unavailable within bounds; strict native admission must stop'
    else:
        report['binary_identity_qualification'] = 'Hashes identify observed loaded files; no claim that every transitive GPU binary was discovered'
    return report
