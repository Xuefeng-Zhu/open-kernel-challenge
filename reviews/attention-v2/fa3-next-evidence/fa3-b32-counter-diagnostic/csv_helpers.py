"""Strict reader for the installed NCU raw wide CSV, not a profiler launcher."""
import csv
import io
import math
import re

MAX_CSV_BYTES = 128 * 1024
REQUIRED = ('ID', 'Kernel Name', 'Grid Size', 'Block Size')
NUMBER = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
TRIPLE = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')


def normalize_symbol(symbol):
    """Whitespace only; never rewrite template types or demangler differences."""
    return ' '.join(symbol.split())


def _triple(text):
    match = TRIPLE.fullmatch(text.strip())
    if not match:
        raise ValueError('Invalid NCU grid/block triple')
    value = [int(part) for part in match.groups()]
    if any(part < 1 or part > 2**31-1 for part in value):
        raise ValueError('NCU grid/block component outside positive integer bound')
    return value


def _number(text):
    value = text.strip()
    if not NUMBER.fullmatch(value):
        raise ValueError('Missing or malformed numeric NCU metric')
    number = float(value.replace(',', ''))
    if not math.isfinite(number) or number < 0:
        raise ValueError('NCU metric must be finite and nonnegative')
    return number


def parse_ncu_csv(text, metrics):
    """Return one actual kernel row with full name, launch geometry and units.

    NCU's raw CSV has a metadata/metric header, then a unit row, then data rows.
    Logs/progress lines are retained by the caller, but cannot count as data.
    This exact command contract is wide CSV; other formats fail explicitly.
    """
    if not isinstance(text, str) or len(text.encode()) > MAX_CSV_BYTES:
        raise ValueError('NCU CSV exceeds fixed byte bound')
    metrics = list(metrics)
    if (not metrics or len(metrics) > 16 or len(set(metrics)) != len(metrics)
            or any(not isinstance(name, str) or not name or name.strip() != name for name in metrics)):
        raise ValueError('Unique nonempty fixed metrics required')
    # NCU logs and our explicit progress JSON are not CSV. Remove only their
    # known line prefixes; strict parsing still applies to the retained table.
    csv_text = '\n'.join(line for line in text.splitlines()
        if not line.startswith('==') and not re.match(r'^[A-Za-z0-9_]+_JSON=', line))
    try:
        rows = list(csv.reader(io.StringIO(csv_text), strict=True))
    except csv.Error as exc:
        raise ValueError('Malformed NCU CSV') from exc
    found = []
    header_count = 0
    index = 0
    while index < len(rows):
        header = rows[index]
        if not all(name in header for name in REQUIRED):
            index += 1
            continue
        header_count += 1
        needed = list(REQUIRED) + metrics
        if any(header.count(name) != 1 for name in needed):
            raise ValueError('NCU header has missing or duplicate required columns')
        width = len(header)
        if index+1 >= len(rows):
            raise ValueError('NCU unit row is absent')
        units = rows[index+1]
        if (len(units) != width or units[header.index('ID')].strip()
                or units[header.index('Kernel Name')].strip()):
            raise ValueError('NCU unit row does not match wide CSV header')
        index += 2
        while index < len(rows):
            row = rows[index]
            if all(name in row for name in REQUIRED):
                break
            id_index = header.index('ID')
            if len(row) > id_index and row[id_index].isdigit():
                if len(row) != width:
                    raise ValueError('NCU data row has inconsistent column count')
                name = row[header.index('Kernel Name')]
                if not name.strip():
                    raise ValueError('NCU data row has no kernel name')
                values = {}
                range_warnings = []
                for metric in metrics:
                    column = header.index(metric)
                    raw, unit = row[column], units[column].strip()
                    value = _number(raw)
                    percent = metric.endswith('.pct') or '.pct_of_peak_' in metric
                    if percent and unit != '%':
                        raise ValueError('NCU percentage unit is invalid')
                    if percent and value > 100:
                        range_warnings.append(dict(metric=metric, value=value,
                            interpretation='Percentage exceeds 100; preserve the reported value without clipping and treat bottleneck-threshold interpretation as inconclusive.'))
                    if not unit:
                        raise ValueError('Requested NCU metric has no unit')
                    values[metric] = dict(value=value, unit=unit, raw=raw)
                metadata = {key: row[header.index(key)] for key in
                    ('Process ID', 'Context', 'Stream', 'Device', 'CC') if key in header}
                found.append(dict(id=row[id_index], kernel_name=name,
                    kernel_name_whitespace_normalized=normalize_symbol(name),
                    grid=_triple(row[header.index('Grid Size')]),
                    block=_triple(row[header.index('Block Size')]), metrics=values,
                    metadata=metadata, column_count=width, range_warnings=range_warnings))
            index += 1
    if header_count == 0:
        raise ValueError('No NCU raw wide CSV header')
    if len(found) != 1:
        raise ValueError('Expected exactly one actual NCU kernel row')
    return {**found[0], 'row_count': 1,
        'symbol_qualification': 'Full reported name preserved; only whitespace is normalized for optional comparison.'}


def compare_kernel_identity(parsed, graph_symbol, graph_grid, graph_block):
    """No structural/type guessing: a demangler difference leaves a mismatch."""
    checks = dict(symbol_equal_after_whitespace_normalization=
        normalize_symbol(parsed['kernel_name']) == normalize_symbol(graph_symbol),
        grid_equal=parsed['grid'] == list(graph_grid),
        block_equal=parsed['block'] == list(graph_block))
    return dict(matches=all(checks.values()), **checks,
        qualification='Name/grid/block equality is necessary identity evidence, not proof of identical graph/eager scheduling state or split counts.')
