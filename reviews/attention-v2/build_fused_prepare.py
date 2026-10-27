"""Offline source assembly with exact parent hash; no import/execution/upload."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parent, output = (HERE/args.parent).resolve(), (HERE/args.output).resolve()
    if parent.parent != HERE or output.parent != HERE or output.exists():
        raise ValueError('Read one local parent; never overwrite an existing candidate')
    raw = parent.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_sha:
        raise ValueError('Parent source identity mismatch')
    if not raw.startswith(b'# okc: problem=muse_attention_decode\n'):
        raise ValueError('Wrong problem contract')
    source = raw.decode()
    component = (HERE/'fa3_prologue_component.py').read_text()
    needle = 'def kernel_fn(data: Inputs) -> torch.Tensor:\n    cfg = data["config"]\n'
    assert source.count(needle) == 1
    added = ('    if (cfg["batch"] == 1 and cfg["q_len"] == 8\n'
             '            and cfg["window"] == 2048 and cfg["rope_theta"] == 500000):\n'
             '        return _fa3_sliding_verify_fused(data)\n')
    source = source.replace(needle, component+'\n\n'+needle+added)
    lines = source.splitlines(keepends=True)
    lines[1] = '# A2-05: only B1/L8/sliding uses fused official-precision prologue before unchanged FA3.\n'
    source = ''.join(lines)
    before = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    assert set(after)-set(before) == {'_fa3_prepare_fused', '_fa3_sliding_verify_fused'}
    for name in before:
        if name != 'kernel_fn':
            assert ast.dump(before[name]) == ast.dump(after[name]), name
    entry = copy.deepcopy(after['kernel_fn'])
    del entry.body[1]
    assert ast.dump(entry) == ast.dump(before['kernel_fn'])
    compile(source, str(output), 'exec')
    output.write_text(source)
    print('Standalone source SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
