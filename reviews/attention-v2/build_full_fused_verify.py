"""Reproduce isolated A2-07 source from its exact checked parent; no imports/upload."""
import ast
import copy
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT_SHA = 'c40127a320442953630276206b0dfd0ebadd550cc58bdbfe124d800b31b13b18'


def build():
    raw = (HERE/'mixed_fa3_fused_verify.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    text = raw.decode()
    needle = 'def kernel_fn(data: Inputs) -> torch.Tensor:\n    cfg = data["config"]\n'
    assert text.count(needle) == 1
    guard = ('    if (cfg["batch"] == 1 and cfg["q_len"] == 8\n'
             '            and cfg["window"] == 0 and cfg["rope_theta"] == 0):\n'
             '        return _fa3_full_verify_fused(data)\n')
    text = text.replace(needle, (HERE/'fa3_full_prologue_component.py').read_text()+'\n\n'+needle+guard)
    lines = text.splitlines(keepends=True)
    lines[1] = '# A2-07: only full B1/L8 adopts the same prologue with ROPE=False.\n'
    text = ''.join(lines)
    old = {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    assert new.keys()-old.keys() == {'_fa3_full_verify_fused'}
    for name, node in old.items():
        actual = copy.deepcopy(new[name])
        if name == 'kernel_fn': del actual.body[1]
        assert ast.dump(node) == ast.dump(actual), name
    compile(text, 'A2-07', 'exec')
    return text


if __name__ == '__main__':
    out = (HERE/(sys.argv[1] if len(sys.argv)>1 else 'mixed_fa3_fused_full_verify1.py')).resolve()
    assert out.parent == HERE and not out.exists()
    out.write_text(build())
    print(hashlib.sha256(out.read_bytes()).hexdigest())
