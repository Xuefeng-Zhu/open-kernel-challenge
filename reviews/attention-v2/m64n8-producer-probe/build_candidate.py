"""Assemble one scoped Attention candidate from the immutable selected source."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SELECTED_SHA = "3da22a162e3c7567b10ba87713e187f7f086376b1d5198edb972fd9b8d0a7cd8"
OLD_LAUNCH = "    _fused[(batch, hkv, splits)](\n"
NEW_LAUNCH = """    # Isolated L1/G8/page1 producer; original allocation/grid/merge ABI retained.
    use_m64n8 = (length == 1 and splits > 1
                  and (g, hq, hkv, d, page) == (8, 16, 2, 64, 1)
                  and ((batch in (1, 16, 32) and window == 2048
                        and cfg["rope_theta"] == 500000)
                       or (batch == 32 and window == 0 and not rope)))
    producer = _m64n8_fused if use_m64n8 else _fused
    producer[(batch, hkv, splits)](
"""
MARKER = "\n\n# Owned full L1 M64N8 producer; see its source-bound experiment card.\n"


def assemble(selected: bytes, kernel: str) -> str:
    if hashlib.sha256(selected).hexdigest() != SELECTED_SHA:
        raise ValueError("Selected source changed")
    source = selected.decode()
    if source.count(OLD_LAUNCH) != 1:
        raise ValueError("Expected one selected producer launch")
    tree = ast.parse(kernel)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if names.count("_m64n8_fused") != 1:
        raise ValueError("Expected one integrated producer")
    def bound_names(module):
        found = set()
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                found.update(n.asname or n.name.split('.')[0] for n in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                found.update(n.id for t in targets for n in ast.walk(t)
                             if isinstance(n, ast.Name))
        return found
    selected_names = bound_names(ast.parse(source))
    if selected_names.intersection(bound_names(tree)):
        raise ValueError("Owned kernel shadows selected definitions")
    result = source.replace(OLD_LAUNCH, NEW_LAUNCH) + MARKER + kernel
    ast.parse(result)
    recovered = result.split(MARKER)[0].replace(NEW_LAUNCH, OLD_LAUNCH)
    if recovered.encode() != selected:
        raise ValueError("Unexpected selected-source mutation")
    if result.splitlines()[0] != "# okc: problem=muse_attention_decode":
        raise ValueError("Official problem marker changed")
    return result


def main() -> None:
    selected = (HERE.parent / "fusion-lookahead-probe/candidate.py").read_bytes()
    candidate = assemble(selected, (HERE / "kernel.py").read_text())
    (HERE / "candidate.py").write_text(candidate)
    print(hashlib.sha256(candidate.encode()).hexdigest())


if __name__ == "__main__":
    main()
