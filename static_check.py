"""Small auditable sanity check, not the challenge's submission validator."""
import ast
import argparse
import json
from pathlib import Path

from harness import ROOT, file_hash


def plain_jit_globals(tree):
    """Flag a known target-compiler error class; not a Triton compilation proof."""
    globals_by_name = {}
    jit_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    globals_by_name.setdefault(target.id, []).append(node.value)
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list
        ):
            jit_functions.append(node)
    errors = []
    for fn in jit_functions:
        local = {a.arg for a in fn.args.args}
        local.update(n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for name in sorted(loaded - local):
            # Module numeric assignments accessed directly inside JIT are rejected by
            # the official image unless instantiated as tl.constexpr. Constants passed
            # from host as constexpr arguments are local and do not trigger this check.
            if any(isinstance(value, ast.Constant) and isinstance(value.value, (int, float, bool))
                   for value in globals_by_name.get(name, [])):
                errors.append(f"{fn.name}: plain numeric global {name}; pass as constexpr argument or instantiate tl.constexpr")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.py")
    args = parser.parse_args()
    source = args.submission.read_text()
    assert source.splitlines()[0] == "# okc: problem=muse_bf16_proj"
    tree = ast.parse(source)
    assert not plain_jit_globals(tree), "; ".join(plain_jit_globals(tree))
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"prepare", "kernel_fn"} <= functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name in {"torch", "triton", "triton.language"} for n in node.names), "Unexpected import"
        if isinstance(node, ast.ImportFrom):
            assert node.module == "typing" or (
                node.module == "triton.tools.tensor_descriptor"
                and [(n.name, n.asname) for n in node.names] == [("TensorDescriptor", None)]
            ), "Unexpected import"
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in {"item", "cpu", "numpy", "synchronize", "print", "open", "eval", "exec"}, name
    manifest = json.loads((ROOT / "official/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert file_hash(ROOT / "official" / name) == expected, f"Official snapshot changed: {name}"
    print("PASS: Python AST, entrypoints, basic import/sync sanity checks, and official snapshot SHA256s")
    print("CUDA graph capture and official validation still require the target environment")


if __name__ == "__main__":
    main()
