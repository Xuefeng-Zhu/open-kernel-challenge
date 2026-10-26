"""Small auditable sanity check, not the challenge's submission validator."""
import ast
import argparse
import json
from pathlib import Path

from harness import ROOT, file_hash


def local_policy_errors(tree):
    """Project preflight conventions, not an official legality/security proof.

    The 2026-10-26 rules audit verified the explicit load_inline/nvcc
    compilation exception. Keep that exception narrow: exact import and a
    direct call at module initialization or in prepare, never kernel_fn.
    This does not authorize subprocess imports, downloads or harness access.
    Ordinary non-tensor diagnostics are allowed outside kernel_fn; AST alone
    cannot establish whether a value hides a synchronization.
    """
    errors = []

    class Policy(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            if not all(n.name in {"torch", "triton", "triton.language"} for n in node.names):
                errors.append("Unexpected import")

        def visit_ImportFrom(self, node):
            imports = [(n.name, n.asname) for n in node.names]
            allowed = node.module == "typing" or (
                node.module == "triton.tools.tensor_descriptor" and imports == [("TensorDescriptor", None)]
            ) or (
                node.module == "torch.utils.cpp_extension" and imports == [("load_inline", None)]
            )
            if not allowed:
                errors.append("Unexpected import")

        def visit_Call(self, node):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"item", "cpu", "numpy", "synchronize", "open", "eval", "exec"}:
                errors.append(name)
            if name == "print" and "kernel_fn" in self.functions:
                errors.append("print inside kernel_fn requires manual synchronization review")
            if name == "load_inline" and self.functions not in ([], ["prepare"]):
                errors.append("load_inline must be direct initialization/prepare code in this preflight")
            self.generic_visit(node)

    Policy().visit(tree)
    return errors


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
    assert not local_policy_errors(tree), "; ".join(local_policy_errors(tree))
    manifest = json.loads((ROOT / "official/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert file_hash(ROOT / "official" / name) == expected, f"Official snapshot changed: {name}"
    print("PASS: Python AST, entrypoints, basic import/sync sanity checks, and official snapshot SHA256s")
    print("CUDA graph capture and official validation still require the target environment")


if __name__ == "__main__":
    main()
