"""Small auditable sanity check, not the challenge's submission validator."""
import ast
import json

from harness import ROOT, file_hash


def main():
    source = (ROOT / "submission.py").read_text()
    assert source.splitlines()[0] == "# okc: problem=muse_bf16_proj"
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"prepare", "kernel_fn"} <= functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name in {"torch"} for n in node.names), "Unexpected import"
        if isinstance(node, ast.ImportFrom):
            assert node.module == "typing", "Unexpected import"
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
