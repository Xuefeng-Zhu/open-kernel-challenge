"""Small auditable sanity check, not the challenge's submission validator."""
import ast
import argparse
import json
from pathlib import Path

from harness import ROOT, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.py")
    args = parser.parse_args()
    source = args.submission.read_text()
    assert source.splitlines()[0] == "# okc: problem=muse_bf16_proj"
    tree = ast.parse(source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"prepare", "kernel_fn"} <= functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name in {"torch", "triton", "triton.language"} for n in node.names), "Unexpected import"
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
