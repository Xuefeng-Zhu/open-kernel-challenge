"""Local source transformation; select a verified parent before generating E3."""
import argparse
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent


def transform(source):
    tree=ast.parse(source)
    fused=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fused')
    loop=next(n for n in fused.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='start')
    lines=source.splitlines(keepends=True)
    original=lines[loop.lineno-1:loop.end_lineno]
    assert original[0]=='    for start in range(s_lo, s_hi, BLOCK_N):\n'
    pipelined=['        for start in tl.range(s_lo, s_hi, BLOCK_N, num_stages=3):\n']+['    '+s for s in original[1:]]
    lines[loop.lineno-1:loop.end_lineno]=['    if PIPELINED_HISTORY:\n']+pipelined+['    else:\n']+['    '+s for s in original]
    out=''.join(lines)
    out=out.replace('           L: tl.constexpr,','           PIPELINED_HISTORY: tl.constexpr, L: tl.constexpr,',1)
    out=out.replace('        L=length, G=g,','        PIPELINED_HISTORY=(window == 0 and batch in (16, 32)), L=length, G=g,',1)
    compile(out,'pipeline.py','exec')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',choices=['submission.py','fixed_chunks.py'])
    args=p.parse_args();dest=HERE/'pipeline.py'
    if dest.exists():raise RuntimeError('Preserve an existing candidate; no overwrite')
    dest.write_text(transform((HERE/args.parent).read_text()))
