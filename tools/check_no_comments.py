import ast
import io
import sys
import tokenize
from pathlib import Path


def violations(path):
    text = Path(path).read_text()
    found = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            found.append((tok.start[0], "comment"))
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                body = node.body[0]
                found.append((body.lineno, "docstring"))
    return found


def collect(args):
    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(p.rglob("*.py"))
        elif p.suffix == ".py":
            files.append(p)
    return [f for f in files if "third_party" not in f.parts and ".venv" not in f.parts]


def main(argv):
    targets = argv or ["."]
    bad = 0
    for f in collect(targets):
        for line, kind in violations(f):
            print(f"{f}:{line}: {kind}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
