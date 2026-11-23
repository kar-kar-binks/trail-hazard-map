import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_comments_or_docstrings():
    r = subprocess.run([sys.executable, str(ROOT / "tools/check_no_comments.py"), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
