#!/usr/bin/env python3
"""Insert registry entries (a text file of dict(...) lines) before the end of the MODELS list in oes_engine/__init__.py.
  python3 tools/add_registry.py entries.txt"""
import pathlib, sys
p = pathlib.Path(__file__).resolve().parents[1] / "oes/engine/oes_engine/__init__.py"
s = p.read_text()
new = pathlib.Path(sys.argv[1]).read_text().rstrip("\n") + "\n"
m = "]\n\nSTAT_MODELS"
assert m in s
s = s.replace(m, new + m, 1)
if 'FI = "finance-guide/examples/"' not in s:
    s = s.replace('OX = "oes/examples/"', 'OX = "oes/examples/"\nFI = "finance-guide/examples/"\nPR = "productivity-guide/examples/"\nNP = "nonparametric-guide/examples/"', 1)
p.write_text(s)
