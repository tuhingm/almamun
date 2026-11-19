#!/usr/bin/env python3
"""Write the method counts of the OES page (meta tags, structured data and the header strip) from the catalogues.

    python3 tools/oes_counts.py

Reads oes/engine/catalog.json and stat_catalog.json (written by tools/oes_sync.py) and updates oes/index.html, so the totals
shown to visitors and to search engines always equal the number of models the page can run."""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
econ = json.loads((ROOT / "oes/engine/catalog.json").read_text())["models"]
stat = json.loads((ROOT / "oes/engine/stat_catalog.json").read_text())["models"]
total, groups = len(econ) + len(stat), len({m["group"] for m in econ + stat})
page = ROOT / "oes/index.html"
s = page.read_text()
s = re.sub(r"run \d+ methods \(\d+ econometric models and \d+ statistical methods\)", f"run {total} methods ({len(econ)} econometric models and {len(stat)} statistical methods)", s)
s = re.sub(r'(<strong id="n-models">)\d+(</strong> methods</span><span><strong>)\d+(</strong> method groups)', rf"\g<1>{total}\g<2>{groups}\g<3>", s)
page.write_text(s)
print(f"OES page: {total} methods ({len(econ)} econometric, {len(stat)} statistical) in {groups} groups")
