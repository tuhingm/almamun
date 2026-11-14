#!/usr/bin/env python3
"""Refresh the 'All method guides' row of every page in a Bayesian or Markov guide folder, so that newly added guides are linked from all pages.
usage: python3 tools/bayes_markov/add_guide.py bayesian|markov
"""
import pathlib, re, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_guides import SECTIONS, meta_of, ROOT  # noqa: E402

key = sys.argv[1]; S = SECTIONS[key]; folder = ROOT / S["folder"]
metas = {s: meta_of(HERE / "guides" / key / f"{s}.md") for s in S["order"] if (HERE / "guides" / key / f"{s}.md").exists()}
for s in S["order"]:
    p = folder / f"{s}.html"
    if not p.exists():
        continue
    row = "".join(f'<a class="{"cur" if t == s else ""}" href="{t}.html">{metas[t]["short"]}</a>' for t in S["order"] if t in metas and (folder / f"{t}.html").exists())
    h = p.read_text()
    new = re.sub(r'(<div class="others"><span class="label">All method guides</span><div><a class="back" href="index.html">[^<]*</a>).*?(</div></div>)',
                 lambda m: m.group(1) + row + m.group(2), h, count=1, flags=re.S)
    if new != h:
        p.write_text(new)
print("nav rows refreshed:", key)
