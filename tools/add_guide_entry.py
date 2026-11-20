#!/usr/bin/env python3
"""Register a guide in a section that grows model by model (finance, productivity, nonparam).
  python3 tools/add_guide_entry.py <section> <slug> "<situation>" "<method name>" [data file ...]"""
import json, pathlib, sys
P = pathlib.Path(__file__).resolve().parent / "bayes_markov" / "extra_sections.json"
d = json.loads(P.read_text()) if P.exists() else {}
sec, slug, sit, meth, *data = sys.argv[1:]
ent = d.setdefault(sec, {"entries": []})["entries"]
ent[:] = [e for e in ent if e["slug"] != slug] + [dict(slug=slug, situation=sit, method=meth, data=data)]
P.write_text(json.dumps(d, indent=1))
