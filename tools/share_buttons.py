#!/usr/bin/env python3
"""Add the share bar (LinkedIn, Facebook, X, copy link) to every .html page. Run after the page builders.

Each page gets assets/css/share.css in <head> and assets/js/share.js before </body>; the script builds the bar at the
end of <main>, using the page's canonical URL. Safe to run repeatedly.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
n = 0
for f in sorted(ROOT.rglob("*.html")):
    rel = f.relative_to(ROOT)
    if f.name.startswith("google") or rel.parts[0] in ("tools", "node_modules") or rel.parts[0].startswith("."):
        continue
    s = f.read_text()
    if "assets/js/share.js" in s:
        continue
    root = "../" * (len(rel.parts) - 1)
    if "</head>" not in s or "</body>" not in s:
        print("skipped:", rel)
        continue
    s = s.replace("</head>", f'<link rel="stylesheet" href="{root}assets/css/share.css">\n</head>', 1)
    s = s.replace("</body>", f'<script src="{root}assets/js/share.js" defer></script>\n</body>', 1)
    f.write_text(s)
    n += 1
print(f"share bar added to {n} pages")
