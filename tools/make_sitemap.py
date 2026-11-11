#!/usr/bin/env python3
"""Write sitemap.xml for the GitHub Pages site. Run after tools/build_courses.py, then tools/make_og_images.py.

Every .html page is listed (except the Google verification file); lastmod is the date of the file's last git commit,
or today for uncommitted files. index.html pages are listed under their folder URL.
A project site cannot serve /robots.txt at the host root, so submit sitemap.xml in Google Search Console instead.
"""
import datetime, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = "https://tuhingm.github.io/almamun/"
today = datetime.date.today().isoformat()


def lastmod(p):
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return today if (dirty or not out) else out


def priority(rel):
    if rel in ("index.html",): return "1.0"
    if rel in ("working-papers/index.html", "teaching/index.html"): return "0.9"
    if rel.count("/") == 2 and rel.endswith("index.html"): return "0.8"
    return "0.6" if "/lectures/" in rel else "0.5"


pages = sorted(p for p in ROOT.rglob("*.html")
               if not p.name.startswith("google") and not any(part.startswith((".", "tools", "node_modules")) for part in p.relative_to(ROOT).parts))
rows = []
for p in pages:
    rel = p.relative_to(ROOT).as_posix()
    url = SITE + (rel[:-len("index.html")] if rel.endswith("index.html") else rel)
    rows.append(f"  <url><loc>{url}</loc><lastmod>{lastmod(p)}</lastmod><priority>{priority(rel)}</priority></url>")
(ROOT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                                  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "\n".join(rows) + "\n</urlset>\n")
print(f"sitemap.xml: {len(rows)} pages")
