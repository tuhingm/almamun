#!/usr/bin/env python3
"""Complete the link-preview tags on every .html page. Run last, after the page builders.

Each page already has og:title, og:description and og:image. This adds the tags some apps need (og:image:alt,
og:image:type, twitter:title, twitter:description, twitter:image) and a version suffix on the image URL so that apps
which cache previews fetch the current image. Safe to run repeatedly.
"""
import html, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = "3"
EXTRA = re.compile(r'\s*<meta (?:property|name)="(?:og:image:alt|og:image:type|og:image:secure_url|twitter:title|twitter:description|twitter:image|twitter:image:alt)" content="[^"]*">')
n = 0
for f in sorted(ROOT.rglob("*.html")):
    rel = f.relative_to(ROOT)
    if f.name.startswith("google") or rel.parts[0] in ("tools", "node_modules"):
        continue
    s = f.read_text()
    get = lambda k: (re.search(rf'<meta property="{k}" content="([^"]*)"', s) or [None, None])[1]
    title, desc, img = get("og:title"), get("og:description"), get("og:image")
    if not (title and desc and img):
        print("skipped (no og tags):", rel)
        continue
    s = EXTRA.sub("", s)
    base = img.split("?")[0]
    s = s.replace(f'<meta property="og:image" content="{img}">', f'<meta property="og:image" content="{base}?v={VERSION}">', 1)
    alt = html.escape(f"Preview card: {html.unescape(title)}", quote=True)
    add = (f'\n<meta property="og:image:type" content="image/jpeg"><meta property="og:image:alt" content="{alt}">'
           f'\n<meta name="twitter:title" content="{title}"><meta name="twitter:description" content="{desc}">'
           f'<meta name="twitter:image" content="{base}?v={VERSION}"><meta name="twitter:image:alt" content="{alt}">')
    s = re.sub(r'(<meta name="twitter:card" content="[^"]*">)', lambda m: m.group(1) + add, s, count=1)
    f.write_text(s)
    n += 1
print(f"preview tags completed on {n} pages")
