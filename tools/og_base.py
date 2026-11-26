#!/usr/bin/env python3
"""Make sure every public .html page has a complete set of link-preview tags and a preview image.

Adds og:*/twitter:*/canonical to pages that have none, adds og:image/twitter:card where only some are present,
and writes a card definition to tools/og_cards_extra.json for every referenced preview image that does not exist.
Run before tools/make_og_images.py and tools/og_tags.py. Safe to run repeatedly.
"""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "https://tuhingm.github.io/almamun/"
SKIP = ("tools", "node_modules", ".git", ".github")
cards = {}


def txt(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def esc(s):
    return html.escape(s, quote=True)


def grab(s, k):
    m = re.search(rf'<meta (?:property|name)="{re.escape(k)}" content="([^"]*)"', s)
    return html.unescape(m.group(1)) if m else None


n_tags = 0
for f in sorted(ROOT.rglob("*.html")):
    rel = f.relative_to(ROOT)
    if rel.parts[0] in SKIP or f.name.startswith("google"):
        continue
    s = f.read_text(encoding="utf-8")
    if "</head>" not in s:
        continue
    path = str(rel)
    url = BASE + (path[:-10] if path.endswith("index.html") else path)
    title = grab(s, "og:title") or txt((re.search(r"<title>(.*?)</title>", s, re.S) or [0, ""])[1]) or "Tuhin Al Mamun, Ph.D."
    desc = grab(s, "og:description") or grab(s, "description")
    if not desc:
        m = re.search(r"<header[^>]*>.*?<p[^>]*>(.*?)</p>", s, re.S) or re.search(r"<main[^>]*>.*?<p[^>]*>(.*?)</p>", s, re.S) or re.search(r"<p[^>]*>(.*?)</p>", s, re.S)
        desc = txt(m.group(1)) if m else title
    if len(desc) > 190:
        desc = desc[:187].rsplit(" ", 1)[0] + "..."
    img = grab(s, "og:image")
    out = (img[len(BASE):].split("?")[0] if img and img.startswith(BASE) else f"assets/og/{path[:-5]}.jpg")
    add = []
    has = lambda k: re.search(rf'<meta (?:property|name)="{re.escape(k)}"', s)
    if not has("og:title"):
        add += [f'<meta property="og:locale" content="en_US">', f'<meta property="og:type" content="article">',
                f'<meta property="og:site_name" content="Tuhin Al Mamun, Ph.D.">',
                f'<meta property="og:title" content="{esc(title)}">', f'<meta property="og:description" content="{esc(desc)}">',
                f'<meta property="og:url" content="{url}">']
    if not has("og:image"):
        add += [f'<meta property="og:image" content="{BASE}{out}">', '<meta property="og:image:width" content="1200">', '<meta property="og:image:height" content="630">']
    if not has("twitter:card"):
        add += ['<meta name="twitter:card" content="summary_large_image">']
    if 'rel="canonical"' not in s:
        add += [f'<link rel="canonical" href="{url}">']
    if not has("description") and not re.search(r'name="description"', s):
        add += [f'<meta name="description" content="{esc(desc)}">']
    if add:
        s = s.replace("</head>", "\n" + "\n".join(add) + "\n</head>", 1)
        f.write_text(s, encoding="utf-8")
        n_tags += 1
    if not (ROOT / out).exists():
        t = re.sub(r"^Moved:\s*", "", title)
        parts = [p.strip() for p in t.split("|")]
        kicker = " · ".join(parts[1:3]) if len(parts) > 1 else "Tuhin Al Mamun, Ph.D."
        cards[out] = dict(out=out, kicker=kicker[:70], title=parts[0], sub=desc[:140])
(ROOT / "tools" / "og_cards_extra.json").write_text(json.dumps(list(cards.values()), indent=1, ensure_ascii=False))
print(f"tags added on {n_tags} pages; {len(cards)} preview images to render")
