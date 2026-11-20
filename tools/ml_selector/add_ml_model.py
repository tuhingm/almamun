#!/usr/bin/env python3
"""Register extra machine-learning guides (built from tools/ml_selector/guides/<slug>.md by tools/bayes_markov/build_guides.py ml <slug>)
in the Machine Learning Guide: adds a card under "More methods" on the tool page (template and ml-guide/index.html), adds the guide to the
"All method guides" row of every guide page, and appends its link-preview card to tools/og_cards_site.json. Idempotent.
usage: python3 tools/ml_selector/add_ml_model.py classification [interpretation ...]
"""
import html, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MD = ROOT / "ml-guide"
MARK = "<!--MORE-ML-->"


def meta(slug):
    head = (HERE / "guides" / f"{slug}.md").read_text().split("\n---\n", 1)[0]
    return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}


def card(slug, m):
    return (f'<article class="gcard"><span class="tag">{html.escape(m["tag"] if "tag" in m else "Method guide")}</span><h3>{html.escape(m["title"])}</h3><p class="when">{html.escape(m["when"])}</p>'
            f'<p>{html.escape(m["data"])}</p><a href="{slug}.html">Open the guide →</a></article>\n')


def patch_index(path, slug, m):
    s = path.read_text()
    if f'href="{slug}.html"' in s.split(MARK)[0][-60000:] and MARK in s and f'<a href="{slug}.html">Open the guide' in s:
        return
    if MARK not in s:
        a = '<div class="gcards extra">\n'
        assert a in s, path
        s = s.replace(a, '<div class="gcards extra">\n' + MARK + "\n", 1)
    s = s.replace(MARK, card(slug, m) + MARK, 1)
    path.write_text(s)


def patch_others(slug, m):
    link = f'<a class="" href="{slug}.html">{m["short"]}</a>'
    for p in sorted(MD.glob("*.html")):
        if p.name in ("index.html", f"{slug}.html"):
            continue
        s = p.read_text(); i = s.find('<div class="others">')
        if i < 0 or f'href="{slug}.html"' in s[i:i + 4000]:
            continue
        j = s.find("</div></div>", i)
        p.write_text(s[:j] + link + s[j:])


def add_og(slug, m):
    f = ROOT / "tools" / "og_cards_site.json"; cards = json.loads(f.read_text()); out = f"assets/og/ml-guide/{slug}.jpg"
    if any(c["out"] == out for c in cards):
        return
    cards.append(dict(out=out, kicker="Machine Learning Guide · Model guide", title=m["title"], sub="A step-by-step guide to writing the empirical paper, with Python code, real output and sample text."))
    f.write_text(json.dumps(cards, ensure_ascii=False, indent=1))


def main():
    for slug in sys.argv[1:]:
        m = meta(slug); add_og(slug, m)
        for p in (HERE / "template.html", MD / "index.html"):
            patch_index(p, slug, m)
        patch_others(slug, m); print("registered", slug)


if __name__ == "__main__":
    main()
