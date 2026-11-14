#!/usr/bin/env python3
"""Register extra panel guides (built from tools/panel_selector/guides/<slug>.md) in the Panel-Data Guide tool page.

Patches tools/panel_selector/template.html and panel-data-guide/index.html in the same way (model card, a leaf "Another goal?"
next to "N and T" in the decision tree, a guided question), adds the new guide to the "All estimator guides" row of every
panel guide page, and appends its link-preview card to tools/og_cards_site.json. Idempotent.
usage: python3 tools/panel_selector/add_panel_model.py twfe [dpthresh ...]
"""
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PD = ROOT / "panel-data-guide"

EXTRA = {
    "twfe": dict(key="TWFE", short="Two-way FE, clustered",
                 name="Two-way fixed effects with clustered standard errors", when="Unit and time effects, with honest standard errors",
                 why="You already know a fixed-effects model is appropriate, and the question is inference. Errors are correlated within units over time, so default standard errors are far too small. Cluster at the level of the treatment or the shocks, check the choice with two-way clustering and the wild cluster bootstrap, and be careful when units adopt a policy at different dates.",
                 steps=["Unit and year effects", "Where the variation is", "Cluster-robust standard errors", "Wild cluster bootstrap", "Placebo test of the standard errors", "Staggered treatment warning"],
                 lec=[["../teaching/econometrics/lectures/07-panel-data.html", "Lecture 07: Static panels"]],
                 q=("I have a fixed-effects model and want correct standard errors", "Clustered, two-way clustered and wild bootstrap inference")),
}


def js(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def patch_tool(path, slug):
    e = EXTRA[slug]; k = e["key"]
    s = path.read_text()
    if f"\n {k}:{{name:" in s:
        return False
    lec = "[" + ",".join(f"[{js(a)},{js(b)}]" for a, b in e["lec"]) + "]"
    steps = "[" + ",".join(js(x) for x in e["steps"]) + "]"
    s = s.replace("\n};\nconst $=s=>document", f",\n {k}:{{name:{js(e['name'])},short:{js(e['short'])},when:{js(e['when'])},why:{js(e['why'])},steps:{steps},lec:{lec}}}\n}};\nconst $=s=>document", 1)
    s = s.replace("PTH:'pthresh'};", f"PTH:'pthresh',{k}:{js(slug)}}};", 1)
    first = "goal:[" not in s
    if first:
        s = s.replace("TS:[792,92,112,'Time series',1],", "TS:[792,92,112,'Time series',1],goal:[250,92,136,'Another goal?',1],", 1)
        s = s.replace("const E=[['start','nt'],", "const E=[['start','nt'],['nt','goal','other goal'],", 1)
        s = s.replace("TL={start:'Your panel',", "TL={goal:'Another goal',start:'Your panel',", 1)
        s = s.replace("function pathTo(m,R){const S=['start','nt'],sh=['dyn'],lo=['cd','ur'];", "const OTHER=[/*OTHER*/];\nfunction pathTo(m,R){const S=['start','nt'],sh=['dyn'],lo=['cd','ur'];if(OTHER.includes(m))return S.concat(['goal']);", 1)
        s = s.replace("['I think the effect differs across regimes of another variable','A threshold regression','PTH',['start','nt','dyn','st']]]},",
                      "['I think the effect differs across regimes of another variable','A threshold regression','PTH',['start','nt','dyn','st']],['Something else: quantiles, spillovers, cointegration, policy timing, inference','Other panel methods','qx',['start','nt','goal']]]},\n qx:{t:'What is the goal?',o:[/*QX*/]},", 1)
    s = s.replace("/*OTHER*/", f"{js(k)},/*OTHER*/", 1)
    s = s.replace("/*QX*/", f"[{js(e['q'][0])},{js(e['q'][1])},{js(k)}],/*QX*/", 1)
    path.write_text(s)
    return True


def patch_others(slug):
    e = EXTRA[slug]; link = f'<a class="" href="{slug}.html">{e["short"]}</a>'
    for p in sorted(PD.glob("*.html")):
        if p.name in ("index.html", f"{slug}.html"):
            continue
        s = p.read_text()
        i = s.find('<div class="others">')
        if i < 0 or f'href="{slug}.html"' in s[i:i + 3000]:
            continue
        j = s.find("</div></div>", i)
        p.write_text(s[:j] + link + s[j:])


def add_card(slug):
    f = ROOT / "tools" / "og_cards_site.json"
    cards = json.loads(f.read_text())
    out = f"assets/og/panel-data-guide/{slug}.jpg"
    if any(c["out"] == out for c in cards):
        return
    head = (HERE / "guides" / f"{slug}.md").read_text().split("\n---\n", 1)[0]
    meta = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}
    cards.append(dict(out=out, kicker="Panel Data Guide · Model guide", title=meta["title"], sub="A step-by-step guide to writing the empirical paper, with Python code, real output and sample text."))
    f.write_text(json.dumps(cards, ensure_ascii=False, indent=1))


def main():
    for slug in sys.argv[1:]:
        add_card(slug)
        for f in (HERE / "template.html", PD / "index.html"):
            patch_tool(f, slug)
        patch_others(slug)
        print("registered", slug)


if __name__ == "__main__":
    main()
