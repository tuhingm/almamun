#!/usr/bin/env python3
"""Register extra causal-design guides (built from tools/causal_selector/guides/<slug>.md) in the Causal Inference Guide tool page.

Patches tools/causal_selector/template.html and causal-inference-guide/index.html in the same way (model card, a leaf "More
designs" under "How is treatment assigned?", a guided question), adds the new guide to the "All design guides" row of every
guide page, and appends its link-preview card to tools/og_cards_site.json. Idempotent.
usage: python3 tools/causal_selector/add_causal_model.py eventstudy [sdid ...]
"""
import json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CD = ROOT / "causal-inference-guide"

EXTRA = {
    "eventstudy": dict(key="ES", short="Event study",
                 name="Event study with pre-trend sensitivity", when="Dynamic effects around a policy, and how much a pre-trend would matter",
                 why="The question is how the effect evolves around adoption and whether it could be a pre-existing trend. Estimate event-time effects with an interaction-weighted estimator (Sun–Abraham) that does not mix cohorts, test the pre-trends and their power, and bound the post-treatment effects under smoothness restrictions on the violation.",
                 steps=["Event time and cohort coverage", "Conventional event study", "Sun–Abraham interaction-weighted estimator", "Pre-trend test and its power", "Sensitivity to violations of parallel trends"],
                 lec=[],
                 q=("I want dynamic effects around a policy and a check on pre-trends", "Event study with Sun–Abraham and sensitivity analysis")),
    "sdid": dict(key="SDID", short="Synthetic DiD",
                 name="Synthetic difference-in-differences", when="One or a few treated units; neither parallel trends nor a perfect synthetic-control fit is credible alone",
                 why="The treated unit differs in level from the comparison pool and the pre-treatment fit is imperfect. Synthetic DiD reweights comparison units and pre-treatment periods and then runs a weighted DiD, so it allows level differences and gives some protection against non-parallel trends. Compare it with DiD and synthetic control, and use placebo inference.",
                 steps=["Design and comparison pool", "DiD and synthetic control on the same data", "Unit and time weights", "Placebo inference", "Regularisation, in-time placebo and scale"],
                 lec=[],
                 q=("I have one treated unit and want to combine DiD with synthetic control", "Synthetic difference-in-differences")),
}


def js(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def patch_tool(path, slug):
    e = EXTRA[slug]; k = e["key"]
    s = path.read_text()
    if f"\n {k}:{{name:" in s:
        if f"{k}:{js(slug)}" not in s:
            s = re.sub(r"(const GUIDE=\{[^}]*)\};", lambda m: m.group(1) + f",{k}:{js(slug)}}};", s, count=1)
            path.write_text(s)
        return False
    lec = "[" + ",".join(f"[{js(a)},{js(b)}]" for a, b in e["lec"]) + "]"
    steps = "[" + ",".join(js(x) for x in e["steps"]) + "]"
    s = s.replace("\n};\nconst $=s=>document", f",\n {k}:{{name:{js(e['name'])},short:{js(e['short'])},when:{js(e['when'])},why:{js(e['why'])},steps:{steps},lec:{lec}}}\n}};\nconst $=s=>document", 1)
    s = re.sub(r"(const GUIDE=\{[^}]*)\};", lambda m: m.group(1) + f",{k}:{js(slug)}}};", s, count=1)
    first = "oth:[" not in s
    if first:
        s = s.replace("as:[380,92,200,'How is treatment assigned?'],", "as:[380,92,200,'How is treatment assigned?'],oth:[380,160,124,'More designs',1],", 1)
        s = s.replace("['as','ot'],['cut','rdc']", "['as','ot'],['as','oth'],['cut','rdc']", 1)
        s = s.replace("TL={start:'Your design',", "TL={oth:'More designs',start:'Your design',", 1)
        s = s.replace("function pathTo(m){const S=['start','as'];", "const OTHER=[/*OTHER*/];\nfunction pathTo(m){const S=['start','as'];if(OTHER.includes(m))return S.concat(['oth']);", 1)
        s = s.replace("'NT',['start','as','ot']]]},", "'NT',['start','as','ot']],['Something else: event-study dynamics, matching, shift-share, synthetic DiD','Other designs','qx',['start','as','oth']]]},\n qx:{t:'What is the goal?',o:[/*QX*/]},", 1)
    s = s.replace("/*OTHER*/", f"{js(k)},/*OTHER*/", 1)
    s = s.replace("/*QX*/", f"[{js(e['q'][0])},{js(e['q'][1])},{js(k)}],/*QX*/", 1)
    path.write_text(s)
    return True


def patch_others(slug):
    e = EXTRA[slug]; link = f'<a class="" href="{slug}.html">{e["short"]}</a>'
    for p in sorted(CD.glob("*.html")):
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
    out = f"assets/og/causal-inference-guide/{slug}.jpg"
    if any(c["out"] == out for c in cards):
        return
    head = (HERE / "guides" / f"{slug}.md").read_text().split("\n---\n", 1)[0]
    meta = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}
    cards.append(dict(out=out, kicker="Causal Inference Guide · Model guide", title=meta["title"], sub="A step-by-step guide to writing the empirical paper, with Python code, real output and sample text."))
    f.write_text(json.dumps(cards, ensure_ascii=False, indent=1))


def main():
    for slug in sys.argv[1:]:
        add_card(slug)
        for f in (HERE / "template.html", CD / "index.html"):
            patch_tool(f, slug)
        patch_others(slug)
        print("registered", slug)


if __name__ == "__main__":
    main()
