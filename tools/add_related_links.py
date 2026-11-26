#!/usr/bin/env python3
"""Add relevant internal links (SEO + navigation). Run after the page builders. Safe to run repeatedly.

1. Statistics Division method pages and validation reports: a "Related" block (same-group methods, report, index, course).
2. Static links from hub pages to guide pages that no other page links to.
3. OES page and Statistics course: a static link to the Statistics Division guide.
"""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
S, E = "<!--related-links-->", "<!--/related-links-->"
esc = html.escape
CAT = json.loads((ROOT / "oes/engine/stat_catalog.json").read_text())["models"]
groups = {}
for m in CAT:
    groups.setdefault(m["group"], []).append(m)
slug = lambda m: pathlib.Path(m["guide"]).stem


def put(path, block, anchor):
    f = ROOT / path
    s = f.read_text(encoding="utf-8")
    s = re.sub(re.escape(S) + r".*?" + re.escape(E), "", s, flags=re.S)
    if anchor not in s:
        print("anchor missing:", path); return
    f.write_text(s.replace(anchor, f"{S}{block}{E}{anchor}", 1), encoding="utf-8")


def links(items):
    return "<ul>" + "".join(f'<li><a href="{h}">{esc(t)}</a></li>' for h, t in items) + "</ul>"


for m in CAT:
    sl, g = slug(m), m["group"]
    same = [x for x in groups[g] if x is not m]
    # method page (stat-guide/<slug>.html)
    items = [(f"{slug(x)}.html", x["name"]) for x in same]
    block = (f'<section><h2>Related {esc(g.lower())} methods</h2>{links(items)}' if items else f'<section><h2>More in the Statistics Division</h2>')
    block += ('<p>See also: <a href="index.html">all Statistics Division methods</a>, '
              f'the <a href="reports/{sl}.html">validation report for {esc(m["name"])}</a>, and the '
              '<a href="../teaching/statistics/index.html">Statistics course</a>.</p></section>')
    put(f"stat-guide/{sl}.html", block, "</div></main>")
    # validation report (stat-guide/reports/<slug>.html)
    items = [(f"{slug(x)}.html", f'{x["name"]}: validation report') for x in same]
    block = (f'<section><h2>Related validation reports</h2>{links(items)}' if items else '<section><h2>More in the Statistics Division</h2>')
    block += (f'<p>Back to the <a href="../{sl}.html">{esc(m["name"])} guide</a> or the '
              '<a href="../index.html">Statistics Division index</a>.</p></section>')
    put(f"stat-guide/reports/{sl}.html", block, "</div></main>")

# hub pages: static links to guide pages with no other inbound link
HUBS = {
    "microeconometrics-guide/index.html": [("conditional-mixed-logit.html", "Conditional and Mixed Logit")],
    "bayesian-guide/index.html": [("count-regression.html", "Bayesian Count Regression")],
    "markov-guide/index.html": [("markov-chain.html", "Markov Chain Transition Model")],
    "panel-data-guide/index.html": [("interactive-fixed-effects.html", "Interactive Fixed Effects (Bai 2009)"),
                                    ("lectures/macro-panel-econometrics.html", "Lecture: Macroeconomic Panel Econometrics")],
    "time-series-guide/index.html": [("time-varying-parameters.html", "Time-Varying-Parameter Regression (Kalman Filter)")],
}
for path, items in HUBS.items():
    block = ('<div class="wrap" style="padding:8px 0 26px;font-size:14px"><strong>Also in this guide:</strong> ' +
             " · ".join(f'<a href="{h}">{esc(t)}</a>' for h, t in items) + "</div>")
    put(path, block, "<footer")

# in-text links
def sub(path, old, new):
    f = ROOT / path; s = f.read_text(encoding="utf-8")
    if new in s: return
    if old not in s: print("text missing:", path); return
    f.write_text(s.replace(old, new, 1), encoding="utf-8")

sub("oes/index.html", "and a statistics division (ANOVA", 'and a <a href="../stat-guide/index.html">statistics division</a> (ANOVA')
sub("teaching/statistics/index.html", "and multivariate analysis.</p>",
    'and multivariate analysis. The methods are documented in the <a href="../../stat-guide/index.html">OES Statistics Division guide</a>.</p>')
print("done")
