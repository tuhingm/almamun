#!/usr/bin/env python3
"""Fit page titles and meta descriptions to what search results display. Run last, after site_polish.py and og_tags.py.

Only <head> metadata changes; nothing visible on the page itself.
  - Titles: Google shows about 60 characters. Long titles keep the page name first and drop the parts that do not
    fit: first the course/guide name, then the shortened author name ("Tuhin Al Mamun"). A page name that is itself
    longer than 60 characters is kept whole.
  - Descriptions: Google shows about 155-160 characters. Longer ones are cut at the last full sentence that fits,
    or at a word boundary.
  - og:title, twitter:title, og:description, twitter:description and the WebPage name/description in the JSON-LD
    follow the same text, so search results, link previews and structured data agree.
Safe to run repeatedly.
"""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TMAX, DMAX = 60, 158
BRAND_FULL, BRAND = "Tuhin G. M. Al Mamun", "Tuhin Al Mamun"

# Pages whose wording is set here because the generated one misses the search terms people use.
FIXED = {
    "research-guide/index.html": dict(title="Time-Series Model Selection Guide | Tuhin Al Mamun"),
    "teaching/index.html": dict(desc="Free courses in advanced macroeconomics, monetary economics, econometrics, "
                                     "statistics, calculus and finance: lecture notes, problem sets and code."),
    "index.html": dict(desc="Tuhin G. M. Al Mamun (Tuhin Al Mamun), Ph.D.: economist working on monetary and fiscal policy, "
                            "HANK and DSGE models, environmental economics and econometrics."),
    "panel-data-guide/index.html": dict(desc="Interactive tool for choosing a panel model: test your panel and see whether "
                                             "pooled OLS, fixed or random effects, GMM, PMG or CS-ARDL fits, with guides."),
    "causal-inference-guide/index.html": dict(desc="Interactive tool for choosing a causal design: test your data and see whether "
                                                   "DiD, synthetic control, RDD, IV or interrupted time series fits, with guides."),
    "dsge-guide/index.html": dict(desc="Interactive DSGE workflow tool with Dynare: RBC, New Keynesian and open-economy models, "
                                       "Bayesian estimation, identification, optimal policy, OccBin."),
    "ml-guide/index.html": dict(desc="Choosing a machine-learning method in economics: LASSO, ridge, random forest, "
                                     "XGBoost, double machine learning or causal forests, with step-by-step guides."),
}


def fit_title(t, shared=False):
    """shared: the page name is used by several pages (e.g. every course's problem-set page), so the course name stays."""
    parts = [p.strip() for p in t.split(" | ")]
    if len(t) <= TMAX or len(parts) < 2 or (shared and len(parts) < 3):
        return t
    name, mid = parts[0], parts[1:-1]
    if shared and mid:
        cands = [" | ".join([name, *mid, BRAND]), " | ".join([name, *mid])]
        return next((c for c in cands if len(c) <= TMAX), cands[-1])
    cands = [" | ".join([name, *mid, BRAND]), f"{name} | {BRAND}", " | ".join([name, *mid]), name]
    return next((c for c in cands if len(c) <= TMAX), name)


def fit_desc(d):
    if len(d) <= DMAX + 2:
        return d
    cut = d[:DMAX + 1]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", cut + " ")]
    if ends and ends[-1] >= 90:
        return d[:ends[-1]].strip()
    if cut.rfind(", ") >= 90:                      # a list: end it cleanly instead of mid-phrase
        return cut[:cut.rfind(", ")] + " and more."
    return cut[:cut.rfind(" ")].rstrip(" ,;:—-") + "…"


def get(s, attr, key):
    m = re.search(rf'<meta {attr}="{re.escape(key)}" content="([^"]*)"', s)
    return m.group(1) if m else None


def put(s, attr, key, val):
    return re.sub(rf'(<meta {attr}="{re.escape(key)}" content=")[^"]*(")', lambda m: m.group(1) + val + m.group(2), s, count=1)


def jsonld(s, old, new):
    """Replace a WebPage name/description string inside the JSON-LD."""
    o, n = json.dumps(html.unescape(old), ensure_ascii=False)[1:-1], json.dumps(html.unescape(new), ensure_ascii=False)[1:-1]
    return s.replace(f'"name":"{o}"', f'"name":"{n}"').replace(f'"description":"{o}"', f'"description":"{n}"')


files = [f for f in sorted(ROOT.rglob("*.html"))
         if not f.name.startswith("google") and f.relative_to(ROOT).parts[0] not in ("tools", "node_modules")]
first = {}
for f in files:
    m = re.search(r"<title>(.*?)</title>", f.read_text(), re.S)
    if m:
        k = m.group(1).split(" | ")[0].strip()
        first[k] = first.get(k, 0) + 1

changed = 0
for f in files:
    rel = f.relative_to(ROOT).as_posix()
    s = orig = f.read_text()
    fx = FIXED.get(rel, {})
    m = re.search(r"<title>(.*?)</title>", s, re.S)
    if m:
        old = m.group(1).strip()
        new = fx.get("title") or ("" if rel == "index.html" else fit_title(old, first.get(old.split(" | ")[0].strip(), 0) > 1)) or old
        if new != old:
            s = s.replace(m.group(0), f"<title>{new}</title>", 1)
            s = jsonld(s, old, new)
            for attr, key in (("property", "og:title"), ("name", "twitter:title")):
                if get(s, attr, key) is not None:
                    s = put(s, attr, key, new)
            s = s.replace(f'content="Preview card: {html.unescape(old)}"', f'content="Preview card: {html.unescape(new)}"')
    d = get(s, "name", "description")
    if d:
        new = fx.get("desc") or fit_desc(html.unescape(d))
        new = html.escape(new, quote=True)
        if new != d:
            s = put(s, "name", "description", new)
            s = jsonld(s, d, new)
            for attr, key in (("property", "og:description"), ("name", "twitter:description")):
                if get(s, attr, key) == d:
                    s = put(s, attr, key, new)
    if s != orig:
        f.write_text(s)
        changed += 1
print(f"search metadata tuned on {changed} pages")
