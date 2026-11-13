#!/usr/bin/env python3
"""Site-wide finishing pass. Run after the page builders and before tools/make_og_images.py and tools/og_tags.py.

On every public .html page it:
  1. writes the same top menu (brand + eight links, current section highlighted) and loads assets/css/site-header.css;
  2. adds link-preview (Open Graph) tags to pages that have none, and lists their preview cards in
     tools/og_cards_site.json for make_og_images.py;
  3. adds search-engine metadata: author, robots, og:locale, unique descriptions, and structured data (JSON-LD) that
     ties every page to the Person "Tuhin G. M. Al Mamun" with all name variants, plus a breadcrumb trail.
Safe to run repeatedly.
"""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = "https://tuhingm.github.io/almamun/"
NAME = "Tuhin G. M. Al Mamun"
ALT_NAMES = ["Tuhin Al Mamun", "Tuhin", "Tuhin G M Al Mamun", "Tuhin G.M. Al Mamun", "Tuhin GM Al Mamun",
             "T. G. M. Al Mamun", "TGM Al Mamun", "Al Mamun Tuhin", "Tuhin Al Mamun, Ph.D.", "Dr. Tuhin Al Mamun"]
SAME_AS = ["https://orcid.org/0009-0005-0275-6922",
           "https://scholar.google.com/citations?user=CMV8R_0AAAAJ&hl=en",
           "https://www.scopus.com/authid/detail.uri?authorId=59544050900",
           "https://www.linkedin.com/in/tuhinecon/",
           "https://sites.google.com/view/tuhin2022/profile",
           "https://www.researchgate.net/profile/Tuhin-G-M-Al-Mamun",
           "https://github.com/tuhingm"]

# (label, target, section folder or None)
NAV = [("Home", "index.html", ""), ("Teaching Courses", "teaching/index.html", "teaching"),
       ("Working Papers", "working-papers/index.html", "working-papers"),
       ("Time-Series Guide", "research-guide/index.html", "research-guide"),
       ("Panel-Data Guide", "panel-data-guide/index.html", "panel-data-guide"),
       ("Causal Inference", "causal-inference-guide/index.html", "causal-inference-guide"),
       ("DSGE Guide", "dsge-guide/index.html", "dsge-guide"), ("ML Guide", "ml-guide/index.html", "ml-guide"),
       ("Bayesian Guide", "bayesian-guide/index.html", "bayesian-guide"), ("Markov Guide", "markov-guide/index.html", "markov-guide"),
       ("Open Econometric Systems", "oes/index.html", "oes"), ("CV", "Tuhin%20Al%20Mamun_CV.pdf", None)]
SECTION = {"teaching": "Teaching Courses", "working-papers": "Working Papers",
           "research-guide": "Empirical Research Guide (Time Series)", "panel-data-guide": "Panel Data Research Guide",
           "causal-inference-guide": "Causal Inference Research Guide", "dsge-guide": "DSGE Research Guide",
           "ml-guide": "Machine Learning Research Guide", "oes": "Open Econometric Systems",
           "bayesian-guide": "Bayesian Research Guide", "markov-guide": "Markov-Switching Research Guide"}
KICKER = {"panel-data-guide": "Panel Data Research Guide", "causal-inference-guide": "Causal Inference Research Guide",
          "dsge-guide": "DSGE Research Guide", "research-guide": "Empirical Research Guide",
          "ml-guide": "Machine Learning Research Guide", "bayesian-guide": "Bayesian Research Guide",
          "markov-guide": "Markov-Switching Research Guide"}
CARD_KICKER = {"panel-data-guide": "Panel Data Guide", "causal-inference-guide": "Causal Inference Guide",
               "dsge-guide": "DSGE Guide · Dynare", "ml-guide": "Machine Learning Guide",
               "bayesian-guide": "Bayesian Guide", "markov-guide": "Markov-Switching Guide"}
INDEX_CARDS = {
    "panel-data-guide/index.html": ("Which panel model fits your data?",
        "Test your panel and follow the path to pooled OLS, FE/RE, GMM, Mean Group, PMG or CS-ARDL. Real examples and Python code."),
    "causal-inference-guide/index.html": ("Which causal design fits your question?",
        "DiD, staggered DiD, synthetic control, RDD, IV and interrupted time series, tested on your data, step by step."),
    "ml-guide/index.html": ("Which machine-learning method fits your data?",
        "LASSO, ridge, elastic net, random forest, XGBoost or double machine learning, chosen by tests on your data."),
    "bayesian-guide/index.html": ("Which Bayesian VAR fits your question?",
        "Bayesian VAR, FAVAR, steady-state, panel, time-varying, stochastic-volatility, mixed-frequency and threshold models, step by step."),
    "markov-guide/index.html": ("Which Markov-switching model fits your data?",
        "Hamilton's regime model, regression, VAR and panel VAR with hidden regimes: tests, regime dates and the text for the paper."),
    "dsge-guide/index.html": ("Which DSGE model fits your question?",
        "RBC, New Keynesian, open economy, Bayesian estimation, OccBin and optimal policy, step by step in Dynare."),
}
COURSES = {}
for m in re.finditer(r"^  ([a-z-]+):\n(?:    .*\n)*?    title: (.+)$", (ROOT / "tools/courses.yml").read_text(), re.M):
    COURSES[m.group(1)] = m.group(2).strip()

HEADER_RE = re.compile(r'<header(?: class="site-header")?>\s*<div class="wrap nav">.*?</header>', re.S)
LD_RE = re.compile(r'\s*<script type="application/ld\+json" id="site-ld">.*?</script>', re.S)
ANY_LD_RE = re.compile(r'\s*<script type="application/ld\+json"[^>]*>.*?</script>', re.S)


def meta(s, key):
    m = re.search(rf'<meta (?:property|name)="{re.escape(key)}" content="([^"]*)"', s)
    return m.group(1) if m else None


def set_meta(s, key, value, attr="property"):
    """Replace a meta tag's content, or insert the tag after the canonical link."""
    pat = re.compile(rf'<meta (?:property|name)="{re.escape(key)}" content="[^"]*">')
    tag = f'<meta {attr}="{key}" content="{value}">'
    if pat.search(s):
        return pat.sub(lambda _: tag, s, count=1)
    return re.sub(r'(<link rel="canonical" href="[^"]*">)', lambda m: m.group(1) + "\n" + tag, s, count=1)


def header(root, section):
    links = []
    for label, target, sec in NAV:
        cur = sec is not None and sec == section
        attrs = ' class="current" aria-current="page"' if cur else ""
        links.append(f'<a href="{root}{target}"{attrs}>{label}</a>')
    return ('<header class="site-header"><div class="wrap nav"><a class="brand" href="' + root + 'index.html" '
            'aria-label="Tuhin G. M. Al Mamun — home">TGM <span>AL MAMUN</span></a>'
            '<button id="menu" type="button" aria-expanded="false" aria-controls="nav">Menu</button>'
            '<nav id="nav" aria-label="Main navigation">' + "".join(links) + '</nav></div></header>')


def short(text, n):
    text = html.unescape(text).strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0].rstrip(",;:·—-")
    return cut + "…"


def _sub(name, desc):
    """Preview-card subtitle: the guide's use case first, then what the page offers."""
    d = html.unescape(desc)
    m = re.search(r"with (\w+) code, real output and sample text\. (.+)$", d)
    if m:
        return f"{m.group(2).rstrip('.')}. Step-by-step guide with {m.group(1)} code and real output."
    if d.lower().startswith(name.lower() + ":"):
        d = d[len(name) + 1:].strip()
        d = d[:1].upper() + d[1:]
    return d


def page_name(title):
    return html.unescape(title.split(" | ")[0]).strip()


pages = sorted(p for p in ROOT.rglob("*.html") if not p.name.startswith("google")
               and not any(part.startswith((".", "tools", "node_modules")) for part in p.relative_to(ROOT).parts))
texts = {p: p.read_text() for p in pages}

# Descriptions shared by several pages (course sub-pages reuse the course summary) get the page name in front.
seen = {}
for p, s in texts.items():
    for key in ("description", "og:description"):
        d = meta(s, key)
        if d:
            seen.setdefault((key, d), []).append(p)

cards, changed = [], 0
for p in pages:
    s = orig = texts[p]
    rel = p.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    root = "../" * (len(parts) - 1)
    section = parts[0] if len(parts) > 1 else ""
    url = SITE + (rel[: -len("index.html")] if rel.endswith("index.html") else rel)

    # 1. Menu ---------------------------------------------------------------------------------------------------
    s, n = HEADER_RE.subn(lambda _: header(root, section), s, count=1)
    if not n:
        print("no header:", rel)
    css = f'<link rel="stylesheet" href="{root}assets/css/site-header.css">'
    if "site-header.css" not in s:
        s = s.replace("</head>", css + "\n</head>", 1)
    if not any(k in s for k in ("course.js", "mobile-nav.js", "getElementById('menu')")):
        s = s.replace("</body>", f'<script src="{root}assets/js/mobile-nav.js" defer></script>\n</body>', 1)

    # Titles such as "Econometrics | Econometrics | …" on course home pages.
    if section == "teaching":
        s = re.sub(r"<title>([^<|]+?) \| \1 \|", lambda m: f"<title>{m.group(1)} — Course Home |", s)
    ogt = meta(s, "og:title")
    if ogt:
        ogt2 = re.sub(r"^([^|]+?) \| \1$", r"\1 — Course Home" if section == "teaching" else r"\1", ogt)
        if "Al Mamun" not in ogt2:
            ogt2 += " | Tuhin Al Mamun"
        s = set_meta(s, "og:title", ogt2)

    title = re.search(r"<title>(.*?)</title>", s, re.S).group(1).strip()
    name = page_name(title)

    for key in ("description", "og:description"):
        d = meta(s, key)
        if d and len(seen.get((key, d), [])) > 1 and not html.unescape(d).startswith(name):
            s = set_meta(s, key, html.escape(f"{name} — ", quote=True) + d, "name" if key == "description" else "property")

    # 2. Link previews for pages that have none -----------------------------------------------------------------
    if not meta(s, "og:title"):
        desc = meta(s, "description") or ""
        img = "assets/og/" + rel[:-5] + ".jpg"
        og = (f'<meta property="og:type" content="{"website" if rel.endswith("index.html") else "article"}">'
              f'<meta property="og:site_name" content="Tuhin Al Mamun, Ph.D.">\n'
              f'<meta property="og:title" content="{html.escape(name, quote=True)} | {html.escape(KICKER.get(section, "Research Guide").split(" · ")[0], quote=True)} | Tuhin Al Mamun">'
              f'<meta property="og:description" content="{desc}">\n'
              f'<meta property="og:url" content="{url}"><meta property="og:image" content="{SITE}{img}">\n'
              f'<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">\n'
              f'<meta name="twitter:card" content="summary_large_image">')
        s = re.sub(r'(<link rel="canonical" href="[^"]*">)', lambda m: m.group(1) + "\n" + og, s, count=1)
    if rel.startswith(("panel-data-guide/", "causal-inference-guide/", "dsge-guide/", "ml-guide/", "bayesian-guide/", "markov-guide/")):
        kicker = CARD_KICKER[section] + (" · Lecture" if "/lectures/" in rel else (" · Model guide" if not rel.endswith("index.html") else ""))
        t_, sub_ = INDEX_CARDS.get(rel, (name, short(_sub(name, meta(s, "description") or ""), 130)))
        cards.append(dict(out="assets/og/" + rel[:-5] + ".jpg", kicker=kicker, title=t_, sub=sub_))

    # 3. Search-engine metadata ---------------------------------------------------------------------------------
    s = set_meta(s, "og:url", url)
    if not meta(s, "author"):
        s = set_meta(s, "author", NAME, "name")
    s = set_meta(s, "robots", "index, follow, max-image-preview:large, max-snippet:-1", "name")
    s = set_meta(s, "og:locale", "en_US")
    if rel == "index.html":
        s = set_meta(s, "og:type", "profile")
        s = set_meta(s, "profile:first_name", "Tuhin")
        s = set_meta(s, "profile:last_name", "Al Mamun")

    person = {"@type": "Person", "@id": SITE + "#person", "name": NAME, "alternateName": ALT_NAMES,
              "givenName": "Tuhin", "familyName": "Al Mamun", "honorificSuffix": "Ph.D.", "url": SITE,
              "image": SITE + "assets/og-image.jpg", "jobTitle": "Economist and Quantitative Researcher",
              "description": "Economist and quantitative researcher working on monetary and fiscal policy, "
                             "DSGE and HANK models, environmental economics and econometrics.",
              "alumniOf": {"@type": "CollegeOrUniversity", "name": "Hannam University",
                           "address": "Daejeon, South Korea"},
              "hasCredential": {"@type": "EducationalOccupationalCredential", "credentialCategory": "degree",
                                "name": "Ph.D. in Economics", "recognizedBy": {"@type": "CollegeOrUniversity", "name": "Hannam University"}},
              "nationality": {"@type": "Country", "name": "Bangladesh"},
              "knowsAbout": ["Monetary policy", "Fiscal policy", "Macroeconomics", "DSGE models",
                             "Heterogeneous Agent New Keynesian (HANK) models", "Bayesian estimation",
                             "Econometrics", "Time-series econometrics", "Panel data econometrics",
                             "Causal inference", "Environmental economics", "Climate economics", "Statistics"],
              "sameAs": SAME_AS}
    website = {"@type": "WebSite", "@id": SITE + "#website", "url": SITE, "name": "Tuhin G. M. Al Mamun, Ph.D.",
               "alternateName": ["Tuhin Al Mamun", "TGM Al Mamun"], "publisher": {"@id": SITE + "#person"},
               "inLanguage": "en"}
    crumbs = [("Home", SITE)]
    if section:
        crumbs.append((SECTION.get(section, section), SITE + section + "/"))
    if section == "teaching" and len(parts) > 2:
        crumbs.append((COURSES.get(parts[1], parts[1]), SITE + f"teaching/{parts[1]}/"))
    if not rel.endswith("index.html") or len(parts) > 3:
        crumbs.append((name, url))
    elif crumbs[-1][1] != url:
        crumbs.append((name, url))
    page = {"@type": "ProfilePage" if rel == "index.html" else "WebPage", "@id": url + "#webpage", "url": url,
            "name": html.unescape(title), "description": html.unescape(meta(s, "description") or ""),
            "isPartOf": {"@id": SITE + "#website"}, "inLanguage": "en",
            "author": {"@id": SITE + "#person"}, "copyrightHolder": {"@id": SITE + "#person"},
            "primaryImageOfPage": {"@type": "ImageObject", "url": (meta(s, "og:image") or "").split("?")[0]}}
    if rel == "index.html":
        page["mainEntity"] = {"@id": SITE + "#person"}
        page["about"] = {"@id": SITE + "#person"}
    graph = [page, person, website]
    if len(crumbs) > 1:
        page["breadcrumb"] = {"@id": url + "#breadcrumb"}
        graph.append({"@type": "BreadcrumbList", "@id": url + "#breadcrumb",
                      "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n_, "item": u}
                                          for i, (n_, u) in enumerate(crumbs)]})
    ld = json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False, separators=(",", ":"))
    s = (ANY_LD_RE if rel == "index.html" else LD_RE).sub("", s)
    s = s.replace("</head>", f'<script type="application/ld+json" id="site-ld">{ld}</script>\n</head>', 1)

    if s != orig:
        p.write_text(s)
        changed += 1

(ROOT / "tools" / "og_cards_site.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1) + "\n")
print(f"site polish: {changed} of {len(pages)} pages updated; {len(cards)} guide preview cards listed")
