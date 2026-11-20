#!/usr/bin/env python3
"""Interactive decision-tree hubs (same page style as the Bayesian and Markov hubs) for the Microeconometrics, Finance, Productivity
and Nonparametric guides. The user answers a few questions, the tree lights up and the page recommends a guide.

    python3 tools/bm_selector/build_ask_hubs.py [micro finance productivity nonparam]

Add a guide to the tree below (a leaf is the guide slug). Then run tools/site_polish.py, share_buttons.py, og_tags.py, make_sitemap.py.
"""
import json, pathlib, re, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "tools/bayes_markov"))
from build_guides import SECTIONS, meta_of  # noqa: E402
import importlib.util
_s = importlib.util.spec_from_file_location("bbs", HERE / "build_bm_selector.py"); _b = importlib.util.module_from_spec(_s); _s.loader.exec_module(_b)
CSS_ADD = _b.CSS_ADD


def Q(id, label, text, *opts):
    """opts: (answer, small text, edge label, child) where child is a Q or a guide slug."""
    return dict(id=id, label=label, text=text, opts=opts)


TREES = {
 "micro": Q("q1", "Outcome type", "What does your outcome look like?",
   ("Yes or no", "passed, defaulted, employed, adopted", "yes/no",
    Q("qb", "Complication?", "Is there a complication with your yes/no outcome?",
      ("No: one cross-section", "a plain binary regression", "plain", "logit-probit"),
      ("The same units are observed in several periods", "a panel with unit effects", "panel", "panel-logit-probit"),
      ("A regressor is endogenous", "you have an instrument", "endogenous", "iv-probit"))),
   ("Ordered categories", "a rating from 1 to 5, low / medium / high", "ordered", "ordered-logit-probit"),
   ("A choice among options with no order", "party, transport mode, occupation", "choice",
    Q("qc", "What varies?", "What explains the choice?",
      ("Characteristics of the person only", "income, age, education", "person", "multinomial-logit"),
      ("Attributes of the alternatives, or tastes that differ", "the price and time of each mode", "alternatives", "conditional-mixed-logit"))),
   ("A count", "visits, patents, accidents, arrests", "count",
    Q("qn", "Many zeros?", "Does the count have more zeros than a Poisson model predicts?",
      ("No, the usual count model is enough", "Poisson or negative binomial", "no", "count-models"),
      ("Yes, many more zeros", "two processes: whether and how many", "yes", "zero-inflated-hurdle"))),
   ("An amount that piles up at a limit, or is seen only for some people", "hours, spending, wages of workers", "limited",
    Q("ql", "Why limited?", "Why is the outcome limited?",
      ("It piles up at zero or another limit", "hours worked, spending", "piles up", "tobit"),
      ("It is observed only for a self-selected part of the sample", "wages of workers", "selected", "heckman"))),
   ("A proportion or a positive skewed amount", "a share between 0 and 1, costs", "share", "glm-fractional"),
   ("Several related outcomes", "labor supply and wages, expenditure shares", "system", "systems-sur-3sls")),
 "finance": Q("q1", "Goal", "What do you want to learn from the financial data?",
   ("Do risk factors explain average returns?", "alpha, beta, the GRS test", "pricing", "capm"),
   ("How much can a portfolio lose?", "value at risk, expected shortfall", "losses", "var-es"),
   ("How do correlations between assets move?", "contagion, hedging, diversification", "correlation", "dcc"),
   ("How should I split money across assets?", "weights, efficient frontier, out of sample", "allocation", "portfolio"),
   ("Which volatility forecast is better?", "GARCH against EWMA against rolling windows", "volatility", "volatility-forecasts"),
   ("Is a price in an explosive bubble?", "episodes of explosive growth", "bubble", "bubbles"),
   ("What does the term structure of rates look like?", "level, slope and curvature", "curve", "yield-curve")),
 "productivity": Q("q1", "Question", "What do you want to know about productivity?",
   ("How far is each unit from the best practice?", "a firm, farm, plant or state", "efficiency",
    Q("qe", "Assume a form?", "Are you willing to assume a production function, with noise in the data?",
      ("Yes: a frontier with noise and inefficiency", "one output; the estimate has standard errors", "yes", "stochastic-frontier"),
      ("No: several inputs and outputs, no functional form", "linear programming; no noise", "no", "dea"))),
   ("How did productivity change over time?", "a panel of units and periods", "over time", "malmquist"),
   ("Why do countries or regions grow at different rates?", "Solow, convergence", "growth", "growth-regressions"),
   ("What is the technology: returns to scale, substitution, costs?", "output elasticities, cost structure", "technology", "production-cost-function"),
   ("Do neighbouring places affect each other?", "regions, countries, neighbourhoods", "space", "spatial-regression")),
 "nonparam": Q("q1", "Problem", "What is the problem?",
   ("I want the shape of a relation without assuming one", "the regression function", "shape",
    Q("qs", "How much form?", "How much do you want to assume about the other regressors?",
      ("One regressor, free shape", "the plain curve with a bandwidth", "one x", "kernel-regression"),
      ("Linear effects for some regressors, a free shape for one control", "Robinson's estimator", "one free", "partially-linear"),
      ("A smooth curve with a chosen smoothness and a test against a line", "penalised B-splines", "smooth", "splines"))),
   ("A few observations may be driving my regression", "outliers, high leverage", "outliers", "robust-regression"),
   ("The errors are correlated within groups", "firms, schools, villages; few groups", "clusters", "wild-cluster-bootstrap"),
   ("I want the shape of a distribution", "skewness, more than one mode", "distribution", "kernel-density"),
   ("I am testing many hypotheses at once", "genes, outcomes, subgroups", "many tests", "multiple-testing"),
   ("I care about the far tail", "large losses, floods, claims", "tail", "extreme-values")),
}

TEXT = {
 "micro": dict(kicker="Empirical Research Guide · Microeconometrics", title="Which microeconometric model fits your outcome?",
   lead="The right microeconometric model depends on what your outcome looks like and on what stands in the way of a plain regression: a yes/no outcome, ordered categories, a choice among options, a count, an amount that is censored or selected. Answer a few questions and follow the decision tree to the guide that fits, then open it for the step-by-step paper.",
   ask="Choose the answer that describes your outcome. Each answer lights up the decision tree and narrows the choice to one guide.",
   note="The tool asks about the form of the outcome first, because that decides the family of models. It then asks the question that separates the models inside the family: a panel, an endogenous regressor, attributes that vary across alternatives, excess zeros, or selection. If two answers fit, the guides say how to compare the models."),
 "finance": dict(kicker="Empirical Research Guide · Finance", title="Which finance method fits your question?",
   lead="Finance questions differ in what they ask of the data: whether risk factors explain returns, how large losses can be, how correlations move, how to allocate, which forecast is better, whether a price is a bubble. Answer one question and follow the tree to the guide that carries the method, with real returns data and the text for the paper.",
   ask="Choose the question that is closest to yours. The decision tree lights up and the page recommends one guide.",
   note="Each question maps to one method. Several of the guides work together: the output of a volatility model feeds value at risk, and the correlations from DCC-GARCH feed the portfolio weights."),
 "productivity": dict(kicker="Empirical Research Guide · Productivity and Spatial Econometrics", title="Which productivity method fits your question?",
   lead="Productivity questions call for different tools depending on whether you measure the distance from the best practice, the change over time, the sources of growth across countries, the technology itself or the spillovers between places. Answer a few questions and follow the tree to the guide.",
   ask="Choose the question that is closest to yours. If you ask about efficiency, a second question separates a frontier with noise from data envelopment analysis.",
   note="The frontier methods answer different questions. A stochastic frontier needs a functional form and treats noise and inefficiency separately; data envelopment analysis needs no form but counts noise as inefficiency. The guides show both on real data."),
 "nonparam": dict(kicker="Empirical Research Guide · Nonparametric and Robust Methods", title="Which nonparametric or robust method fits your problem?",
   lead="Nonparametric and robust methods relax an assumption: a functional form, normal errors, no outliers, independent observations, one test at a time. Answer a few questions to find the guide that relaxes the assumption that worries you, with real data, simulations and the text for the paper.",
   ask="Choose the problem that is closest to yours. If the problem is the shape of a relation, a second question asks how much you want to assume.",
   note="Each guide relaxes one assumption and shows what happens when it does not hold, with a simulation. If more than one assumption worries you, combine them: a partially linear model can be run with robust standard errors, and a kernel density can be used on the residuals of a robust fit."),
}


LEAF = {"logit-probit": "Logit / probit", "ordered-logit-probit": "Ordered logit", "multinomial-logit": "Multinomial", "heckman": "Heckman",
        "zero-inflated-hurdle": "Zero-inflated", "glm-fractional": "GLM", "systems-sur-3sls": "SUR / 3SLS", "panel-logit-probit": "Panel logit",
        "conditional-mixed-logit": "Mixed logit", "production-cost-function": "Production / cost", "volatility-forecasts": "Volatility forecasts"}


def layout(tree, models):
    """Top-down layout: leaves in order, staggered over two rows when the tree is wide; questions centred over their children."""
    nodes, edges, QS, TL = {}, [], {}, {"start": "Your data"}
    leaves = []

    def walk(q, depth):
        QS[q["id"]] = dict(t=q["text"], o=[[a, s, (c["id"] if isinstance(c, dict) else c)] for a, s, e, c in q["opts"]])
        nodes[q["id"]] = dict(depth=depth, label=q["label"], kids=[])
        for a, s, e, c in q["opts"]:
            cid = c["id"] if isinstance(c, dict) else c
            edges.append([q["id"], cid, e]); nodes[q["id"]]["kids"].append(cid)
            if isinstance(c, dict):
                walk(c, depth + 1)
            else:
                nodes[cid] = dict(depth=depth + 1, label=LEAF.get(cid, models[cid]["short"]), kids=[], leaf=True); leaves.append(cid)
    walk(tree, 1)
    nodes["start"] = dict(depth=0, label="Start", kids=[tree["id"]])
    edges.insert(0, ["start", tree["id"], None])
    TL.update({k: v["label"] for k, v in nodes.items() if not v.get("leaf")}); TL["start"] = "Start"
    wid = lambda n: max(70, int(len(nodes[n]["label"]) * 6.8 + 20))
    # a parent with three or more leaf children staggers them over two rows so that they can sit closer together
    for k, v in nodes.items():
        lk = [c for c in v["kids"] if nodes[c].get("leaf")]
        for j, c in enumerate(lk):
            nodes[c]["row"] = (j % 2) if len(lk) >= 3 else 0
    lane = lambda k: nodes[k]["depth"] * 70 + (34 * nodes[k].get("row", 0) if nodes[k].get("leaf") else 0)
    GAP = 12

    def pack(n):
        """Returns (rel_x, contour) for the subtree of n: rel_x of every node relative to n, contour = lane -> (min, max) edges."""
        kids = nodes[n]["kids"]
        w = wid(n)
        if not kids:
            return {n: 0.0}, {lane(n): [-w / 2, w / 2]}
        pos = {}; cont = {}; offs = []
        for c in kids:
            cp, cc = pack(c)
            if not cont:
                d = 0.0
            else:
                sh = [cont[l][1] + GAP - cc[l][0] for l in cc if l in cont]
                d = max(sh) if sh else max(v[1] for v in cont.values()) + GAP - min(v[0] for v in cc.values())
                d = max(d, offs[-1] + 24)                     # keep the order of the options
            offs.append(d)
            for k2, x in cp.items(): pos[k2] = x + d
            for l, (lo, hi) in cc.items():
                cur = cont.get(l)
                cont[l] = [lo + d, hi + d] if not cur else [min(cur[0], lo + d), max(cur[1], hi + d)]
        mid = (pos[kids[0]] + pos[kids[-1]]) / 2
        out = {k2: x - mid for k2, x in pos.items()}; out[n] = 0.0
        cont = {l: [lo - mid, hi - mid] for l, (lo, hi) in cont.items()}
        lw = lane(n)
        cur = cont.get(lw)
        cont[lw] = [-w / 2, w / 2] if not cur else [min(cur[0], -w / 2), max(cur[1], w / 2)]
        return out, cont
    rel, _ = pack("start")
    for k, x in rel.items(): nodes[k]["x"] = x
    minx = min(v["x"] - wid(k) / 2 for k, v in nodes.items()); shift = 14 - minx
    N = {}
    for k, v in nodes.items():
        y = 26 + v["depth"] * 70 + (34 * v.get("row", 0) if v.get("leaf") else 0)
        N[k] = [round(v["x"] + shift), y, wid(k), v["label"], 1] if v.get("leaf") else [round(v["x"] + shift), y, wid(k), v["label"]]
    W = max(n[0] + n[2] / 2 for n in N.values()) + 14; Hh = max(n[1] for n in N.values()) + 34
    return N, edges, QS, TL, f"0 0 {int(W)} {int(Hh)}"


def build(key):
    S, T = SECTIONS[key], TEXT[key]
    folder = ROOT / S["folder"]
    models = {}
    for s in S["order"]:
        m = meta_of(ROOT / "tools/bayes_markov/guides" / key / f"{s}.md")
        md = (ROOT / "tools/bayes_markov/guides" / key / f"{s}.md").read_text()
        steps = [re.sub(r"[`*]", "", h).strip() for h in re.findall(r"^## (.+)$", md, re.M)]
        models[s] = dict(name=m["title"], short=m["short"], when=m["when"], why=m["lead"], steps=steps)
    placed = set()
    def leaves(q):
        for a, s, e, c in q["opts"]:
            placed.update(leaves(c) if isinstance(c, dict) else [c]) if False else None
            if isinstance(c, dict): leaves(c)
            else: placed.add(c)
    leaves(TREES[key])
    assert placed == set(S["order"]), (key, set(S["order"]) ^ placed)
    N, E, QS, TL, vb = layout(TREES[key], models)
    hub = dict(key=key, vb=vb, known=False, GUIDE={s: s for s in S["order"]}, MODELS=models, N=N, E=E, TL=TL, first=TREES[key]["id"], QS=QS)
    ptpl = (ROOT / "tools/panel_selector/template.html").read_text()
    ts = (ROOT / "tools/model_selector/template.html").read_text()
    css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
    head = ptpl[: ptpl.index('<section class="hero">')]
    head = head.replace("/*TS_CSS*/", css).replace("</style>", CSS_ADD + "</style>", 1)
    desc = re.sub(r"<[^>]+>", "", T["lead"]).split(". ")[0] + "."
    head = re.sub(r"<title>.*?</title>", f"<title>{S['name']} | Tuhin G. M. Al Mamun</title>", head, 1)
    head = re.sub(r'(<meta name="description" content=")[^"]*"', lambda m: m.group(1) + desc.replace("\\", "") + '"', head, 1)
    head = re.sub(r'(<link rel="canonical" href=")[^"]*"', lambda m: m.group(1) + f"https://tuhingm.github.io/almamun/{S['folder']}/" + '"', head, 1)
    hero = (f'<section class="hero"><div class="wrap"><div class="kicker">{T["kicker"]}</div><h1>{T["title"]}</h1><p class="lead">{T["lead"]}</p>'
            f'<div class="meta"><span><strong>{len(S["order"])}</strong> step-by-step guides</span><span>No data needed to choose</span><span>Real data, code and output in every guide</span></div></div></section>\n')
    body = (HERE / "body_ask.html").read_text().replace("#EXSUB#", "").replace("#ASKSUB#", T["ask"]).replace("#GH#", f"{len(S['order'])} routes to a finished paper").replace("#NOTE#", T["note"])
    foot = (f'<footer><div class="wrap foot"><span><b>Tuhin G. M. Al Mamun, Ph.D.</b> · {S["name"]}</span>'
            f'<span><a href="../time-series-guide/index.html">Time-Series Guide</a> · © <span id="year"></span></span></div></footer>\n')
    js = "const HUB=" + json.dumps(hub, separators=(",", ":")) + ";\n" + (HERE / "common.js").read_text().replace("const HUB=", "const _unused=") + "\nsetMode('ask');\n"
    page = head + hero + body + foot + f'<script>\nconst EXAMPLES=[];\n{js}</script>\n</body>\n</html>\n'
    (folder / "index.html").write_text(page)
    print(key, "decision-tree hub written", len(page) // 1024, "KB", vb)


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(TREES)):
        build(k)
