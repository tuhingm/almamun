#!/usr/bin/env python3
"""Register extra model guides (built from tools/bayes_markov/guides/lp/<slug>.md) in the Time-Series Guide tool page.

Patches tools/model_selector/template.html and time-series-guide/index.html in the same way (decision-tree node under
"Another goal?", the model card, the guided question) and adds the new guide to the "All method guides" row of every
existing time-series guide page. Idempotent. usage: python3 tools/model_selector/add_model.py arima [garch ...]
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TS = ROOT / "time-series-guide"

# key = slug.capitalised as shown on the cards; slot = x position of the leaf in the "Another goal?" row
EXTRA = {
    "arima": dict(key="ARIMA", short="ARIMA / SARIMA", node="ARIMA", w=90, x=70,
                  name="ARIMA and seasonal ARIMA (SARIMA)", when="One series, to describe its dynamics or to forecast it",
                  why="You model a single variable on its own past, to forecast it or to measure how persistent it is. An ARIMA model chooses the differencing order from unit-root tests, the lags from information criteria, and is judged by its residual tests and by out-of-sample forecasts.",
                  steps=["Choose the differencing order d", "ACF and PACF, then p and q by information criteria", "Estimate and check the roots", "Ljung–Box, ARCH-LM and seasonality tests on the residuals", "Seasonal terms, only if they help", "Out-of-sample forecasts and the Diebold–Mariano test"],
                  lec=[["09-time-series.html", "Lecture: Time-series foundations"]],
                  q=("Forecast one series, or describe how persistent it is", "ARIMA and seasonal ARIMA")),
    "garch": dict(key="GARCH", short="GARCH family", node="GARCH", w=90, x=200,
                  name="GARCH family (GARCH, EGARCH, GJR-GARCH, DCC)", when="Volatility that changes over time, or correlations that move",
                  why="Returns, exchange rates and inflation show calm and turbulent periods. A GARCH model lets the variance depend on past shocks, asymmetric versions let bad news raise volatility more than good news, and DCC lets the correlation between two series move over time.",
                  steps=["ARCH-LM test on the mean residuals", "Fit GARCH, EGARCH and GJR-GARCH", "Choose the error distribution", "Check the standardised residuals", "News-impact curve and persistence", "Conditional correlation (DCC)"],
                  lec=[["../../finance/lectures/15-financial-econometrics.html", "Lecture: Financial econometrics (finance course)"]], q=("Volatility that changes over time, or time-varying correlation", "GARCH, EGARCH, GJR-GARCH and DCC")),
    "quantile": dict(key="Quantile", short="Quantile regression", node="Quantile", w=104, x=330,
                  name="Quantile regression", when="Effects that differ across the distribution of the outcome",
                  why="The mean hides the tails. Quantile regression estimates the effect of a regressor at the 5th, 50th or 95th percentile of the outcome, which answers questions about risk, such as how much growth could fall in a bad year.",
                  steps=["Choose the quantiles", "Estimate the quantile coefficients", "Inference: bootstrap and heteroskedasticity-robust", "Test whether the slopes differ across quantiles", "Check that the fitted quantiles do not cross", "Quantile forecasts and their calibration"],
                  lec=[], q=("Effects in the tails of the distribution (risk, growth at risk)", "Quantile regression")),
    "cointegration": dict(key="Cointegration", short="Cointegration tests", node="Coint. tests", w=110, x=460,
                  name="Cointegration tests (Engle–Granger, Phillips–Ouliaris, Gregory–Hansen)", when="Is there a long-run relationship, possibly with a break?",
                  why="Before you estimate a VECM or an ARDL, you must know whether a long-run relationship exists. Engle–Granger and Phillips–Ouliaris test the residuals of the long-run regression, and Gregory–Hansen allows the relationship to shift at an unknown date.",
                  steps=["Confirm that every series is I(1)", "Engle–Granger test", "Phillips–Ouliaris test", "Gregory–Hansen test with a break", "Date of the break", "Which model to use next"],
                  lec=[["12-cointegration-ardl.html", "Lecture: Cointegration and ARDL"]], q=("Test for a long-run relationship, with or without a break", "Cointegration tests")),
    "wavelet": dict(key="Wavelet", short="Wavelet coherence", node="Wavelet", w=96, x=590,
                  name="Wavelet coherence", when="Co-movement that differs across time and frequency",
                  why="Two series can move together at business-cycle frequencies and not at short ones, and the link can change over time. Wavelet coherence maps the strength of the co-movement by period and by date.",
                  steps=["Continuous wavelet transform", "Wavelet power of each series", "Coherence and phase", "Significance by Monte Carlo", "Averages by frequency band", "Read the lead and lag"],
                  lec=[], q=("Co-movement across time and frequency", "Wavelet coherence")),
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
    s = s.replace("\n};\nconst $=s=>document", f",\n {k}:{{name:{js(e['name'])},when:{js(e['when'])},why:{js(e['why'])},steps:{steps},lec:{lec}}}\n}};\nconst $=s=>document", 1)
    first = "goal:[" not in s
    node = (f"goal:[330,548,176,'Another goal?']," if first else "") + f"{k}:[{e['x']},624,{e['w']},{js(e['node'])},1]"
    s = s.replace("};\nconst E=[", f",{node}}};\nconst E=[", 1)
    s = s.replace("];\nconst PATHS=", f",['goal','{k}']];\nconst PATHS=", 1)
    s = s.replace("};\nconst TL=", f",{k}:['goal','{k}']}};\nconst TL=", 1)
    s = s.replace("};\nfunction drawTree", (",goal:'Another goal'" if first else "") + f",{k}:{js(e['node'])}}};\nfunction drawTree", 1)
    s = s.replace('viewBox="0 0 724 510"', 'viewBox="0 0 724 660"')
    if first:
        s = s.replace("['Yes','Panel data, or instruments needed','GMM']]}", "['No. I want to forecast one series, or study its volatility, quantiles or cycles','ARIMA, GARCH, quantile regression, cointegration tests, wavelets','q6',['start','q1']],['Yes','Panel data, or instruments needed','GMM']]}", 1)
        s = s.replace("\n q4:{t:", "\n q6:{t:'What is the goal?',o:[/*Q6*/]},\n q4:{t:", 1)
    s = s.replace("/*Q6*/", f"[{js(e['q'][0])},{js(e['q'][1])},{js(k)}],/*Q6*/", 1)
    path.write_text(s)
    return True


def patch_guides_list(path, slug):
    s = path.read_text()
    m = re.search(r'const GUIDES=\[(.*?)\];', s)
    if not m or f'"{slug}"' in m.group(1):
        return
    s = s.replace(m.group(0), f'const GUIDES=[{m.group(1)}, "{slug}"];', 1)
    path.write_text(s)


def patch_others(slug):
    e = EXTRA[slug]; link = f'<a class="" href="{slug}.html">{e["short"]}</a>'
    for p in sorted(TS.glob("*.html")):
        if p.name in ("index.html", f"{slug}.html"):
            continue
        s = p.read_text()
        i = s.find('<div class="others">')
        if i < 0 or f'href="{slug}.html"' in s[i:i + 3000]:
            continue
        j = s.find("</div></div>", i)
        p.write_text(s[:j] + link + s[j:])


def add_card(slug):
    """Append the link-preview card of the guide to tools/model_selector/guide_cards.json (idempotent)."""
    import json
    f = HERE / "guide_cards.json"
    cards = json.loads(f.read_text())
    out = f"assets/og/time-series-guide-{slug}.jpg"
    if any(c["out"] == out for c in cards):
        return
    head = (ROOT / "tools" / "bayes_markov" / "guides" / "lp" / f"{slug}.md").read_text().split("\n---\n", 1)[0]
    meta = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}
    cards.append(dict(out=out, kicker="Empirical Research Guide · Model guide", title=meta["title"], sub=meta["when"] + ". " + meta["data"],
                      tags=["Python code", "Real output", "How to write it"]))
    f.write_text(json.dumps(cards, ensure_ascii=False, indent=0))


def main():
    for slug in sys.argv[1:]:
        add_card(slug)
        for f in (HERE / "template.html", TS / "index.html"):
            patch_tool(f, slug)
        patch_guides_list(TS / "index.html", slug)
        patch_others(slug)
        print("registered", slug)


if __name__ == "__main__":
    main()
