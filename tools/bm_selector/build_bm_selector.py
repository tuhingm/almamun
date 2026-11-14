"""Builds bayesian-guide/index.html and markov-guide/index.html as interactive model selectors (same pattern as the
Panel Data Guide): precomputed example results, Python tests in the browser, a decision tree and a questionnaire.
Run:  /home/claude/pyo-env/bin/python tools/bm_selector/build_bm_selector.py   then the site pipeline."""
import json, pathlib, re, shutil, sys, warnings
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import bm_analysis as B

CASES = {
 "bayesian": dict(
  folder="bayesian-guide", kicker="Empirical Research Guide · Bayesian Econometrics",
  title="Which Bayesian model fits your data?",
  lead="The right Bayesian model depends on the structure of your data and on what the tests find, not on preference. Test a real dataset, upload your own, or enter your test results, and follow the decision tree to the model your paper needs.",
  desc="An interactive model-selection tool for Bayesian econometrics: test your data and find out whether a Bayesian VAR, FAVAR, steady-state BVAR, panel VAR, TVP-VAR, stochastic-volatility VAR, mixed-frequency BVAR, threshold VAR or Bayesian regression fits.",
  exsub="Five real datasets. Each one leads to a different model, and the decision tree shows why.",
  gh="Routes to a finished paper",
  note="Panels go to the panel VAR; a regression equation to Bayesian regression; a series observed at mixed frequencies to the mixed-frequency BVAR; 15 or more indicators to the FAVAR. For a set of macro series the tool checks whether the series are stationary and, if they are not I(1) in levels, tests three kinds of nonlinearity on a VAR(1): changing volatility (variance ratio, Levene and ARCH tests), unstable coefficients (CUSUM and sup-F) and a threshold (a bootstrap sup-LR test). The strongest rejection (p &lt; 0.01 for volatility, p &lt; 0.05 for the others) decides between the stochastic-volatility, time-varying-parameter and threshold VAR; if none rejects, a constant-parameter Bayesian VAR is adequate. A two-regime Markov-switching fit with LR above about 14 sends you to the Markov-Switching Guide. The tests are screening devices on a VAR(1) and do not replace the diagnostics inside each guide.",
  jsfile="hub_bayesian.js", slug="bayesian-guide", name="Bayesian Guide", nav="Bayesian Guide",
  examples=[
   ("us3","bayesian-guide/examples/us_macro.csv",dict(vars=["gdp_growth","inflation","tbilrate"]),"U.S. macro","Growth, inflation and the T-bill rate","Quarterly U.S. series. The tests ask whether volatility or coefficients change."),
   ("money","bayesian-guide/examples/money.csv",dict(),"U.S. money demand","Money, income and interest rates in levels","Trending series: what the selector does when the variables are I(1)."),
   ("mixed","bayesian-guide/examples/mixed_frequency.csv",dict(),"Mixed frequency","Monthly indicators and quarterly GDP","One variable is observed far less often than the others."),
   ("garrett","bayesian-guide/examples/garrett.csv",dict(vars=["gdp_growth","unem"]),"Panel of countries","14 countries, 25 years","A panel: one VAR for all the countries."),
   ("wage","bayesian-guide/examples/wage2.csv",dict(aim="regression",y="lwage"),"Cross-section","Log wages and 14 regressors","A single regression equation: is the prior needed?")]),
 "markov": dict(
  folder="markov-guide", kicker="Empirical Research Guide · Markov-Switching Models",
  title="Which Markov-switching model fits your data?",
  lead="The right Markov-switching model depends on whether your series are stationary, whether the regime is observed or hidden, and how your data are structured. Test a real dataset, upload your own, or enter your test results, and follow the decision tree to the model your paper needs.",
  desc="An interactive model-selection tool for Markov-switching models: test your data and find out whether a Hamilton MS-AR, Markov-switching regression, MS-VAR or panel MS-VAR fits, with the linearity test and the checks for a collapsed regime.",
  exsub="Five real datasets. Each one leads to a different outcome, including the case where Markov switching is the wrong tool.",
  gh="Routes to a finished paper",
  note="Markov-switching models need stationary series, so trending levels are sent back for transformation. If an observed variable above or below a threshold defines the regime, the threshold VAR (Bayesian Guide) is the better tool. Otherwise the structure decides: one series (MS-AR), a regression (MS regression), several series (MS-VAR) or a panel (panel MS-VAR). The page also reports the evidence for regimes: the likelihood-ratio statistic of a two-regime model against one regime, compared with the Garcia critical value of about 14, the AIC and BIC of both models, and the share of periods in the smaller regime, which flags a collapsed regime. These are screening numbers from a simple two-regime fit with no autoregressive terms and a switching variance, so they can differ from the full test in each guide (the Hamilton GNP guide, with AR(4) and a switching mean only, does not reject linearity at the Garcia value). The guide for each model covers its own test.",
  jsfile="hub_markov.js", slug="markov-guide", name="Markov Guide", nav="Markov Guide",
  examples=[
   ("gnp","markov-guide/examples/hamilton_gnp.csv",dict(vars=["gnp_growth"]),"One series","U.S. GNP growth","Hamilton's data: does one series switch between expansion and recession?"),
   ("phil","markov-guide/examples/us_macro.csv",dict(aim="regression",y="inflation",vars=["inflation","unemp","m1_growth"]),"Regression","Inflation on unemployment and money growth","A regression whose level, slopes or variance might switch."),
   ("us3","markov-guide/examples/us_macro.csv",dict(vars=["gdp_growth","inflation","tbilrate"]),"Several series","Growth, inflation and the T-bill rate","Do three variables share one regime path?"),
   ("garrett","markov-guide/examples/garrett.csv",dict(vars=["gdp_growth"]),"Panel of countries","GDP growth in 14 countries","Do the countries share the same regimes?"),
   ("fed","markov-guide/examples/fedfunds.csv",dict(),"Trending levels","The federal funds rate","A series in levels: the case where Markov switching is the wrong first step.")]),
}

CSS_ADD = """
/* bayesian / markov selector additions */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}.stat{background:#f8fafe;border:1px solid #e6ebf4;border-radius:9px;padding:9px 10px}.stat b{display:block;font:800 17px Manrope;color:var(--navy)}.stat span{font-size:10.5px;color:var(--muted)}
.grid>*{min-width:0}.etable{display:block;max-width:100%;overflow-x:auto;border-collapse:collapse;font-size:12.5px;margin:10px 0}.etable th{font-size:9.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#7f8aa1;text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}.etable td{padding:8px;border-bottom:1px solid #edf1f7;vertical-align:top}
.tree .el{font-size:10.5px}.tree .n text{font-size:12.5px}.tree .n.leaf text{font-size:13px}
.ctrl{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:12px 0}.ctrl label,.vlist label,.known{font-size:11.5px;font-weight:700;color:var(--muted)}.ctrl label{display:flex;flex-direction:column;gap:5px}
.ctrl select{font:600 13px 'DM Sans',sans-serif;border:1px solid var(--line);border-radius:9px;padding:9px;background:#fff;color:var(--ink)}
.vlist{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 0 12px;align-items:center}.vlist .label{width:100%;margin-bottom:2px}.vlist label{font-weight:600;color:var(--ink);display:flex;gap:6px;align-items:center}
.known{display:flex;gap:8px;align-items:center;margin:0 0 14px;color:var(--ink);font-weight:600}
.result ul{margin:0 0 4px;padding-left:19px}.result li{margin:4px 0;font-size:13.5px;line-height:1.55}.result li code{background:#ffffff22;padding:1px 5px;border-radius:4px}
@media(max-width:650px){.stats{grid-template-columns:repeat(2,1fr)}.etable{font-size:11.5px}}
"""


def build(key):
    C = CASES[key]
    out = ROOT / C["folder"]
    (out / "examples").mkdir(exist_ok=True)
    shutil.copy(HERE / "bm_analysis.py", out / "examples" / "bm_analysis.py")
    examples = []
    for k, f, kw, meta, title, desc in C["examples"]:
        r = json.loads(json.dumps(B.analyse((ROOT / f).read_text(), **kw), default=float))
        assert "error" not in r, (k, r)
        print(key, k, {a: r.get(a) for a in ("time", "unit", "mixed_freq", "y")}, "n_series", r.get("n_series"))
        examples.append(dict(key=k, title=title, meta=meta, desc=desc, result=r))
    ptpl = (ROOT / "tools/panel_selector/template.html").read_text()
    ts = (ROOT / "tools/model_selector/template.html").read_text()
    css = ts[ts.index("<style>") + 7: ts.index("</style>")].strip()
    head = ptpl[: ptpl.index('<section class="hero">')]
    head = head.replace("/*TS_CSS*/", css).replace("</style>", CSS_ADD + "</style>", 1)
    head = re.sub(r"<title>.*?</title>", f"<title>{C['name']} | Tuhin G. M. Al Mamun</title>", head, 1)
    head = re.sub(r'(<meta name="description" content=")[^"]*"', lambda m: m.group(1) + C["desc"] + '"', head, 1)
    head = re.sub(r'(<link rel="canonical" href=")[^"]*"', lambda m: m.group(1) + f"https://tuhingm.github.io/almamun/{C['slug']}/" + '"', head, 1)
    hero = (f'<section class="hero"><div class="wrap"><div class="kicker">{C["kicker"]}</div><h1>{C["title"]}</h1><p class="lead">{C["lead"]}</p>'
            f'<div class="meta"><span><strong>{len(json.loads(json.dumps(C["examples"], default=str)))}</strong> real example datasets</span><span>Tests run in your browser</span><span>Python · statsmodels</span></div></div></section>\n')
    body = (HERE / "body.html").read_text().replace("#EXSUB#", C["exsub"]).replace("#GH#", C["gh"]).replace("#NOTE#", C["note"])
    foot = (f'<footer><div class="wrap foot"><span><b>Tuhin G. M. Al Mamun, Ph.D.</b> · {C["name"]}</span>'
            f'<span><a href="../time-series-guide/index.html">Time-Series Guide</a> · © <span id="year"></span></span></div></footer>\n')
    js = ((HERE / C["jsfile"]).read_text() + "\n" + (HERE / "common.js").read_text())
    page = (head + hero + body + foot +
            f'<script type="text/plain" id="py-src">{(HERE / "bm_analysis.py").read_text()}</script>\n<script>\nconst EXAMPLES={json.dumps(examples, default=float, separators=(",", ":"))};\n{js}</script>\n</body>\n</html>\n')
    page = page.replace("</script>\n</body>", "</script>\n</body>")
    (out / "index.html").write_text(page)
    print(key, "written", len(page) // 1024, "KB")


if __name__ == "__main__":
    for k in (sys.argv[1:] or CASES):
        build(k)
