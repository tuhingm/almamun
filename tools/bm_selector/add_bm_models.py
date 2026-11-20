#!/usr/bin/env python3
"""Register the extra Bayesian and Markov guides (built by tools/bayes_markov) in the decision trees of bayesian-guide/index.html and
markov-guide/index.html, and in the sources hub_bayesian.js / hub_markov.js. Adds a model card, a leaf, an edge, a guided question and the GUIDE link.
usage: python3 tools/bm_selector/add_bm_models.py   (idempotent)"""
import pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def js(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def model(k, name, short, when, why, steps):
    return f" {k}:{{name:{js(name)},short:{js(short)},when:{js(when)},why:{js(why)},steps:[{','.join(js(x) for x in steps)}]}}"


BAYES = {
    "BLP": ("blp", model("BLP", "Bayesian local projections", "Bayesian LP", "Impulse responses without a full VAR, smoother and with probability statements",
                         "You want the robustness of local projections to VAR misspecification, but their impulse responses are noisy at long horizons. A smoothness prior on how the response moves across horizons, with its tightness chosen by the marginal likelihood, shrinks the erratic part and gives posterior probabilities.",
                         ["Run the local projections", "Compare with a VAR", "Choose the smoothness prior", "Posterior statements", "Prior, lag and sample sensitivity", "Write up the responses"]),
            [830, 228, 100, "Bayesian LP"]),
    "SIGN": ("sign-bvar", model("SIGN", "Sign-restricted Bayesian VAR", "Sign-restricted BVAR", "A structural shock identified by theory-based sign restrictions",
                                "You can say how some variables must respond to a shock (a tightening raises the rate and lowers inflation and money growth) but cannot defend a recursive ordering. The model draws structural rotations that satisfy the restrictions and reports sets of responses.",
                                ["Estimate the BVAR", "Draw rotations that satisfy the signs", "Responses and credible sets", "How informative are the restrictions", "Restriction horizon and prior sensitivity", "Write up the identification"]),
             [942, 228, 116, "Sign-restricted"]),
    "HIER": ("hier-panel", model("HIER", "Bayesian hierarchical panel regression", "Hierarchical panel", "Effects differ across countries or groups and some groups have little data",
                                 "Each country has too few observations for its own slope, but you think the effects differ. A hierarchical model estimates a slope for each unit and the distribution the slopes come from, so noisy unit estimates are shrunk towards the average by an amount the data choose.",
                                 ["Look at the heterogeneity", "Gibbs sampler and convergence", "Partial pooling", "Out-of-sample check", "Prior and leave-one-out sensitivity", "Write up the pooling"]),
             [830, 296, 140, "Hierarchical panel"]),
}
# models added after the first release: key -> (slug, model, leaf label, edge label, question text); leaves are laid out automatically
EXTRA_BAYES = {
    "BMA": ("bma", model("BMA", "Bayesian model averaging", "Model averaging", "Many candidate regressors and no theory that says which belong",
                         "You have more candidate regressors than you can defend one by one, and choosing a single model by testing hides how much the answer depends on the choice. Model averaging weights every combination of regressors by its posterior probability and reports the probability that each variable belongs.",
                         ["Choose the g-prior and the model prior", "Posterior inclusion probabilities", "Best models and model size", "Compare with the full regression", "Prior sensitivity", "Check the search (MC3)"]),
            "Model averaging", "which variables", "Many candidate regressors and I do not know which belong"),
    "BLOGIT": ("bayes-logit", model("BLOGIT", "Bayesian logit and probit", "Bayesian logit/probit", "A yes/no outcome, with probabilities for the effects and a prior",
                                    "The outcome is binary (works or not, defaults or not). A Bayesian logit or probit reports the effect on the probability with credible intervals computed draw by draw, takes a prior, and keeps working when maximum likelihood fails, as with a rare outcome or perfect separation.",
                                    ["Choose the link and the prior", "Estimate and check the chains", "Odds ratios and marginal effects", "Posterior predictive check", "Prior and link sensitivity", "Separation and small samples"]),
               "Bayesian logit/probit", "binary outcome", "A yes/no outcome and I want probabilities and a prior"),
    "SHRINK": ("bayes-shrinkage", model("SHRINK", "Bayesian shrinkage regression (horseshoe, lasso, ridge)", "Shrinkage regression", "Many regressors relative to the sample: interactions, polynomials, dozens of controls",
                                        "OLS overfits when the model has almost as many parameters as observations. A shrinkage prior pulls the coefficients the data do not support towards zero and leaves the supported ones alone, and the posterior reports how sure you are about each.",
                                        ["Choose the prior", "Estimate and find the effects", "Compare with OLS", "Check the sampler", "Prior sensitivity", "Out-of-sample prediction"]),
               "Shrinkage regression", "many regressors", "I have more regressors than I can trust OLS with"),
    "CPT": ("change-points", model("CPT", "Bayesian change points", "Change points", "A level or a relationship that shifted at unknown dates",
                                   "A series or a relationship may have shifted at dates you do not know. The posterior over all segmentations is computed exactly, so you get the probability of 0, 1, 2 ... breaks and the probability of a break after each date, with regime coefficients, and no sampler to check.",
                                   ["Choose the prior on the number of breaks", "Number of breaks", "Break dates and regimes", "Compare with Bai-Perron", "Prior sensitivity", "False-alarm check"]),
            "Change points", "unknown dates", "A level or a relationship shifted at dates I do not know"),
    "QREG": ("quantile-regression", model("QREG", "Bayesian quantile regression", "Quantile regression", "The effect of a regressor may differ between low and high values of the outcome",
                                          "The mean hides who gains and who does not. Bayesian quantile regression estimates the effect at several quantiles, gives a posterior and a probability that the effects differ, and rescales the intervals so they have the right width.",
                                          ["Choose the quantiles", "Estimate at each quantile", "Do the effects differ", "Correct the intervals", "Check the sampler", "Simulation coverage"]),
             "Quantile regression", "effect differs", "The effect may differ between low and high outcomes"),
    "BSTS": ("structural-time-series", model("BSTS", "Bayesian structural time series", "Structural time series", "A series with a drifting level, trend or seasonal pattern to decompose, interpolate or forecast",
                                             "Split a series into a level, a trend and a seasonal pattern, fill in missing observations and forecast with an interval that carries every source of uncertainty. The Bayesian version gives a posterior for each component and for every forecast.",
                                             ["Choose the components", "Estimate and compare with the Kalman filter", "Check the sampler", "Fill in a gap", "Seasonal pattern and forecast", "Test the forecast on held-out data"]),
             "Structural time series", "level, trend, seasonal", "I want to decompose, interpolate or forecast a series"),
}
MARKOV = {
    "MSTVTP": ("ms-tvtp", model("MSTVTP", "Markov switching with time-varying transition probabilities", "MS with TVTP", "Hidden regimes, and an observed variable that moves the probability of switching",
                                "The regimes are hidden, but you think an observed variable (a policy rate, a spread, a confidence index) changes the probability of moving between them. A logit lets the probabilities depend on that variable, and tests whether the dependence is real.",
                                ["Estimate the constant-probability model", "Estimate with time-varying probabilities", "What the driver does to the probabilities", "Regime dates against a reference", "Permutation test", "Driver and start-value robustness"]),
               [800, 296, 150, "MS with TVTP"]),
}


def run(path, items, key):
    s = path.read_text()
    for k, (slug, mdl, node) in items.items():
        if f"\n {k}:{{name:" in s:
            continue
        assert "}},\nN:{" in s, "unexpected layout of " + str(path)
        s = s.replace("}},\nN:{", f"}},\n{mdl}}},\nN:{{", 1)
        s = re.sub(r"(GUIDE:\{[^}]*)\}", lambda m: m.group(1) + f",{k}:{js(slug)}}}", s, count=1)
        x, y, w, lab = node
        # node: append before the closing brace of N
        s = re.sub(r"(N:\{.*?)(\},\nE:\[)", lambda m: m.group(1) + f",\n {k}:[{x},{y},{w},{js(lab)},1]" + m.group(2), s, count=1, flags=re.S)
    path.write_text(s)


def tree_bayes(path):
    s = path.read_text()
    if "oth:[" in s:
        return
    s = s.replace("vb:'0 0 880 462'", "vb:'0 0 1080 462'", 1)
    s = re.sub(r"(N:\{.*?)(\},\nE:\[)", lambda m: m.group(1) + ",\n oth:[886,160,120,'More methods']" + m.group(2), s, count=1, flags=re.S)
    s = s.replace("['q1','q2','time series'],", "['q1','q2','time series'],['q1','oth','something else'],", 1)
    s = s.replace("['q5','BVAR','no']]", "['q5','BVAR','no'],['oth','BLP','responses'],['oth','SIGN','signs'],['oth','HIER','groups']]", 1)
    s = s.replace("TL:{start:'Your data',", "TL:{oth:'More methods',start:'Your data',", 1)
    s = s.replace("['Several time series','Macroeconomic series observed over time','q2']]}", "['Several time series','Macroeconomic series observed over time','q2'],['Something else: impulse responses, sign restrictions, groups that differ','Methods for particular questions','qx']]},\n qx:{t:'What is your goal?',o:[['Impulse responses without trusting a VAR, but smoother and with probabilities','Bayesian local projections','BLP'],['I can sign the responses but cannot defend a recursive ordering','Sign-restricted Bayesian VAR','SIGN'],['Effects differ across countries or groups and some groups have little data','Hierarchical panel regression','HIER']]}", 1)
    s = s.replace(" BLP:[", " BLP:[", 1)
    path.write_text(s)


def leaf_pos(i, label):
    """Extra Bayesian leaves form one column under the 'More methods' node, clear of the Markov-guide node on the left."""
    return [886, 364 + 68 * i, int(min(max(84, 8 * len(label) + 20), 150)), label]


def extend_bayes(path):
    s = path.read_text()
    base = 3                                                 # BLP, SIGN, HIER have fixed positions above
    for i, (k, (slug, mdl, label, edge, q)) in enumerate(EXTRA_BAYES.items()):
        if f"\n {k}:{{name:" in s:
            pass
        else:
            assert "}},\nN:{" in s
            s = s.replace("}},\nN:{", f"}},\n{mdl}}},\nN:{{", 1)
            s = re.sub(r"(GUIDE:\{[^}]*)\}", lambda m: m.group(1) + f",{k}:{js(slug)}}}", s, count=1)
            x, y, w, lab = leaf_pos(i, label)
            s = re.sub(r"(N:\{.*?)(\},\nE:\[)", lambda m: m.group(1) + f",\n {k}:[{x},{y},{w},{js(lab)},1]" + m.group(2), s, count=1, flags=re.S)
            last = list(re.finditer(r"\['oth','\w+','[^']*'\]", s))[-1]
            s = s[:last.end()] + f",['oth','{k}',{js(edge)}]" + s[last.end():]
            m = re.search(r"qx:\{t:'What is your goal\?',o:\[.*?\]\]\}", s, flags=re.S)
            assert m, "qx block not found"
            s = s[:m.end() - 2] + f",[{js(q)},{js(mdl.split('name:')[1].split(',short')[0].strip(chr(39)))},{js(k)}]" + s[m.end() - 2:]
    for i, (k, (slug, mdl, label, edge, q)) in enumerate(EXTRA_BAYES.items()):
        x, y, w, lab = leaf_pos(i, label)
        s = re.sub(rf" {k}:\[\d+,\d+,\d+,'[^']*',1\]", f" {k}:[{x},{y},{w},{js(lab)},1]", s)
    need_h = max(462, 364 + 68 * (len(EXTRA_BAYES) - 1) + 40)
    s = re.sub(r"vb:'0 0 1080 \d+'", f"vb:'0 0 1080 {need_h}'", s, count=1)
    path.write_text(s)


def tree_markov(path):
    s = path.read_text()
    if "['q2','MSTVTP'" in s:
        return
    s = s.replace("['q2','MSPVAR','panel']]", "['q2','MSPVAR','panel'],['q2','MSTVTP','switching driver']]", 1)
    s = s.replace("['A panel of countries that share the same regimes','A global slowdown or common volatility regime','MSPVAR']]", "['A panel of countries that share the same regimes','A global slowdown or common volatility regime','MSPVAR'],['Hidden regimes, with an observed variable that changes the probability of switching','A policy rate or a spread drives the switching','MSTVTP']]", 1)
    path.write_text(s)


def positions_bayes(path):
    s = path.read_text()
    for k, (slug, mdl, node) in BAYES.items():
        s = re.sub(rf" {k}:\[\d+,\d+,\d+,'[^']*',1\]", f" {k}:[{node[0]},{node[1]},{node[2]},{js(node[3])},1]", s)
    path.write_text(s)


def main():
    for hub, items, treefn in [("bayesian", BAYES, tree_bayes), ("markov", MARKOV, tree_markov)]:
        for p in [HERE / f"hub_{hub}.js", ROOT / f"{hub}-guide" / "index.html"]:
            run(p, items, hub); treefn(p)
            if hub == "bayesian":
                positions_bayes(p)
                extend_bayes(p)
    print("registered:", ", ".join(list(BAYES) + list(EXTRA_BAYES) + list(MARKOV)))


if __name__ == "__main__":
    main()
