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
    print("registered:", ", ".join(list(BAYES) + list(MARKOV)))


if __name__ == "__main__":
    main()
