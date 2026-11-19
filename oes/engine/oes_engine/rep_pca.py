"""Principal component analysis (stat_pca) publication report.

Steps: 1 data, 2 suitability (KMO, Bartlett), 3 eigenvalues and retention, 4 loadings, 5 component scores.
The decomposition is that of stat_multivariate.pca (same matrix, eigh, sign convention, parallel-analysis seed), so the numbers
equal the standard OES result; this layer adds the steps and figures around them.
"""
import math

import numpy as np

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, int_option, need
from .rep_model import Figure, Report, Step, TestResult
from .rep_tables import fnum, make_table, tests_table

FIGSTYLE = "Source: author's calculations."


def _hist(x, bins):
    c, e = np.histogram(x, bins=bins)
    return ((e[:-1] + e[1:]) / 2).tolist(), c.tolist(), float(e[1] - e[0])


def build(df, spec):
    import sys
    from .stat_multivariate import _num_matrix, _kmo, _bartlett_sphericity, _parallel
    from .stat_core import thin
    from . import __version__

    d, Xm, vs, dropped = _num_matrix(df, need(spec, "vars", "variables"), 2, 3)
    n, p = Xm.shape
    basis = spec.get("basis") or "correlation"
    seed = int_option(spec, "seed", 0, 0, 10 ** 9, "Random seed")
    Z = Xm - Xm.mean(0)
    if basis == "correlation":
        Z = Z / Xm.std(0, ddof=1)
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = np.clip(w[o], 0, None), V[:, o]
    for j in range(p):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    ratio = w / w.sum()
    cum = np.cumsum(ratio)
    pa = _parallel(n, p, 200, seed) if basis == "correlation" else None
    k_kaiser = int((w > (1 if basis == "correlation" else w.mean())).sum())
    k_pa = int(np.sum(w > pa)) if pa is not None else None
    k = spec.get("n_components")
    k = max(1, k_pa if k_pa else k_kaiser) if k in (None, "") else int(k)
    if not 1 <= k <= p:
        raise OESError(f"The number of components must be between 1 and {p}.")
    L = V * np.sqrt(w)
    scores = Z @ V
    comm = (L[:, :k] ** 2).sum(1) / (1 if basis == "correlation" else np.diag(C))
    comp = [f"PC{i + 1}" for i in range(p)]
    R = np.corrcoef(Xm, rowvar=False)

    rep = Report("stat_pca", "Principal component analysis of " + ", ".join(vs), references={})
    rep.meta = dict(vars=vs, nobs=n, basis=basis, components=k, engine=__version__, python=".".join(map(str, sys.version_info[:3])))
    data = d[vs].astype(float)

    s1 = Step("data", "Data and descriptive statistics", "Sample and distribution of every variable.", validated="pandas/scipy recomputation (test_pca_data)")
    s1.tables.append(T.descriptives_table("t_desc", data, vs, "Descriptive statistics"))
    s1.tables.append(T.correlation_table("t_corr", data, vs, "Correlation matrix"))
    s1.figures.append(Figure("f_corr", "Correlation matrix of the variables", {"ncols": 1, "panels": [{"layers": [{"type": "heatmap", "matrix": R.tolist(), "xlabels": vs, "ylabels": vs}], "ml": 70}]},
                             "Cells show Pearson correlations. Blue is negative, orange positive.", FIGSTYLE, 4.6, 4.0))
    rep.steps.append(s1)

    s2 = Step("suitability", "Suitability for components", "Kaiser-Meyer-Olkin measure of sampling adequacy and Bartlett's test of sphericity (correlation basis only).",
              validated="KMO and Bartlett statistic recomputed from the correlation matrix and its inverse (test_pca_suitability)")
    if basis == "correlation":
        kmo, msa = _kmo(R)
        chi, dfb, pb = _bartlett_sphericity(R, n)
        tt = [TestResult("Bartlett's test of sphericity", chi, pb, "The correlation matrix is the identity", f"{dfb:.0f}", bool(pb < 0.05), False, "")]
        s2.tests = tt
        s2.tables.append(tests_table("t_suit", "Bartlett's test of sphericity", tt, ["Rejection means the variables are correlated enough for components to summarise them."]))
        s2.tables.append(make_table("t_kmo", "Kaiser-Meyer-Olkin measure of sampling adequacy", ["Variable", "MSA"], [[v, fnum(m, 4)] for v, m in zip(vs, msa)] + [["Overall KMO", fnum(kmo, 4)]],
                                    ["Values above 0.6 are usually considered acceptable.", f"Determinant of the correlation matrix: {fnum(float(np.linalg.det(R)), 6)}."],
                                    raw={"kmo": kmo, "msa": [float(v) for v in msa]}))
        s2.figures.append(Figure("f_msa", "Sampling adequacy by variable", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": "MSA", "ylim": [0, 1], "xticks": [[float(i), v] for i, v in enumerate(vs)],
                                                                                                  "layers": [{"type": "bar", "x": [float(i) for i in range(p)], "y": [float(v) for v in msa], "fill": "#8a8a8a"}, {"type": "hline", "y": 0.6, "dash": (3, 2)}]}]},
                                 "Bars show each variable's measure of sampling adequacy; the dashed line marks 0.6.", FIGSTYLE, 6.0, 2.6))
    else:
        s2.tables.append(make_table("t_cond", "Conditioning of the analysed matrix", ["Statistic", "Value"], [["Condition number", fnum(float(w[0] / max(w[-1], 1e-300)), 2)]],
                                    ["KMO and Bartlett's test apply to the correlation basis only."]))
    rep.steps.append(s2)

    s3 = Step("eigen", "Eigenvalues and number of components", "Variance explained by each component and three retention rules.",
              validated="eigenvalues equal numpy.linalg.eigvalsh of the analysed matrix and sklearn's PCA explained variance; broken-stick from its definition; parallel analysis is Monte-Carlo and NOT VALIDATED against a reference (test_pca_eigen)")
    bs = np.array([sum(1.0 / i for i in range(j, p + 1)) / p for j in range(1, p + 1)])
    rows = [[comp[i], fnum(w[i], 4), fnum(ratio[i] * 100, 2), fnum(cum[i] * 100, 2), fnum(bs[i] * 100, 2), fnum(pa[i], 4) if pa is not None else "", "retain" if i < k else ""] for i in range(p)]
    s3.tables.append(make_table("t_eig", "Eigenvalues and variance explained", ["Component", "Eigenvalue", "% of variance", "Cumulative %", "Broken-stick %", "Parallel 95th pct.", f"Retained (k = {k})"], rows,
                                [f"Kaiser rule suggests {k_kaiser}; parallel analysis suggests {k_pa if k_pa is not None else 'n/a (covariance basis)'}.", "Parallel analysis uses 200 simulated data sets and the reported seed."],
                                raw={"eigenvalues": w.tolist(), "ratio": ratio.tolist()}))
    xt = [[float(i + 1), str(i + 1)] for i in range(p)]
    xs = [float(i + 1) for i in range(p)]
    lay = [{"type": "line", "x": xs, "y": w.tolist(), "k": 0, "label": "Eigenvalue"}, {"type": "scatter", "x": xs, "y": w.tolist(), "r": 2.4, "fill": "#000000", "alpha": 1.0}]
    if pa is not None:
        lay.append({"type": "line", "x": xs, "y": pa.tolist(), "k": 1, "label": "Parallel analysis (95th pct.)"})
        lay.append({"type": "hline", "y": 1.0, "dash": (3, 2)})
    s3.figures.append(Figure("f_scree", "Scree plot", {"ncols": 1, "panels": [{"xlabel": "Component", "ylabel": "Eigenvalue", "xticks": xt, "legend": pa is not None, "layers": lay}]},
                             "Eigenvalues of the analysed matrix by component." + (" The dashed line is the Kaiser threshold of 1." if pa is not None else ""), FIGSTYLE, 5.0, 3.2))
    s3.figures.append(Figure("f_cum", "Variance explained", {"ncols": 2, "panels": [
        {"title": "Share of variance", "xlabel": "Component", "ylabel": "Per cent", "xticks": xt, "layers": [{"type": "bar", "x": xs, "y": (ratio * 100).tolist(), "fill": "#8a8a8a"}, {"type": "line", "x": xs, "y": (bs * 100).tolist(), "k": 1}]},
        {"title": "Cumulative share", "xlabel": "Component", "ylabel": "Per cent", "xticks": xt, "layers": [{"type": "line", "x": xs, "y": (cum * 100).tolist(), "k": 0}, {"type": "hline", "y": 80.0, "dash": (3, 2)}]}]},
        "Left: variance share of each component (bars) against the broken-stick benchmark (line). Right: cumulative share; the dashed line is 80 per cent.", FIGSTYLE, 6.5, 2.7))
    rep.steps.append(s3)

    s4 = Step("loadings", "Loadings", f"Correlations of each variable with the first {k} components (correlation basis) or scaled eigenvectors (covariance basis), and the component weights.",
              validated="loadings and weights equal sklearn's components_ up to sign and the SVD of the standardised data; communalities from the squared loadings (test_pca_loadings)")
    s4.tables.append(make_table("t_load", "Loadings", ["Variable"] + comp[:k] + ["Communality"], [[v] + [fnum(L[i, j], 4) for j in range(k)] + [fnum(comm[i], 4)] for i, v in enumerate(vs)],
                                ["Sign convention: the largest absolute weight of each component is positive."], raw={"loadings": L[:, :k].tolist()}))
    s4.tables.append(make_table("t_vec", "Component weights (eigenvectors)", ["Variable"] + comp[:k], [[v] + [fnum(V[i, j], 5) for j in range(k)] for i, v in enumerate(vs)], raw={"weights": V[:, :k].tolist()}))
    s4.figures.append(Figure("f_loadmap", "Loadings heat map", {"ncols": 1, "panels": [{"layers": [{"type": "heatmap", "matrix": L[:, :k].tolist(), "xlabels": comp[:k], "ylabels": vs}], "ml": 70}]},
                             "Cells show the loading of each variable on each retained component. Blue is negative, orange positive.", FIGSTYLE, 4.0 + 0.5 * k, 1.2 + 0.45 * p))
    kk = min(k, 4)
    s4.figures.append(Figure("f_loadbars", "Loadings on the leading components", {"ncols": 2 if kk > 1 else 1, "panels": [
        {"title": comp[j], "xlabel": "", "ylabel": "Loading", "xticks": [[float(i), v] for i, v in enumerate(vs)],
         "layers": [{"type": "bar", "x": [float(i) for i in range(p)], "y": L[:, j].tolist(), "fill": "#8a8a8a"}, {"type": "hline", "y": 0.0}]} for j in range(kk)]},
        "Bars show each variable's loading on the component named above the panel." + (f" Only the first {kk} retained components are drawn." if k > kk else ""), FIGSTYLE, 6.5, 2.5 * math.ceil(kk / 2) + 0.4))
    s4.figures.append(Figure("f_comm", "Communalities", {"ncols": 1, "panels": [{"xlabel": "", "ylabel": "Communality", "ylim": [0, 1], "xticks": [[float(i), v] for i, v in enumerate(vs)],
                                                                                "layers": [{"type": "bar", "x": [float(i) for i in range(p)], "y": comm.tolist(), "fill": "#8a8a8a"}]}]},
                             f"Share of each variable's variance reproduced by the first {k} component(s).", FIGSTYLE, 6.0, 2.6))
    rep.steps.append(s4)

    s5 = Step("scores", "Component scores", "Observations in the space of the leading components.",
              validated="scores equal the centred (and scaled) data times the eigenvectors, and sklearn's transform up to sign (test_pca_scores)")
    a, b = thin(list(scores[:, 0]), list(scores[:, 1]), 400)
    sc = np.abs(scores[:, :2]).max()
    ld = L[:, :2] / max(np.abs(L[:, :2]).max(), 1e-12) * sc * 0.9
    s5.figures.append(Figure("f_scores", "Component scores", {"ncols": 1, "panels": [{"xlabel": "PC1", "ylabel": "PC2", "layers": [{"type": "scatter", "x": a, "y": b}, {"type": "hline", "y": 0.0}, {"type": "vline", "x": 0.0}]}]},
                             "Each point is one observation (at most 400 points are drawn).", FIGSTYLE, 4.6, 3.8))
    s5.figures.append(Figure("f_biplot", "Biplot", {"ncols": 1, "panels": [{"xlabel": "PC1", "ylabel": "PC2", "layers": [{"type": "scatter", "x": a, "y": b, "alpha": 0.25}, {"type": "hline", "y": 0.0}, {"type": "vline", "x": 0.0}]
                                                                           + [{"type": "line", "x": [0.0, float(ld[i, 0])], "y": [0.0, float(ld[i, 1])], "k": 0} for i in range(p)]
                                                                           + [{"type": "text", "x": float(ld[i, 0]), "y": float(ld[i, 1]), "s": vs[i], "size": 7} for i in range(p)]}]},
                             "Points: observations. Lines: variables, drawn from the origin to their loadings on PC1 and PC2 (rescaled to the score range).", FIGSTYLE, 4.8, 4.0))
    hx, hy, bw = _hist(scores[:, 0], 20)
    s5.figures.append(Figure("f_hist", "Distribution of the first component score", {"ncols": 1, "panels": [{"xlabel": "PC1 score", "ylabel": "Count", "layers": [{"type": "bar", "x": hx, "y": hy, "width": bw, "fill": "#8a8a8a"}]}]},
                             "Histogram of the PC1 scores with 20 bins.", FIGSTYLE, 4.6, 3.0))
    rep.steps.append(s5)

    rep.methods.append(X.P("method", "We extract principal components from the {basis} matrix of {p} variables on {n} observations by eigen-decomposition. The number of components retained is {k}, chosen by {how}. Sampling adequacy is judged with the Kaiser-Meyer-Olkin measure and Bartlett's test. Estimation used the Open Econometric Systems engine (version {ver}) in Python {py}.",
                           basis=basis, p=p, n=n, k=k, how=("the user" if spec.get("n_components") not in (None, "") else ("parallel analysis" if pa is not None else "the Kaiser rule")), ver=__version__, py=rep.meta["python"]))
    if dropped:
        rep.limitations.append(X.P("note", "{m} row(s) with missing values were dropped.", m=int(dropped)))
    rep.limitations.append(X.P("note", "Parallel analysis is a Monte-Carlo benchmark and is not validated against a reference implementation. Components are linear summaries and have no causal meaning."))
    return rep.finalize()
