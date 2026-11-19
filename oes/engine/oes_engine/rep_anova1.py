"""One-way ANOVA publication report (Statistics Division, A1).

Steps: 1 descriptive statistics by group, 2 assumption checks (normality, equal variances), 3 the ANOVA table with effect sizes,
and tests that do not assume equal variances, 4 post-hoc pairwise comparisons (Tukey-Kramer, Games-Howell) with standardised
differences. The F test, effect sizes and robust tests are the ones of stat_anova.anova1 (same formulas); this layer arranges them
into steps, tables, figures and text and adds the assumption tests and post-hoc tables.
"""
import itertools
import math

import numpy as np
from scipy import stats

from . import rep_tables as T
from . import rep_text as X
from .core import OESError, distinct, need
from .rep_model import Figure, Report, Step, TestResult
from .rep_tables import fint, fnum, fp, make_table, tests_table
from .stat_core import clean_frame, eta2_ci, levels, qq_points, thin

REFS = {
    "fisher1925": "Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver and Boyd.",
    "welch1951": "Welch, B. L. (1951). On the comparison of several mean values: an alternative approach. Biometrika, 38(3-4), 330-336.",
    "brownforsythe1974": "Brown, M. B., & Forsythe, A. B. (1974). The small sample behavior of some statistics which test the equality of several means. Technometrics, 16(1), 129-132.",
    "levene1960": "Levene, H. (1960). Robust tests for equality of variances. In I. Olkin (Ed.), Contributions to Probability and Statistics (pp. 278-292). Stanford University Press.",
    "bartlett1937": "Bartlett, M. S. (1937). Properties of sufficiency and statistical tests. Proceedings of the Royal Society of London A, 160(901), 268-282.",
    "shapirowilk1965": "Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3-4), 591-611.",
    "tukey1953": "Tukey, J. W. (1953). The problem of multiple comparisons. Unpublished manuscript, Princeton University.",
    "kramer1956": "Kramer, C. Y. (1956). Extension of multiple range tests to group means with unequal numbers of replications. Biometrics, 12(3), 307-310.",
    "gameshowell1976": "Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal n's and/or variances: a Monte Carlo study. Journal of Educational Statistics, 1(2), 113-125.",
    "hedges1981": "Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and related estimators. Journal of Educational Statistics, 6(2), 107-128.",
    "kruskal1952": "Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260), 583-621.",
    "steiger2004": "Steiger, J. H. (2004). Beyond the F test: effect size confidence intervals and tests of close fit in the analysis of variance and contrast analysis. Psychological Methods, 9(2), 164-182.",
    "cohen1988": "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum.",
    "scipy2020": "Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",
}


def effect_label(e):
    return "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def games_howell(data, alpha=0.05):
    """Games-Howell comparisons for every pair: returns rows (i, j, diff, se, df, q, p, lo, hi)."""
    k = len(data)
    n = np.array([len(v) for v in data])
    m = np.array([v.mean() for v in data])
    v = np.array([x.var(ddof=1) for x in data])
    out = []
    for i, j in itertools.combinations(range(k), 2):
        se2 = v[i] / n[i] + v[j] / n[j]
        se = math.sqrt(se2)
        dfw = se2 ** 2 / ((v[i] / n[i]) ** 2 / (n[i] - 1) + (v[j] / n[j]) ** 2 / (n[j] - 1))
        d = m[j] - m[i]
        q = abs(d) / se * math.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, dfw))
        qc = float(stats.studentized_range.ppf(1 - alpha, k, dfw))
        half = qc / math.sqrt(2) * se
        out.append((i, j, float(d), se, float(dfw), float(q), p, float(d - half), float(d + half)))
    return out


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (b.mean() - a.mean()) / sp
    J = 1 - 3 / (4 * (na + nb - 2) - 1)
    g = d * J
    se = math.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return float(g), float(se)


def boxstats(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    return dict(lo=float(inside.min()), q1=float(q1), med=float(med), q3=float(q3), hi=float(inside.max()), out=[float(x) for x in v[(v < lo_f) | (v > hi_f)]])


def build(df, spec):
    import sys
    import scipy
    from . import __version__
    y = need(spec, "y", "outcome variable")
    g = need(spec, "group", "grouping variable")
    distinct(y, g)
    alpha = float(spec.get("alpha") or 0.05)
    if not 0 < alpha < 1:
        raise OESError("The significance level must be between 0 and 1.")
    lvl = int(round((1 - alpha) * 100))
    d, dropped = clean_frame(df, numeric=[y], factors=[g])
    lv = levels(d[g])
    lab = [str(l) for l in lv]
    k = len(lv)
    if k < 2:
        raise OESError("The grouping variable needs at least two groups.")
    data = [d.loc[d[g] == l, y].to_numpy(float) for l in lv]
    n_i = np.array([len(v) for v in data])
    if (n_i < 2).any():
        raise OESError("Every group needs at least two observations. Groups with one observation: " + ", ".join(l for l, n in zip(lab, n_i) if n < 2))
    N = int(n_i.sum())
    means = np.array([v.mean() for v in data])
    sds = np.array([v.std(ddof=1) for v in data])
    grand = float(np.concatenate(data).mean())
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum((n_i - 1) * sds ** 2))
    sst = ssb + ssw
    df1, df2 = k - 1, N - k
    msb, msw = ssb / df1, ssw / df2
    if msw <= 0:
        raise OESError("There is no variation within the groups, so the F test is undefined.")
    F = msb / msw
    p = float(stats.f.sf(F, df1, df2))
    eta = ssb / sst
    omega = max(0.0, (ssb - df1 * msw) / (sst + msw))
    eps = max(0.0, (ssb - df1 * msw) / sst)
    cohen_f = math.sqrt(eta / (1 - eta)) if eta < 1 else float("inf")
    elo, ehi = eta2_ci(F, df1, df2, N, 1 - alpha)
    resid = np.concatenate([v - v.mean() for v in data])
    fitted = np.concatenate([np.full(len(v), v.mean()) for v in data])

    rep = Report("stat_anova1", f"One-way analysis of variance of {y} by {g}", references=REFS)
    rep.meta = dict(y=y, group=g, groups=lab, nobs=N, alpha=alpha, engine=__version__, scipy=scipy.__version__, python=".".join(map(str, sys.version_info[:3])))
    rep.meta["extra_refs"] = ["scipy2020", "cohen1988"]
    pos = list(range(k))
    xt = [[i, lab[i]] for i in pos]
    gx = [float(i) for i in pos]

    # ------------------------------------------------------------------ 1 descriptives
    tc = stats.t.ppf(1 - alpha / 2, n_i - 1)
    s1 = Step("data", "Descriptive statistics by group", "Sample size, centre and spread of the outcome in every group before any test is run.",
              validated="group means, standard deviations, t-based confidence intervals, quartiles recomputed with numpy/scipy (test_anova_descriptives)", refs=["fisher1925"])
    dec = T.decimals_for(list(means) + list(sds), 2, 4)
    rows = []
    for i in range(k):
        se = sds[i] / math.sqrt(n_i[i])
        rows.append([lab[i], fint(n_i[i]), fnum(means[i], dec), fnum(sds[i], dec), fnum(se, dec), fnum(means[i] - tc[i] * se, dec), fnum(means[i] + tc[i] * se, dec),
                     fnum(data[i].min(), dec), fnum(np.median(data[i]), dec), fnum(data[i].max(), dec)])
    allv = np.concatenate(data)
    rows.append(T._row(["All groups", fint(N), fnum(grand, dec), fnum(allv.std(ddof=1), dec), fnum(allv.std(ddof=1) / math.sqrt(N), dec), "", "",
                        fnum(allv.min(), dec), fnum(np.median(allv), dec), fnum(allv.max(), dec)], "stat"))
    s1.tables.append(make_table("t_desc", f"Descriptive statistics of {y} by {g}", ["Group", "N", "Mean", "Std. dev.", "Std. err.", f"{lvl}% CI low", f"{lvl}% CI high", "Min", "Median", "Max"], rows,
                                [f"Std. dev. uses n-1; each group's {lvl}% confidence interval for the mean uses that group's own standard deviation and a t distribution."],
                                raw={"groups": lab, "n": [int(v) for v in n_i], "mean": means.tolist(), "sd": sds.tolist()}))
    boxes = [boxstats(v) for v in data]
    s1.figures.append(Figure("f_box", f"Distribution of {y} by {g}",
                             {"ncols": 1, "panels": [{"xlabel": g, "ylabel": y, "xticks": xt, "xlim": [-0.6, k - 0.4],
                                                      "layers": [{"type": "box", "x": gx, "boxes": boxes},
                                                                 {"type": "errv", "x": gx, "y": means.tolist(), "lo": [float(means[i] - tc[i] * sds[i] / math.sqrt(n_i[i])) for i in pos],
                                                                  "hi": [float(means[i] + tc[i] * sds[i] / math.sqrt(n_i[i])) for i in pos]}]}]},
                             f"Boxes span the quartiles, the bar is the median and whiskers reach the most extreme values within 1.5 interquartile ranges; circles are outliers. The dot and thin interval show the group mean and its {lvl}% confidence interval.",
                             "Source: author's calculations.", 4.6, 3.2))
    jit = np.random.RandomState(7)
    px, py = [], []
    for i, v in enumerate(data):
        xs_, ys_ = thin([i + jit.uniform(-0.18, 0.18) for _ in v], v.tolist(), 150)
        px += xs_
        py += ys_
    s1.figures.append(Figure("f_means", f"Group means with {lvl}% confidence intervals",
                             {"ncols": 1, "panels": [{"xlabel": g, "ylabel": y, "xticks": xt, "xlim": [-0.6, k - 0.4], "layers": [
                                 {"type": "scatter", "x": px, "y": py, "r": 1.3, "alpha": 0.35},
                                 {"type": "errv", "x": gx, "y": means.tolist(), "lo": [float(means[i] - tc[i] * sds[i] / math.sqrt(n_i[i])) for i in pos],
                                  "hi": [float(means[i] + tc[i] * sds[i] / math.sqrt(n_i[i])) for i in pos]}]}]},
                             "Grey points are individual observations (spread horizontally within a group for legibility; at most 150 per group are drawn). Dots and bars are group means with their confidence intervals.",
                             "Source: author's calculations.", 4.6, 3.2))
    dl = []
    lo_all, hi_all = float(allv.min()), float(allv.max())
    gridd = np.linspace(lo_all - 0.1 * (hi_all - lo_all), hi_all + 0.1 * (hi_all - lo_all), 120)
    for i in pos:
        if np.ptp(data[i]) > 0:
            dl.append({"type": "line", "x": gridd.tolist(), "y": stats.gaussian_kde(data[i])(gridd).tolist(), "k": i, "label": lab[i]})
    if dl:
        s1.figures.append(Figure("f_density", f"Kernel density of {y} by {g}", {"ncols": 1, "panels": [{"xlabel": y, "ylabel": "Density", "layers": dl}]},
                                 "Gaussian kernel density estimates with the default bandwidth (Scott's rule); every group has its own line style.", "Source: author's calculations.", 4.6, 3.0))
    s1.paragraphs.append(X.P("result", "The sample contains {N} observations in {k} groups defined by {g_} ({sizes}). Group means of {y} range from {lo:.3f} ({lol}) to {hi:.3f} ({hil}).{drop}",
                             N=N, k=k, g_=g, sizes=X.join([f"{lab[i]}: n = {int(n_i[i])}" for i in pos]), y=y, lo=float(means.min()), lol=lab[int(means.argmin())],
                             hi=float(means.max()), hil=lab[int(means.argmax())], drop=(f" {dropped} rows were dropped because the outcome or the group was missing." if dropped else "")))
    if n_i.max() / n_i.min() > 1.5:
        s1.paragraphs.append(X.P("note", "The design is unbalanced (group sizes range from {a} to {b}); the F test is then more sensitive to unequal variances, and the Welch test in Step 3 is the safer primary test.", a=int(n_i.min()), b=int(n_i.max())))
    rep.steps.append(s1)

    # ------------------------------------------------------------------ 2 assumptions
    s2 = Step("assumptions", "Assumption checks", "ANOVA assumes independent observations, normally distributed residuals and equal variances across groups.",
              validated="scipy.stats.shapiro / levene / bartlett called directly and compared with the statistic stored in the report (test_anova_assumptions)", refs=["shapirowilk1965", "levene1960", "bartlett1937"])
    tests = []

    def add(name, st_, pv_, null, dfm="", ref="", problem=True):
        tests.append(TestResult(name, float(st_), float(pv_), null, dfm, bool(pv_ < alpha), problem, ref))

    for i in pos:
        if 3 <= n_i[i] <= 5000 and np.ptp(data[i]) > 0:
            w = stats.shapiro(data[i])
            add(f"Shapiro-Wilk, {lab[i]}", w.statistic, w.pvalue, "The group is normally distributed", "", "shapirowilk1965")
    if np.ptp(resid) > 0 and N <= 5000:
        w = stats.shapiro(resid)
        add("Shapiro-Wilk, all residuals", w.statistic, w.pvalue, "The residuals are normally distributed", "", "shapirowilk1965")
    lv_bf = stats.levene(*data, center="median")
    lv_mn = stats.levene(*data, center="mean")
    bt = stats.bartlett(*data)
    add("Brown-Forsythe (Levene, median)", lv_bf.statistic, lv_bf.pvalue, "Equal variances in all groups", f"{df1}, {df2}", "brownforsythe1974")
    add("Levene (mean)", lv_mn.statistic, lv_mn.pvalue, "Equal variances in all groups", f"{df1}, {df2}", "levene1960")
    add("Bartlett", bt.statistic, bt.pvalue, "Equal variances in all groups (assumes normality)", f"{df1}", "bartlett1937")
    s2.tests = tests
    s2.tables.append(tests_table("t_assump", "Tests of the ANOVA assumptions", tests,
                                [f"Decisions are at the {int(alpha * 100) if alpha * 100 == int(alpha * 100) else alpha * 100}% level. For these tests rejection signals a violated assumption. The Brown-Forsythe version of Levene's test is robust to non-normality; Bartlett's test is more powerful but sensitive to it."]))
    z, r_std = qq_points(resid)
    qx, qy = thin(z.tolist(), r_std.tolist(), 200)
    zl = [float(z.min()), float(z.max())]
    sx, sy = thin(fitted.tolist(), resid.tolist(), 250)
    nb = int(min(30, max(8, round(math.sqrt(N)))))
    cnt, edges = np.histogram(resid, bins=nb, density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    grid = np.linspace(edges[0], edges[-1], 80)
    W1, H1 = 3.2, 2.7
    s2.figures.append(Figure("f_qq", "Normal quantile plot of the residuals",
                             {"ncols": 1, "panels": [{"xlabel": "Theoretical quantiles", "ylabel": "Standardised residuals",
                                                      "layers": [{"type": "scatter", "x": qx, "y": qy}, {"type": "line", "x": zl, "y": zl, "k": 2}]}]},
                             "Points near the dashed 45-degree line indicate normally distributed residuals; systematic curvature indicates skewness or heavy tails.", "Source: author's calculations.", W1, H1))
    s2.figures.append(Figure("f_resfit", "Residuals against group mean",
                             {"ncols": 1, "panels": [{"xlabel": f"Group mean of {y}", "ylabel": "Residual", "zero": True, "layers": [{"type": "scatter", "x": sx, "y": sy}]}]},
                             "A funnel shape (wider scatter at larger group means) indicates unequal variances.", "Source: author's calculations.", W1, H1))
    s2.figures.append(Figure("f_spread", "Group standard deviation against group mean",
                             {"ncols": 1, "panels": [{"xlabel": "Group mean", "ylabel": "Group std. dev.", "layers": [{"type": "scatter", "x": means.tolist(), "y": sds.tolist(), "r": 2.6, "alpha": 0.9}]}]},
                             "Each point is one group. A rising pattern means that the spread grows with the level, so equal variances are doubtful.", "Source: author's calculations.", W1, H1))
    s2.figures.append(Figure("f_reshist", "Distribution of the residuals",
                             {"ncols": 1, "panels": [{"xlabel": "Residual", "ylabel": "Density",
                                                      "layers": [{"type": "bar", "x": mid.tolist(), "y": cnt.tolist(), "width": float(edges[1] - edges[0]), "fill": "#c8c8c8"},
                                                                 {"type": "line", "x": grid.tolist(), "y": stats.norm.pdf(grid, resid.mean(), resid.std(ddof=1)).tolist(), "k": 2, "lw": 0.9}]}]},
                             "Histogram of the within-group residuals; the dashed line is the normal density with the same mean and standard deviation.", "Source: author's calculations.", W1, H1))
    bad = [t for t in tests if t.reject]
    bad_norm = [t for t in bad if t.name.startswith("Shapiro")]
    bad_var = [t for t in bad if not t.name.startswith("Shapiro")]
    if np.ptp(resid) > 0 and N <= 5000:
        swr = stats.shapiro(resid)
        s2.paragraphs.append(X.P("result", "Equal variances were assessed with the Brown-Forsythe version of Levene's test, F({d1}, {d2}) = {f:.3f}, {p}. Normality of the residuals was assessed with the Shapiro-Wilk test, W = {w:.3f}, {pw}.",
                                 d1=df1, d2=df2, f=float(lv_bf.statistic), p=X.pv(lv_bf.pvalue), w=float(swr.statistic), pw=X.pv(swr.pvalue)))
    else:
        s2.paragraphs.append(X.P("result", "Equal variances were assessed with the Brown-Forsythe version of Levene's test, F({d1}, {d2}) = {f:.3f}, {p}. The Shapiro-Wilk test was not computed because it needs between 3 and 5000 residuals that are not all equal.",
                                 d1=df1, d2=df2, f=float(lv_bf.statistic), p=X.pv(lv_bf.pvalue)))
    if bad_var:
        s2.paragraphs.append(X.P("warning", "The assumption of equal variances is rejected by {names} at the {lv}% level. The classical F test may then give a wrong p-value, especially with unequal group sizes; rely on the Welch and Brown-Forsythe tests in Step 3 and on the Games-Howell comparisons in Step 4.",
                                 names=X.join([t.name for t in bad_var]), lv=int(round(alpha * 100))))
    if bad_norm:
        s2.paragraphs.append(X.P("warning", "Normality is rejected by {names} at the {lv}% level. With group sizes of {nmin} or more the F test is fairly robust to moderate non-normality, but heavy tails or strong skewness can distort it; the Kruskal-Wallis test in Step 3 does not assume normality.",
                                 names=X.join([t.name for t in bad_norm]), lv=int(round(alpha * 100)), nmin=int(n_i.min())))
    if not bad:
        s2.paragraphs.append(X.P("result", "None of the assumption tests rejects at the {lv}% level. This does not prove the assumptions hold, and with small groups these tests have little power.", lv=int(round(alpha * 100))))
    rep.steps.append(s2)

    # ------------------------------------------------------------------ 3 ANOVA table
    s3 = Step("anova", "Analysis of variance", "The omnibus test of equal group means, its effect sizes and tests that do not assume equal variances.",
              validated="F, p, sums of squares equal scipy.stats.f_oneway and statsmodels anova_lm; eta/omega squared closed forms; Welch equals statsmodels anova_oneway(use_var='unequal'); equals oes_engine.run('stat_anova1') (test_anova_table)",
              refs=["fisher1925", "welch1951", "brownforsythe1974", "kruskal1952", "steiger2004", "cohen1988"])
    s3.tables.append(make_table("t_anova", f"One-way analysis of variance of {y} by {g}", ["Source", "Sum of squares", "d.f.", "Mean square", "F", "p-value", "η²", "ω²"],
                                [["Between groups", fnum(ssb, 3), fint(df1), fnum(msb, 3), fnum(F, 3), fp(p), fnum(eta, 3), fnum(omega, 3)],
                                 ["Within groups", fnum(ssw, 3), fint(df2), fnum(msw, 3), "", "", "", ""],
                                 T._row(["Total", fnum(sst, 3), fint(N - 1), "", "", "", "", ""], "stat")],
                                [f"η² is the share of the total variance explained by {g}; ω² is its less biased estimate. Reading: η² below 0.06 is small, 0.06 to 0.14 medium, above 0.14 large (Cohen, 1988)."],
                                raw=dict(ssb=ssb, ssw=ssw, df1=df1, df2=df2, F=F, p=p, eta2=eta, omega2=omega)))
    w_ = n_i / sds ** 2
    W_ = w_.sum()
    mw = (w_ * means).sum() / W_
    lam = ((1 - w_ / W_) ** 2 / (n_i - 1)).sum()
    Fw = ((w_ * (means - mw) ** 2).sum() / df1) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2w = (k ** 2 - 1) / (3 * lam)
    pw = float(stats.f.sf(Fw, df1, df2w))
    c_i = 1 - n_i / N
    Fb = ssb / np.sum(c_i * sds ** 2)
    dfb = np.sum(c_i * sds ** 2) ** 2 / np.sum(c_i ** 2 * sds ** 4 / (n_i - 1))
    pb = float(stats.f.sf(Fb, df1, dfb))
    kw = stats.kruskal(*data)
    s3.tables.append(make_table("t_robust", "Tests that do not assume equal variances or normality", ["Test", "Statistic", "d.f. 1", "d.f. 2", "p-value"],
                                [["Welch ANOVA (F)", fnum(Fw, 3), fint(df1), fnum(df2w, 2), fp(pw)], ["Brown-Forsythe (F)", fnum(Fb, 3), fint(df1), fnum(dfb, 2), fp(pb)],
                                 ["Kruskal-Wallis (H)", fnum(kw.statistic, 3), fint(df1), "", fp(kw.pvalue)]],
                                ["Welch and Brown-Forsythe allow unequal group variances; Kruskal-Wallis compares rank distributions and does not assume normality (tie-corrected)."],
                                raw=dict(welch=[Fw, df2w, pw], brown_forsythe=[Fb, dfb, pb], kruskal=[float(kw.statistic), float(kw.pvalue)])))
    s3.tables.append(make_table("t_effect", "Effect sizes", ["Measure", "Estimate", f"{lvl}% CI low", f"{lvl}% CI high", "Reading"],
                                [["Eta squared (η²)", fnum(eta, 3), fnum(elo, 3), fnum(ehi, 3), effect_label(eta)], ["Omega squared (ω²)", fnum(omega, 3), "", "", effect_label(omega)],
                                 ["Epsilon squared (ε²)", fnum(eps, 3), "", "", effect_label(eps)], ["Cohen's f", fnum(cohen_f, 3), "", "", "0.10 small, 0.25 medium, 0.40 large"]],
                                [f"The interval for η² inverts the noncentral F distribution (Steiger, 2004). Intervals are {lvl}%."],
                                raw=dict(eta2=[eta, elo, ehi], omega2=omega, epsilon2=eps, cohen_f=cohen_f)))
    ptxt = X.pv(p)
    s3.paragraphs.append(X.P("result", "A one-way analysis of variance " + ("found a statistically significant" if p < alpha else "did not find a statistically significant") +
                             " difference in mean {y} across the levels of {g}, F({d1}, {d2}) = {F:.3f}, {p}, η² = {e:.3f} ({lvl}% CI [{lo:.3f}, {hi:.3f}]), a {lab_} effect by conventional benchmarks (ω² = {om:.3f}).",
                             y=y, g=g, d1=df1, d2=df2, F=F, p=ptxt, e=eta, lvl=lvl, lo=elo, hi=ehi, lab_=effect_label(eta), om=omega))
    s3.paragraphs.append(X.P("result", "The Welch test, which does not assume equal variances, gives F({d1}, {d2:.2f}) = {Fw:.3f}, {pw}; the Kruskal-Wallis test gives H({d1}) = {H:.3f}, {pk}.",
                             d1=df1, d2=float(df2w), Fw=float(Fw), pw=X.pv(pw), H=float(kw.statistic), pk=X.pv(kw.pvalue)))
    if (p < alpha) != (pw < alpha) or (p < alpha) != (kw.pvalue < alpha):
        s3.paragraphs.append(X.P("warning", "The classical F test, the Welch test and the Kruskal-Wallis test do not all reach the same decision at the {lv}% level. The conclusion about equal means depends on the assumptions made; report all three and state which assumption you consider credible.", lv=int(round(alpha * 100))))
    rep.steps.append(s3)

    # ------------------------------------------------------------------ 4 post hoc
    s4 = Step("posthoc", "Post-hoc pairwise comparisons", "Which groups differ, with familywise error control, and how large each difference is.",
              validated="Tukey-Kramer equals statsmodels pairwise_tukeyhsd and scipy.stats.tukey_hsd; Games-Howell and Hedges' g recomputed from their closed forms with scipy.stats.studentized_range (no third-party reference implementation is installed) (test_anova_posthoc)",
              refs=["tukey1953", "kramer1956", "gameshowell1976", "hedges1981"])
    pairs = list(itertools.combinations(range(k), 2))
    tk = stats.tukey_hsd(*data)
    cit = tk.confidence_interval(1 - alpha)
    rows, errs = [], []
    tuk = []
    for (i, j) in pairs:
        diff = float(means[j] - means[i])
        se = math.sqrt(msw / 2 * (1 / n_i[i] + 1 / n_i[j]))
        pj = float(tk.pvalue[i, j])
        # scipy reports mean(i) - mean(j); the table shows mean(j) - mean(i), so the interval is reflected
        lo_, hi_ = -float(cit.high[i, j]), -float(cit.low[i, j])
        rows.append([f"{lab[j]} − {lab[i]}", fnum(diff, 3), fnum(se, 3), fnum(lo_, 3), fnum(hi_, 3), fp(pj) + T.stars(pj)])
        errs.append((f"{lab[j]} − {lab[i]}", diff, lo_, hi_))
        tuk.append(dict(pair=[lab[i], lab[j]], diff=diff, p=pj, lo=lo_, hi=hi_))
    s4.tables.append(make_table("t_tukey", "Tukey-Kramer pairwise comparisons", ["Comparison", "Mean difference", "Std. err.", f"{lvl}% CI low", f"{lvl}% CI high", "Adjusted p-value"], rows,
                                ["Uses the pooled within-group variance; adjusts p-values for all pairwise comparisons (Tukey-Kramer when group sizes differ). The difference is the second group minus the first.", T.STAR_NOTE],
                                raw={"pairs": tuk}))
    gh = games_howell(data, alpha)
    grows = [[f"{lab[j]} − {lab[i]}", fnum(dd, 3), fnum(se, 3), fnum(dfw, 2), fnum(lo_, 3), fnum(hi_, 3), fp(pp) + T.stars(pp)] for (i, j, dd, se, dfw, q, pp, lo_, hi_) in gh]
    s4.tables.append(make_table("t_gh", "Games-Howell pairwise comparisons (unequal variances)", ["Comparison", "Mean difference", "Std. err.", "d.f.", f"{lvl}% CI low", f"{lvl}% CI high", "Adjusted p-value"], grows,
                                ["Does not assume equal variances or equal group sizes; Welch degrees of freedom and the studentised range distribution.", T.STAR_NOTE],
                                raw={"pairs": [dict(pair=[lab[a], lab[b]], diff=dd, se=se, df=dfw, p=pp, lo=lo_, hi=hi_) for (a, b, dd, se, dfw, q, pp, lo_, hi_) in gh]}))
    hg = []
    hrows = []
    for (i, j) in pairs:
        gg, se = hedges_g(data[i], data[j])
        z_ = stats.norm.ppf(1 - alpha / 2)
        hg.append((gg, gg - z_ * se, gg + z_ * se))
        hrows.append([f"{lab[j]} − {lab[i]}", fnum(gg, 3), fnum(gg - z_ * se, 3), fnum(gg + z_ * se, 3), "small" if abs(gg) < 0.5 else "medium" if abs(gg) < 0.8 else "large"])
    s4.tables.append(make_table("t_hedges", "Standardised mean differences (Hedges' g)", ["Comparison", "Hedges' g", f"{lvl}% CI low", f"{lvl}% CI high", "Reading"], hrows,
                                ["g is the mean difference divided by the pooled standard deviation of the two groups, with the small-sample correction of Hedges (1981); the interval uses a normal approximation. Reading: |g| below 0.5 small, below 0.8 medium, otherwise large (Cohen, 1988). These intervals are not adjusted for multiple comparisons."],
                                raw={"pairs": [dict(pair=[lab[i], lab[j]], g=h[0], lo=h[1], hi=h[2]) for (i, j), h in zip(pairs, hg)]}))
    ne = len(errs)
    s4.figures.append(Figure("f_pairs", f"Pairwise differences in mean {y} with {lvl}% Tukey-Kramer intervals",
                             {"ncols": 1, "panels": [{"xlabel": f"Difference in mean {y}", "ylabel": "", "ylab_off": 0, "yticks": [[float(ne - 1 - i), errs[i][0]] for i in range(ne)], "ylim": [-0.7, ne - 0.3], "ml": 86, "zero": False,
                                                      "layers": [{"type": "errh", "y": [float(ne - 1 - i) for i in range(ne)], "x": [e[1] for e in errs], "lo": [e[2] for e in errs], "hi": [e[3] for e in errs]}, {"type": "vline", "x": 0.0}]}]},
                             "An interval that excludes zero (dashed line) marks a pair whose means differ at the familywise level.", "Source: author's calculations.", 5.2, max(2.2, 0.42 * ne + 1.1)))
    sig = [(f"{lab[j]} and {lab[i]}", float(tk.pvalue[i, j]), float(means[j] - means[i])) for (i, j) in pairs if tk.pvalue[i, j] < alpha]
    nsig = len(sig)
    if nsig:
        ex = sorted(sig, key=lambda t: t[1])[0]
        tail = " The smallest adjusted p-value is for {pair} (difference = {d:.3f}, {p})." if nsig > 1 else " For {pair} the difference in means is {d:.3f} ({p})."
        s4.paragraphs.append(X.P("result", "Tukey-Kramer comparisons show that {m} of {t} pairs of groups differ at the {lv}% familywise level{lst}." + tail,
                                 m=nsig, t=len(pairs), lv=int(round(alpha * 100)), lst=(": " + X.join([s[0] for s in sig]) if nsig <= 4 else ""), pair=ex[0], d=ex[2], p=X.pv(ex[1])))
    else:
        s4.paragraphs.append(X.P("result", "No pair of groups differs at the {lv}% familywise level according to the Tukey-Kramer comparisons ({t} pairs).", lv=int(round(alpha * 100)), t=len(pairs)))
    gh_sig = sum(1 for r in gh if r[6] < alpha)
    if gh_sig != nsig:
        s4.paragraphs.append(X.P("warning", "The Games-Howell comparisons, which do not assume equal variances, find {a} significant pair(s), against {b} for Tukey-Kramer. Prefer the Games-Howell result when the variance tests in Step 2 reject.", a=gh_sig, b=nsig))
    rep.steps.append(s4)

    # ------------------------------------------------------------------ methods, limitations
    rep.methods.append(X.P("method", "Differences in {y} across the {k} levels of {g} were tested with a one-way analysis of variance on {N} observations. "
                                     "Normality of the residuals was examined with Shapiro-Wilk tests and equality of variances with the Brown-Forsythe, Levene and Bartlett tests. "
                                     "Because these assumptions may fail, Welch's ANOVA, the Brown-Forsythe F test and the Kruskal-Wallis test are reported beside the classical F test. "
                                     "Effect sizes are eta squared with a {lvl}% confidence interval from the noncentral F distribution, omega squared and Cohen's f. "
                                     "Pairwise differences use Tukey-Kramer and Games-Howell comparisons and Hedges' g. The significance level is {a}. "
                                     "Analyses were run with the Open Econometric Systems engine (version {ver}) using SciPy (version {sv}) in Python {py}.",
                           y=y, k=k, g=g, N=N, lvl=lvl, a=alpha, ver=__version__, sv=scipy.__version__, py=rep.meta["python"]))
    for s in rep.steps:
        for pg in s.paragraphs:
            if pg.kind == "warning":
                rep.limitations.append(pg)
    if not rep.limitations:
        rep.limitations.append(X.P("note", "None of the checks in this report signals a violated assumption at the {lv}% level; this does not prove the assumptions hold, and observations are assumed independent, which no test here can verify.", lv=int(round(alpha * 100))))
    return rep.finalize()
