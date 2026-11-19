"""Category A of the Statistics Division: analysis of variance.

One-way ANOVA, two-way ANOVA, repeated-measures ANOVA, ANCOVA, MANOVA and MANCOVA. Every method is written out with NumPy and
SciPy: sums of squares from nested least-squares fits, F tests from the central F distribution, effect-size intervals from the
noncentral F, and the four multivariate statistics from the eigenvalues of E^-1 H."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .core import OESError, as_list, chart, f, float_option, need, result, stat, table, test_row, tests_table, distinct
from .stat_core import (bonferroni, clean_frame, dummy_cols, effect_cols, eta2_ci, holm, levels, lstsq, mlm_fit, mv_statistics,
                        normality_tests, qq_points, rss_rank, term_tests, thin)


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _ci_label(alpha):
    return f"{int(round((1 - alpha) * 100))}%"


def _levene_rows(groups, label="Levene / Brown-Forsythe equality of variances"):
    rows = []
    try:
        w = stats.levene(*groups, center="median")
        rows.append([label + " (median-centred)", f(w.statistic, 4), f(w.pvalue, 4), "H0: equal variances"])
        b = stats.bartlett(*groups)
        rows.append(["Bartlett equality of variances", f(b.statistic, 4), f(b.pvalue, 4), "H0: equal variances; sensitive to non-normality"])
    except ValueError:
        pass
    return rows


def _interpret_eta(e):
    return "small" if e < 0.06 else "medium" if e < 0.14 else "large"


# ---------------------------------------------------------------------------------------------------------------
# A1. One-way ANOVA
# ---------------------------------------------------------------------------------------------------------------
def anova1(df, spec):
    y, g = need(spec, "y", "outcome variable"), need(spec, "group", "grouping variable")
    distinct(y, g)
    alpha = _alpha(spec)
    posthoc = spec.get("posthoc") or "tukey"
    d, dropped = clean_frame(df, numeric=[y], factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 2:
        raise OESError("The grouping variable needs at least two groups.")
    data = [d.loc[d[g] == l, y].to_numpy(float) for l in lv]
    n_i = np.array([len(v) for v in data])
    if (n_i < 2).any():
        raise OESError("Every group needs at least two observations. Groups with one observation: " + ", ".join(l for l, n in zip(lv, n_i) if n < 2))
    N = int(n_i.sum())
    means = np.array([v.mean() for v in data])
    vars_ = np.array([v.var(ddof=1) for v in data])
    grand = float(np.concatenate(data).mean())
    ssb = float(np.sum(n_i * (means - grand) ** 2))
    ssw = float(np.sum((n_i - 1) * vars_))
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
    cohen_f = float(np.sqrt(eta / (1 - eta))) if eta < 1 else np.inf
    lo, hi = eta2_ci(F, df1, df2, N, 1 - alpha)
    # robust tests
    w = n_i / vars_
    W = w.sum()
    mw = (w * means).sum() / W
    lam = ((1 - w / W) ** 2 / (n_i - 1)).sum()
    Fw = ((w * (means - mw) ** 2).sum() / df1) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2w = (k ** 2 - 1) / (3 * lam)
    pw = float(stats.f.sf(Fw, df1, df2w))
    c_i = 1 - n_i / N
    Fb = ssb / np.sum(c_i * vars_)
    dfb = (np.sum(c_i * vars_)) ** 2 / np.sum(c_i ** 2 * vars_ ** 2 / (n_i - 1))
    pb = float(stats.f.sf(Fb, df1, dfb))

    tcrit = stats.t.ppf(1 - alpha / 2, df2)
    desc = [[l, int(n), f(m), f(np.sqrt(v)), f(np.sqrt(msw / n)), f(m - tcrit * np.sqrt(msw / n)), f(m + tcrit * np.sqrt(msw / n)),
             f(vals.min()), f(np.median(vals)), f(vals.max())]
            for l, n, m, v, vals in zip(lv, n_i, means, vars_, data)]
    t_desc = table("Group descriptives", ["Group", "n", "Mean", "Std. dev.", "Std. err. (pooled)", f"{_ci_label(alpha)} CI low", f"{_ci_label(alpha)} CI high",
                                         "Min", "Median", "Max"], desc, "Confidence intervals use the pooled within-group variance and its degrees of freedom.")
    t_anova = table("Analysis of variance", ["Source", "Sum of squares", "df", "Mean square", "F", "p-value", "Eta squared", "Omega squared"],
                    [["Between groups", f(ssb), df1, f(msb), f(F, 4), f(p, 6), f(eta, 4), f(omega, 4)],
                     ["Within groups", f(ssw), df2, f(msw), None, None, None, None],
                     ["Total", f(sst), N - 1, None, None, None, None, None]])
    t_rob = table("Tests that do not assume equal variances", ["Test", "Statistic", "df1", "df2", "p-value"],
                  [["Welch ANOVA", f(Fw, 4), df1, f(df2w, 3), f(pw, 6)], ["Brown-Forsythe", f(Fb, 4), df1, f(dfb, 3), f(pb, 6)],
                   ["Kruskal-Wallis (rank-based)", f(stats.kruskal(*data).statistic, 4), df1, None, f(stats.kruskal(*data).pvalue, 6)]])
    t_eff = table("Effect sizes", ["Measure", "Estimate", f"{_ci_label(alpha)} CI low", f"{_ci_label(alpha)} CI high", "Reading"],
                  [["Eta squared", f(eta, 4), f(lo, 4), f(hi, 4), _interpret_eta(eta)], ["Omega squared (less biased)", f(omega, 4), None, None, _interpret_eta(omega)],
                   ["Epsilon squared", f(eps, 4), None, None, _interpret_eta(eps)], ["Cohen's f", f(cohen_f, 4), None, None, "small 0.10, medium 0.25, large 0.40"]],
                  "The interval for eta squared inverts the noncentral F distribution.")
    resid = np.concatenate([v - v.mean() for v in data])
    diag = _levene_rows(data) + normality_tests(resid)
    tabs = [t_desc, t_anova, t_rob, t_eff, tests_table(diag, "Assumption checks")]
    # post hoc
    if posthoc != "none" and k >= 2:
        rows = []
        pairs = list(itertools.combinations(range(k), 2))
        pv = []
        for i, j in pairs:
            diff = means[i] - means[j]
            if posthoc == "games_howell":
                se = np.sqrt(vars_[i] / n_i[i] + vars_[j] / n_i[j])
                dfij = (vars_[i] / n_i[i] + vars_[j] / n_i[j]) ** 2 / ((vars_[i] / n_i[i]) ** 2 / (n_i[i] - 1) + (vars_[j] / n_i[j]) ** 2 / (n_i[j] - 1))
                q = abs(diff) / se * np.sqrt(2)
                pij = float(stats.studentized_range.sf(q, k, dfij))
                crit = stats.studentized_range.ppf(1 - alpha, k, dfij) / np.sqrt(2)
            else:
                se = np.sqrt(msw / 2 * (1 / n_i[i] + 1 / n_i[j]))
                q = abs(diff) / se
                pij = float(stats.studentized_range.sf(q, k, df2))
                crit = stats.studentized_range.ppf(1 - alpha, k, df2) / np.sqrt(2)
                se = np.sqrt(msw * (1 / n_i[i] + 1 / n_i[j]))
            pv.append(pij)
            rows.append([f"{lv[i]} - {lv[j]}", f(diff), f(se), f(q, 4), f(pij, 6), f(diff - crit * se), f(diff + crit * se)])
        name = "Games-Howell" if posthoc == "games_howell" else "Tukey HSD (Tukey-Kramer when unbalanced)"
        tabs.append(table(f"Pairwise comparisons: {name}", ["Pair", "Mean difference", "Std. err.", "q", "Adjusted p", f"{_ci_label(alpha)} CI low", f"{_ci_label(alpha)} CI high"], rows,
                          "The p-values and intervals use the studentized range distribution, so they hold the family-wise error rate at the chosen level."))
    qz, qs = qq_points(resid)
    qx, qy = thin(list(qz), list(qs))
    charts = [chart("Group means with confidence intervals", lv, [("Mean", means)], "Group", y, kind="bar",
                    band=(means - tcrit * np.sqrt(msw / n_i), means + tcrit * np.sqrt(msw / n_i))),
              chart("Normal Q-Q plot of the residuals", qx, [("Standardised residual", qy)], "Theoretical normal quantile", "Residual", kind="scatter")]
    notes = [f"{k} groups, {N} observations" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations, normal residuals and equal variances; the Welch and Brown-Forsythe tests relax the last assumption."]
    if p < alpha:
        notes.append(f"The group means differ at the {alpha:g} level (F({df1}, {df2}) = {F:.3f}, p = {p:.4g}, eta squared = {eta:.3f}, a {_interpret_eta(eta)} effect).")
    else:
        notes.append(f"No difference in the group means is detected at the {alpha:g} level (F({df1}, {df2}) = {F:.3f}, p = {p:.4g}).")
    summary = [stat("F", F, 4), stat("p-value", p, 6), stat("Eta squared", eta, 4), stat("Groups", k), stat("Welch p", pw, 6)]
    return result("stat_anova1", "One-way ANOVA", N, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# A2. Two-way ANOVA
# ---------------------------------------------------------------------------------------------------------------
def anova2(df, spec):
    y, a, b = need(spec, "y", "outcome variable"), need(spec, "factor_a", "first factor"), need(spec, "factor_b", "second factor")
    distinct(y, a, b)
    alpha = _alpha(spec)
    kind = spec.get("ss_type") or "III"
    if kind not in ("II", "III"):
        raise OESError("The sum-of-squares type must be II or III.")
    inter = (spec.get("interaction") or "yes") != "no"
    d, dropped = clean_frame(df, numeric=[y], factors=[a, b])
    Xa, la = effect_cols(d[a])
    Xb, lb = effect_cols(d[b])
    if len(la) < 2 or len(lb) < 2:
        raise OESError("Each factor needs at least two levels.")
    cell = d.groupby([a, b])[y].agg(["count", "mean", "std"]).reset_index()
    if len(cell) < len(la) * len(lb) and inter:
        raise OESError("Some combinations of the two factors have no observations, so the interaction cannot be estimated. Remove the interaction or merge levels.")
    yv = d[y].to_numpy(float)
    N = len(yv)
    base = np.ones((N, 1))
    terms = {a: (frozenset([a]), Xa), b: (frozenset([b]), Xb)}
    if inter:
        Xab = np.hstack([Xa[:, [i]] * Xb[:, [j]] for i in range(Xa.shape[1]) for j in range(Xb.shape[1])])
        terms[f"{a} x {b}"] = (frozenset([a, b]), Xab)
    tt, (sse, dfe) = term_tests(yv, base, terms, kind)
    if dfe <= 0:
        raise OESError("There are no residual degrees of freedom; each cell needs more than one observation.")
    mse = sse / dfe
    ybar = yv.mean()
    sst = float(np.sum((yv - ybar) ** 2))
    rows = []
    for name, (ss, df_) in tt.items():
        Fv = (ss / df_) / mse if df_ > 0 else np.nan
        pv = float(stats.f.sf(Fv, df_, dfe)) if df_ > 0 else np.nan
        pe = ss / (ss + sse)
        pw = max(0.0, (ss - df_ * mse) / (ss + (N - df_) * mse))
        lo, hi = eta2_ci(Fv, df_, dfe, N, 1 - alpha) if df_ > 0 else (None, None)
        rows.append([name, f(ss), df_, f(ss / df_ if df_ else np.nan), f(Fv, 4), f(pv, 6), f(pe, 4), f(pw, 4), f(lo, 4), f(hi, 4)])
    rows.append(["Residuals", f(sse), dfe, f(mse), None, None, None, None, None, None])
    rows.append(["Total (corrected)", f(sst), N - 1, None, None, None, None, None, None, None])
    t_an = table(f"Two-way ANOVA, type {kind} sums of squares", ["Source", "Sum of squares", "df", "Mean square", "F", "p-value", "Partial eta sq.", "Partial omega sq.",
                                                                  f"Partial eta sq. {_ci_label(alpha)} CI low", "CI high"], rows,
                 "Sum-to-zero coding is used, so type III tests are invariant to the coding. With a balanced design types II and III coincide.")
    cells = [[str(r[a]), str(r[b]), int(r["count"]), f(r["mean"]), f(r["std"])] for _, r in cell.iterrows()]
    t_cell = table("Cell means", [a, b, "n", "Mean", "Std. dev."], cells)
    mm_a = [[l, f(np.mean([cell.loc[(cell[a] == l) & (cell[b] == m), "mean"].iloc[0] for m in lb if ((cell[a] == l) & (cell[b] == m)).any()]))] for l in la]
    mm_b = [[l, f(np.mean([cell.loc[(cell[b] == l) & (cell[a] == m), "mean"].iloc[0] for m in la if ((cell[b] == l) & (cell[a] == m)).any()]))] for l in lb]
    t_marg = table("Estimated marginal means (unweighted mean of the cell means)", ["Factor", "Level", "Marginal mean"],
                   [[a, l, v] for l, v in mm_a] + [[b, l, v] for l, v in mm_b])
    X = np.hstack([base] + [m for _, m in terms.values()])
    _, e, _, _ = lstsq(X, yv)
    groups = [d.loc[(d[a] == la_) & (d[b] == lb_), y].to_numpy(float) for la_ in la for lb_ in lb if ((d[a] == la_) & (d[b] == lb_)).any()]
    diag = _levene_rows([g for g in groups if len(g) > 1]) + normality_tests(e)
    cnt = cell["count"].to_numpy()
    if cnt.min() != cnt.max():
        diag.append(["Design", None, None, f"Unbalanced: cell sizes range from {int(cnt.min())} to {int(cnt.max())}; type III tests are used for the marginal hypotheses."])
    series = []
    for m in lb:
        series.append((f"{b} = {m}", [float(cell.loc[(cell[a] == l) & (cell[b] == m), "mean"].iloc[0]) if ((cell[a] == l) & (cell[b] == m)).any() else np.nan for l in la]))
    qz, qs = qq_points(e)
    qx, qy = thin(list(qz), list(qs))
    charts = [chart(f"Interaction plot: mean of {y}", la, series, a, y), chart("Normal Q-Q plot of the residuals", qx, [("Standardised residual", qy)], "Theoretical normal quantile", "Residual", kind="scatter")]
    notes = [f"{N} observations in {len(la)} x {len(lb)} cells" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations, normal residuals and equal variances across cells."]
    if inter:
        pint = float(stats.f.sf((tt[f'{a} x {b}'][0] / tt[f'{a} x {b}'][1]) / mse, tt[f'{a} x {b}'][1], dfe))
        notes.append("The interaction is " + ("significant" if pint < alpha else "not significant") + f" (p = {pint:.4g}); " +
                     ("interpret the main effects with care and look at the interaction plot." if pint < alpha else "the main effects can be read on their own."))
    summary = [stat("Residual df", dfe), stat("Residual std. error", np.sqrt(mse), 4), stat("R-squared", 1 - sse / sst, 4), stat("SS type", kind)]
    return result("stat_anova2", "Two-way ANOVA", N, summary, [t_an, t_cell, t_marg, tests_table(diag, "Assumption checks")], charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# A3. Repeated-measures ANOVA (one within-subject factor)
# ---------------------------------------------------------------------------------------------------------------
def _mauchly(S, n):
    k = S.shape[0]
    # orthonormal contrasts orthogonal to the constant
    H = np.eye(k) - np.ones((k, k)) / k
    U, s_, _ = np.linalg.svd(H)
    C = U[:, :k - 1].T
    Sc = C @ S @ C.T
    p = k - 1
    tr = np.trace(Sc)
    W = np.linalg.det(Sc) / (tr / p) ** p
    W = float(min(max(W, 1e-300), 1.0))
    df_ = n - 1
    fac = df_ - (2 * p ** 2 + p + 2) / (6 * p)
    chi = -fac * np.log(W)
    pdf = p * (p + 1) / 2 - 1
    eps_gg = float(tr ** 2 / (p * np.trace(Sc @ Sc)))
    eps_hf = float(min(1.0, (n * p * eps_gg - 2) / (p * (n - 1 - p * eps_gg))))
    return W, chi, pdf, float(stats.chi2.sf(chi, pdf)) if pdf > 0 else np.nan, eps_gg, max(eps_hf, 1.0 / p)


def rm_anova(df, spec):
    subj, w, y = need(spec, "subject", "subject identifier"), need(spec, "within", "within-subject factor"), need(spec, "y", "outcome variable")
    distinct(subj, w, y)
    alpha = _alpha(spec)
    corr = spec.get("correction") or "auto"
    d, dropped = clean_frame(df, numeric=[y], factors=[subj, w])
    lv = levels(d[w])
    k = len(lv)
    if k < 2:
        raise OESError("The within-subject factor needs at least two levels.")
    cnt = d.groupby([subj, w]).size()
    if (cnt > 1).any():
        raise OESError("Some subjects have more than one observation in the same condition. Average them first or add a column that makes each row unique.")
    wide = d.pivot(index=subj, columns=w, values=y)[lv]
    complete = wide.dropna()
    lost = len(wide) - len(complete)
    n = len(complete)
    if n < 3:
        raise OESError("At least three subjects with a value in every condition are needed.")
    Y = complete.to_numpy(float)
    gm = Y.mean()
    cm = Y.mean(0)
    sm = Y.mean(1)
    ss_c = float(n * np.sum((cm - gm) ** 2))
    ss_s = float(k * np.sum((sm - gm) ** 2))
    ss_t = float(np.sum((Y - gm) ** 2))
    ss_e = ss_t - ss_c - ss_s
    df_c, df_s, df_e = k - 1, n - 1, (k - 1) * (n - 1)
    if ss_e <= 0:
        raise OESError("There is no residual variation, so the F test is undefined.")
    ms_c, ms_e = ss_c / df_c, ss_e / df_e
    F = ms_c / ms_e
    p_unc = float(stats.f.sf(F, df_c, df_e))
    S = np.cov(Y, rowvar=False)
    if k > 2:
        Wm, chi, pdf_, pm, e_gg, e_hf = _mauchly(S, n)
    else:
        Wm, chi, pdf_, pm, e_gg, e_hf = 1.0, 0.0, 0, np.nan, 1.0, 1.0
    use = corr
    if corr == "auto":
        use = "none" if (k == 2 or (pm == pm and pm > alpha)) else ("gg" if e_gg < 0.75 else "hf")
    eps = {"none": 1.0, "gg": e_gg, "hf": e_hf}[use]
    p_gg = float(stats.f.sf(F, df_c * e_gg, df_e * e_gg))
    p_hf = float(stats.f.sf(F, df_c * e_hf, df_e * e_hf))
    p_used = float(stats.f.sf(F, df_c * eps, df_e * eps))
    pe = ss_c / (ss_c + ss_e)
    ge = ss_c / (ss_c + ss_s + ss_e)
    lo, hi = eta2_ci(F, df_c, df_e, n * k, 1 - alpha)
    t_an = table("Repeated-measures ANOVA (sphericity assumed)", ["Source", "Sum of squares", "df", "Mean square", "F", "p-value", "Partial eta sq.", "Generalised eta sq."],
                 [[f"Within factor: {w}", f(ss_c), df_c, f(ms_c), f(F, 4), f(p_unc, 6), f(pe, 4), f(ge, 4)], ["Error (within x subject)", f(ss_e), df_e, f(ms_e), None, None, None, None],
                  ["Subjects", f(ss_s), df_s, f(ss_s / df_s), None, None, None, None]])
    t_sph = table("Sphericity and corrected tests", ["Item", "Statistic", "df", "p-value", "Epsilon"],
                  [["Mauchly's test of sphericity", f(Wm, 4), pdf_ if k > 2 else None, f(pm, 4), None],
                   ["Greenhouse-Geisser corrected F", f(F, 4), f"{df_c * e_gg:.3f}, {df_e * e_gg:.3f}", f(p_gg, 6), f(e_gg, 4)],
                   ["Huynh-Feldt corrected F", f(F, 4), f"{df_c * e_hf:.3f}, {df_e * e_hf:.3f}", f(p_hf, 6), f(e_hf, 4)],
                   ["Used in the summary", f(F, 4), f"{df_c * eps:.3f}, {df_e * eps:.3f}", f(p_used, 6), f(eps, 4)]],
                  "Mauchly's test is applied to orthonormal contrasts of the repeated measures. With two conditions sphericity holds automatically.")
    tcrit = stats.t.ppf(1 - alpha / 2, df_e)
    se_c = np.sqrt(ms_e / n)
    t_desc = table("Condition descriptives", ["Condition", "n", "Mean", "Std. dev.", "Std. err. (error term)", f"{_ci_label(alpha)} CI low", f"{_ci_label(alpha)} CI high"],
                   [[l, n, f(m), f(Y[:, i].std(ddof=1)), f(se_c), f(m - tcrit * se_c), f(m + tcrit * se_c)] for i, (l, m) in enumerate(zip(lv, cm))],
                   "Intervals use the repeated-measures error term, which removes between-subject variation.")
    pairs, pv, rows = list(itertools.combinations(range(k), 2)), [], []
    for i, j in pairs:
        dd = Y[:, i] - Y[:, j]
        t, pp = stats.ttest_1samp(dd, 0.0)
        pv.append(pp)
    adj = holm(pv)
    for (i, j), pp, aj in zip(pairs, pv, adj):
        dd = Y[:, i] - Y[:, j]
        sd = dd.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(1 - alpha / 2, n - 1)
        rows.append([f"{lv[i]} - {lv[j]}", f(dd.mean()), f(sd), f(dd.mean() / sd, 4), n - 1, f(pp, 6), f(aj, 6), f(dd.mean() - tc * sd), f(dd.mean() + tc * sd), f(dd.mean() / dd.std(ddof=1), 4)])
    t_ph = table("Pairwise paired t tests (Holm adjustment)", ["Pair", "Mean difference", "Std. err.", "t", "df", "p-value", "Holm p", f"{_ci_label(alpha)} CI low", "CI high", "Cohen's dz"], rows)
    diag = []
    resid = (Y - cm[None, :] - sm[:, None] + gm).ravel()
    diag += normality_tests(resid)
    diag.append(["Friedman rank test (distribution-free check)", f(stats.friedmanchisquare(*[Y[:, i] for i in range(k)]).statistic, 4) if k >= 3 else None,
                 f(stats.friedmanchisquare(*[Y[:, i] for i in range(k)]).pvalue, 6) if k >= 3 else None, "H0: no difference between conditions"])
    qz, qs = qq_points(resid)
    qx, qy = thin(list(qz), list(qs))
    charts = [chart("Condition means with confidence intervals", lv, [("Mean", cm)], w, y, kind="bar", band=(cm - tcrit * se_c, cm + tcrit * se_c)),
              chart("Subject profiles (first 30 subjects)", lv, [(str(s), complete.iloc[i].to_numpy(float)) for i, s in enumerate(complete.index[:30])], w, y),
              chart("Normal Q-Q plot of the residuals", qx, [("Standardised residual", qy)], "Theoretical normal quantile", "Residual", kind="scatter")]
    notes = [f"{n} subjects measured in {k} conditions" + (f"; {lost} subject(s) with a missing condition were dropped." if lost else ".")]
    notes.append("Assumes normal residuals and sphericity of the covariance of the repeated measures; the Greenhouse-Geisser and Huynh-Feldt corrections adjust the degrees of freedom when sphericity fails.")
    summary = [stat("F", F, 4), stat("p-value (used)", p_used, 6), stat("Partial eta squared", pe, 4), stat("Greenhouse-Geisser epsilon", e_gg, 4), stat("Mauchly p", pm, 4) if k > 2 else stat("Conditions", k)]
    return result("stat_rm_anova", "Repeated-measures ANOVA", n * k, summary, [t_desc, t_an, t_sph, t_ph, tests_table(diag, "Assumption checks")], charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# A4. ANCOVA
# ---------------------------------------------------------------------------------------------------------------
def ancova(df, spec):
    y, g = need(spec, "y", "outcome variable"), need(spec, "group", "grouping variable")
    cov = as_list(need(spec, "covariates", "covariate(s)"))
    distinct(y, g, cov)
    alpha = _alpha(spec)
    kind = spec.get("ss_type") or "III"
    d, dropped = clean_frame(df, numeric=[y] + cov, factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 2:
        raise OESError("The grouping variable needs at least two groups.")
    yv = d[y].to_numpy(float)
    N = len(yv)
    Xg_e, _ = effect_cols(d[g], lv)
    Xc = d[cov].to_numpy(float)
    Xc_c = Xc - Xc.mean(0)
    base = np.ones((N, 1))
    terms = {g: (frozenset([g]), Xg_e)}
    for j, c in enumerate(cov):
        terms[c] = (frozenset([c]), Xc_c[:, [j]])
    tt, (sse, dfe) = term_tests(yv, base, terms, kind)
    if dfe <= 0:
        raise OESError("There are no residual degrees of freedom.")
    mse = sse / dfe
    sst = float(np.sum((yv - yv.mean()) ** 2))
    rows = []
    for name, (ss, df_) in tt.items():
        Fv = (ss / df_) / mse
        pv = float(stats.f.sf(Fv, df_, dfe))
        lo, hi = eta2_ci(Fv, df_, dfe, N, 1 - alpha)
        rows.append([name + (" (covariate)" if name in cov else ""), f(ss), df_, f(ss / df_), f(Fv, 4), f(pv, 6), f(ss / (ss + sse), 4), f(max(0.0, (ss - df_ * mse) / (ss + (N - df_) * mse)), 4), f(lo, 4), f(hi, 4)])
    rows.append(["Residuals", f(sse), dfe, f(mse), None, None, None, None, None, None])
    t_an = table(f"ANCOVA, type {kind} sums of squares", ["Source", "Sum of squares", "df", "Mean square", "F", "p-value", "Partial eta sq.", "Partial omega sq.", "Partial eta sq. CI low", "CI high"], rows)
    # adjusted means from the treatment-coded model
    Xd, _ = dummy_cols(d[g], lv)
    X = np.hstack([base, Xd, Xc])
    beta, e, rank, XtXp = lstsq(X, yv)
    cmean = Xc.mean(0)
    tcrit = stats.t.ppf(1 - alpha / 2, dfe)
    adj_rows, adj_vals, adj_se, x0s = [], [], [], []
    for j, l in enumerate(lv):
        x0 = np.concatenate([[1.0], (np.arange(1, k) == j).astype(float), cmean])
        m = float(x0 @ beta)
        se = float(np.sqrt(mse * x0 @ XtXp @ x0))
        raw = d.loc[d[g] == l, y]
        adj_rows.append([l, int((d[g] == l).sum()), f(raw.mean()), f(m), f(se), f(m - tcrit * se), f(m + tcrit * se)])
        adj_vals.append(m)
        adj_se.append(se)
        x0s.append(x0)
    t_adj = table("Unadjusted and covariate-adjusted group means", ["Group", "n", "Unadjusted mean", "Adjusted mean", "Std. err.", f"{_ci_label(alpha)} CI low", f"{_ci_label(alpha)} CI high"], adj_rows,
                  "Adjusted means are predictions at the mean of every covariate: " + ", ".join(f"{c} = {m:.4g}" for c, m in zip(cov, cmean)) + ".")
    pairs = list(itertools.combinations(range(k), 2))
    pv, prow = [], []
    for i, j in pairs:
        c = x0s[i] - x0s[j]
        est = float(c @ beta)
        se = float(np.sqrt(mse * c @ XtXp @ c))
        t = est / se
        pv.append(2 * stats.t.sf(abs(t), dfe))
        prow.append((i, j, est, se, t))
    adjp = holm(pv)
    t_pw = table("Pairwise differences of adjusted means (Holm adjustment)", ["Pair", "Difference", "Std. err.", "t", "df", "p-value", "Holm p", f"{_ci_label(alpha)} CI low", "CI high"],
                 [[f"{lv[i]} - {lv[j]}", f(est), f(se), f(t, 4), dfe, f(p_, 6), f(a_, 6), f(est - tcrit * se), f(est + tcrit * se)] for (i, j, est, se, t), p_, a_ in zip(prow, pv, adjp)])
    names = ["Intercept"] + [f"{g} = {l} (vs {lv[0]})" for l in lv[1:]] + cov
    se_b = np.sqrt(np.diag(XtXp) * mse)
    t_b = table("Regression coefficients (treatment coding)", ["Term", "Coef.", "Std. err.", "t", "p-value", f"{_ci_label(alpha)} CI low", "CI high"],
                [[n_, f(b_), f(s_), f(b_ / s_, 4), f(2 * stats.t.sf(abs(b_ / s_), dfe), 6), f(b_ - tcrit * s_), f(b_ + tcrit * s_)] for n_, b_, s_ in zip(names, beta, se_b)])
    # homogeneity of regression slopes
    diag = []
    Xint = np.hstack([Xg_e[:, [i]] * Xc_c[:, [j]] for j in range(len(cov)) for i in range(k - 1)])
    rss_r, r_r = rss_rank(np.hstack([base, Xg_e, Xc_c]), yv)
    rss_f, r_f = rss_rank(np.hstack([base, Xg_e, Xc_c, Xint]), yv)
    dfi, dff = r_f - r_r, N - r_f
    if dfi > 0 and dff > 0:
        Fi = ((rss_r - rss_f) / dfi) / (rss_f / dff)
        diag.append(["Homogeneity of regression slopes (group x covariate)", f(Fi, 4), f(float(stats.f.sf(Fi, dfi, dff)), 4), "H0: parallel slopes; ANCOVA needs this to hold"])
    groups = [e[(d[g] == l).to_numpy()] for l in lv]
    diag += _levene_rows([x for x in groups if len(x) > 1], "Levene test of the residuals") + normality_tests(e)
    if len(cov) >= 1:
        # covariate balance across groups
        for j, c in enumerate(cov):
            Fb = stats.f_oneway(*[d.loc[d[g] == l, c].to_numpy(float) for l in lv])
            diag.append([f"Covariate balance: {c} across groups", f(Fb.statistic, 4), f(Fb.pvalue, 4), "H0: equal covariate means (large imbalance makes adjustment model dependent)"])
    qz, qs = qq_points(e)
    qx, qy = thin(list(qz), list(qs))
    c0 = cov[0]
    charts = [chart("Unadjusted and adjusted group means", lv, [("Unadjusted", [float(d.loc[d[g] == l, y].mean()) for l in lv]), ("Adjusted", adj_vals)], g, y, kind="bar"),
              chart(f"Outcome against {c0}", list(d[c0])[:400], [(y, list(yv)[:400])], c0, y, kind="scatter"),
              chart("Normal Q-Q plot of the residuals", qx, [("Standardised residual", qy)], "Theoretical normal quantile", "Residual", kind="scatter")]
    notes = [f"{N} observations, {k} groups, {len(cov)} covariate(s)" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations, normal residuals, equal variances, a linear covariate effect and parallel slopes across groups."]
    summary = [stat("Group F", tt[g][0] / tt[g][1] / mse, 4), stat("Group p-value", float(stats.f.sf(tt[g][0] / tt[g][1] / mse, tt[g][1], dfe)), 6), stat("R-squared", 1 - sse / sst, 4), stat("Residual std. error", np.sqrt(mse), 4)]
    return result("stat_ancova", "Analysis of covariance (ANCOVA)", N, summary, [t_an, t_adj, t_pw, t_b, tests_table(diag, "Assumption checks")], charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# A5 and A6. MANOVA and MANCOVA
# ---------------------------------------------------------------------------------------------------------------
def _box_m(groups):
    g = len(groups)
    p = groups[0].shape[1]
    ni = np.array([len(x) for x in groups])
    N = ni.sum()
    Ss = [np.cov(x, rowvar=False) for x in groups]
    Sp = sum((n - 1) * S for n, S in zip(ni, Ss)) / (N - g)
    ld = lambda S: np.linalg.slogdet(S)[1]
    M = (N - g) * ld(Sp) - sum((n - 1) * ld(S) for n, S in zip(ni, Ss))
    c = (np.sum(1.0 / (ni - 1)) - 1.0 / (N - g)) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    chi = M * (1 - c)
    dfc = p * (p + 1) * (g - 1) / 2
    return float(chi), dfc, float(stats.chi2.sf(chi, dfc))


def _mardia(R):
    n, p = R.shape
    Rc = R - R.mean(0)
    S = Rc.T @ Rc / n
    Si = np.linalg.pinv(S)
    D = Rc @ Si @ Rc.T
    b1 = float((D ** 3).sum() / n ** 2)
    b2 = float(np.mean(np.diag(D) ** 2))
    z1 = n * b1 / 6
    df1 = p * (p + 1) * (p + 2) / 6
    z2 = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return z1, df1, float(stats.chi2.sf(z1, df1)), b2, float(z2), float(2 * stats.norm.sf(abs(z2)))


def _mv_table(name, H, E, q, dfe, N, alpha):
    out, ev = mv_statistics(H, E, q, dfe)
    s = min(E.shape[0], q)
    rows = []
    for lab, (val, Fv, d1, d2) in out.items():
        pv = float(stats.f.sf(Fv, d1, d2)) if np.isfinite(Fv) else 0.0
        rows.append([lab, f(val, 5), f(Fv, 4), f(d1, 3), f(d2, 3), f(pv, 6)])
    lam = out["Wilks' lambda"][0]
    pe = 1 - lam ** (1 / s) if lam > 0 else 1.0
    return rows, float(pe), ev, out


def _univariate_follow(Y, ynames, X_full, X_red, alpha):
    rows, pv = [], []
    for j, nm in enumerate(ynames):
        rss_f_, r_f = rss_rank(X_full, Y[:, j])
        rss_r_, r_r = rss_rank(X_red, Y[:, j])
        dfn, dfd = r_f - r_r, len(Y) - r_f
        Fv = ((rss_r_ - rss_f_) / dfn) / (rss_f_ / dfd)
        pv.append(float(stats.f.sf(Fv, dfn, dfd)))
        rows.append([nm, f(rss_r_ - rss_f_), dfn, f(rss_f_ / dfd), f(Fv, 4), pv[-1], f((rss_r_ - rss_f_) / rss_r_, 4)])
    adj = holm(pv)
    return [r[:5] + [f(r[5], 6), f(a, 6), r[6]] for r, a in zip(rows, adj)]


def manova(df, spec):
    ys, g = as_list(need(spec, "y_vars", "response variables")), need(spec, "group", "grouping variable")
    if len(ys) < 2:
        raise OESError("MANOVA needs at least two response variables.")
    distinct(ys, g)
    alpha = _alpha(spec)
    d, dropped = clean_frame(df, numeric=ys, factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 2:
        raise OESError("The grouping variable needs at least two groups.")
    Y = d[ys].to_numpy(float)
    N, p = Y.shape
    n_i = np.array([(d[g] == l).sum() for l in lv])
    if N - k <= p:
        raise OESError(f"Too few observations: the error degrees of freedom ({N - k}) must exceed the number of responses ({p}).")
    if (n_i <= p).any():
        raise OESError("Every group needs more observations than there are response variables.")
    X = np.hstack([np.ones((N, 1)), dummy_cols(d[g], lv)[0]])
    B, E, rank, XtXp = mlm_fit(Y, X)
    gm = Y.mean(0)
    Hm = sum(n * np.outer(Y[(d[g] == l).to_numpy()].mean(0) - gm, Y[(d[g] == l).to_numpy()].mean(0) - gm) for n, l in zip(n_i, lv))
    dfe = N - k
    if abs(np.linalg.det(E)) < 1e-300:
        raise OESError("The error SSP matrix is singular: some response variables are linear combinations of the others.")
    rows, pe, ev, out = _mv_table(g, Hm, E, k - 1, dfe, N, alpha)
    t_mv = table(f"Multivariate tests for {g}", ["Statistic", "Value", "Approx. F", "Num df", "Den df", "p-value"], rows,
                 "Wilks uses Rao's F approximation; Pillai, Hotelling-Lawley and Roy follow the SAS and R conventions. Roy's F is an upper bound on the true p-value.")
    t_eff = table("Multivariate effect size", ["Measure", "Estimate"], [["Partial eta squared (1 - Wilks^(1/s))", f(pe, 4)], ["Eigenvalues of E^-1 H"] + [", ".join(f"{v:.4f}" for v in ev)]])
    uni = _univariate_follow(Y, ys, X, np.ones((N, 1)), alpha)
    t_uni = table("Follow-up univariate ANOVAs (Holm adjustment across responses)", ["Response", "SS between", "df", "MS error", "F", "p-value", "Holm p", "Eta squared"], uni)
    desc = [[l, int(n)] + [f(Y[(d[g] == l).to_numpy(), j].mean()) for j in range(p)] for l, n in zip(lv, n_i)]
    t_desc = table("Group means", ["Group", "n"] + ys, desc)
    diag = []
    R = Y - np.vstack([Y[(d[g] == gi).to_numpy()].mean(0) for gi in d[g]])
    try:
        bm = _box_m([Y[(d[g] == l).to_numpy()] for l in lv])
        diag.append(["Box's M equality of covariance matrices", f(bm[0], 4), f(bm[2], 4), "H0: equal covariance matrices (sensitive to non-normality; chi-square approximation)"])
    except Exception:
        pass
    z1, dfz, pz1, b2, z2, pz2 = _mardia(R)
    diag.append(["Mardia multivariate skewness", f(z1, 4), f(pz1, 4), "H0: multivariate normal residuals"])
    diag.append(["Mardia multivariate kurtosis", f(z2, 4), f(pz2, 4), "H0: multivariate normal residuals"])
    md = np.einsum("ij,jk,ik->i", R, np.linalg.pinv(E / dfe), R)
    out_n = int((stats.chi2.sf(md, p) < 0.001).sum())
    diag.append(["Multivariate outliers (Mahalanobis p < 0.001)", out_n, None, f"{out_n} observation(s) flagged"])
    corr = np.corrcoef(R, rowvar=False)
    t_corr = table("Pooled within-group correlations of the responses", ["Variable"] + ys, [[ys[i]] + [f(corr[i, j], 3) for j in range(p)] for i in range(p)])
    charts = [chart("Group means of each response", lv, [(nm, [float(Y[(d[g] == l).to_numpy(), j].mean()) for l in lv]) for j, nm in enumerate(ys)], g, "Mean")]
    notes = [f"{N} observations, {k} groups, {p} responses" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations, multivariate normal residuals and equal covariance matrices across groups."]
    w = out["Wilks' lambda"]
    notes.append(f"Wilks' lambda = {w[0]:.4f}, F({w[2]:.0f}, {w[3]:.1f}) = {w[1]:.3f}, p = {float(stats.f.sf(w[1], w[2], w[3])):.4g}.")
    summary = [stat("Wilks' lambda", w[0], 4), stat("Approx. F", w[1], 4), stat("p-value", float(stats.f.sf(w[1], w[2], w[3])), 6), stat("Pillai's trace", out["Pillai's trace"][0], 4), stat("Partial eta squared", pe, 4)]
    return result("stat_manova", "Multivariate ANOVA (MANOVA)", N, summary, [t_mv, t_eff, t_uni, t_desc, tests_table(diag, "Assumption checks"), t_corr], charts, notes)


def mancova(df, spec):
    ys, g = as_list(need(spec, "y_vars", "response variables")), need(spec, "group", "grouping variable")
    cov = as_list(need(spec, "covariates", "covariate(s)"))
    if len(ys) < 2:
        raise OESError("MANCOVA needs at least two response variables.")
    distinct(ys, g, cov)
    alpha = _alpha(spec)
    d, dropped = clean_frame(df, numeric=ys + cov, factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 2:
        raise OESError("The grouping variable needs at least two groups.")
    Y = d[ys].to_numpy(float)
    C = d[cov].to_numpy(float)
    N, p = Y.shape
    q = len(cov)
    if N - k - q <= p:
        raise OESError(f"Too few observations for {p} responses, {k} groups and {q} covariate(s).")
    Xg = dummy_cols(d[g], lv)[0]
    Xfull = np.hstack([np.ones((N, 1)), Xg, C])
    B, E, rank, XtXp = mlm_fit(Y, Xfull)
    dfe = N - rank
    tabs = []
    terms = [(g, np.hstack([np.ones((N, 1)), C])), ] + [(c, np.hstack([np.ones((N, 1)), Xg, np.delete(C, j, axis=1)])) for j, c in enumerate(cov)]
    summ = {}
    for name, Xr in terms:
        _, Er, rr, _ = mlm_fit(Y, Xr)
        H = Er - E
        qq = rank - rr
        rows, pe, ev, out = _mv_table(name, H, E, qq, dfe, N, alpha)
        tabs.append(table(f"Multivariate tests for {name}" + (" (covariate)" if name in cov else " (adjusted for the covariates)"), ["Statistic", "Value", "Approx. F", "Num df", "Den df", "p-value"], rows))
        summ[name] = (out["Wilks' lambda"], pe)
    uni = _univariate_follow(Y, ys, Xfull, np.hstack([np.ones((N, 1)), C]), alpha)
    tabs.append(table(f"Follow-up univariate ANCOVAs for {g} (Holm adjustment across responses)", ["Response", "SS group", "df", "MS error", "F", "p-value", "Holm p", "Partial eta sq."], uni))
    cm = C.mean(0)
    adj = []
    for j, l in enumerate(lv):
        x0 = np.concatenate([[1.0], (np.arange(1, k) == j).astype(float), cm])
        adj.append(x0 @ B)
    adj = np.array(adj)
    raw = np.array([Y[(d[g] == l).to_numpy()].mean(0) for l in lv])
    tabs.append(table("Adjusted group means (covariates at their means)", ["Group", "Type"] + ys, [[l, "adjusted"] + [f(v) for v in adj[i]] for i, l in enumerate(lv)] + [[l, "unadjusted"] + [f(v) for v in raw[i]] for i, l in enumerate(lv)]))
    coefrows = []
    for j, c in enumerate(cov):
        coefrows.append([c] + [f(B[1 + (k - 1) + j, i], 5) for i in range(p)])
    tabs.append(table("Covariate slopes for each response", ["Covariate"] + ys, coefrows))
    # homogeneity of regression (group x covariate interaction)
    Xint = np.hstack([Xg[:, [i]] * (C[:, [j]] - cm[j]) for j in range(q) for i in range(k - 1)])
    Xh = np.hstack([Xfull[:, :1 + (k - 1)], C - cm, Xint])
    _, Eh, rh, _ = mlm_fit(Y, Xh)
    diag = []
    dfh = rh - rank
    if dfh > 0 and N - rh > p:
        out_h, _ = mv_statistics(E - Eh, Eh, dfh, N - rh)
        w_ = out_h["Wilks' lambda"]
        diag.append(["Homogeneity of regression (group x covariate), Wilks", f(w_[0], 4), f(float(stats.f.sf(w_[1], w_[2], w_[3])), 4), "H0: parallel slopes in every response"])
    Rres = Y - Xfull @ B
    z1, dfz, pz1, b2, z2, pz2 = _mardia(Rres)
    diag.append(["Mardia multivariate skewness", f(z1, 4), f(pz1, 4), "H0: multivariate normal residuals"])
    diag.append(["Mardia multivariate kurtosis", f(z2, 4), f(pz2, 4), "H0: multivariate normal residuals"])
    try:
        bm = _box_m([Rres[(d[g] == l).to_numpy()] for l in lv])
        diag.append(["Box's M on the residual covariances", f(bm[0], 4), f(bm[2], 4), "H0: equal residual covariance matrices"])
    except Exception:
        pass
    tabs.append(tests_table(diag, "Assumption checks"))
    charts = [chart("Unadjusted and adjusted means by response", ys, [(f"{l} adjusted", list(adj[i])) for i, l in enumerate(lv)] + [(f"{l} unadjusted", list(raw[i])) for i, l in enumerate(lv)], "Response", "Mean")]
    w = summ[g][0]
    notes = [f"{N} observations, {k} groups, {p} responses and {q} covariate(s)" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations, multivariate normal residuals, equal residual covariance matrices and parallel regression slopes across groups."]
    summary = [stat("Wilks' lambda (group)", w[0], 4), stat("Approx. F", w[1], 4), stat("p-value", float(stats.f.sf(w[1], w[2], w[3])), 6), stat("Partial eta squared", summ[g][1], 4)]
    return result("stat_mancova", "Multivariate ANCOVA (MANCOVA)", N, summary, tabs, charts, notes)
