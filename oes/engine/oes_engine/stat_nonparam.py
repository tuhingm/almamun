"""Category B of the Statistics Division: rank-based (nonparametric) methods.

Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman and Kendall's tau. The exact null distributions are computed by
counting (generating polynomials with integer arithmetic), the large-sample versions use the tie-corrected variances, and the
location estimates and intervals (Hodges-Lehmann) come from the pairwise differences and Walsh averages."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table, tests_table
from .stat_core import benjamini_hochberg, bonferroni, clean_frame, holm, levels

EXACT_MAX = 50


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _adjust(p, how):
    return {"holm": holm, "bonferroni": bonferroni, "bh": benjamini_hochberg, "none": lambda x: np.asarray(x, float)}[how](p)


# ---------------------------------------------------------------------------------------------------------------
# exact null distributions
# ---------------------------------------------------------------------------------------------------------------
def u_distribution(n1, n2):
    """Probability mass function of the Mann-Whitney U statistic under H0 with no ties: coefficients of the Gaussian binomial."""
    poly = [1]
    for i in range(1, n1 + 1):
        # multiply by (1 - q^(n2+i))
        out = poly + [0] * (n2 + i)
        for k, c in enumerate(poly):
            out[k + n2 + i] -= c
        # divide by (1 - q^i): r[k] = out[k] + r[k - i]
        for k in range(i, len(out)):
            out[k] += out[k - i]
        poly = out
    poly = poly[:n1 * n2 + 1]
    tot = sum(poly)
    return np.array([c / tot for c in poly], float)


def signrank_distribution(n):
    """Probability mass function of the Wilcoxon signed-rank statistic V = sum of the ranks of the positive differences, no ties."""
    poly = [1] + [0] * (n * (n + 1) // 2)
    top = 0
    for r in range(1, n + 1):
        top += r
        for k in range(top, r - 1, -1):
            poly[k] += poly[k - r]
    tot = 2 ** n
    return np.array([c / tot for c in poly], float)


def kendall_distribution(n):
    """Probability mass function of the number of discordant pairs under independence with no ties (Mahonian numbers)."""
    poly = [1]
    for i in range(2, n + 1):
        out = [0] * (len(poly) + i - 1)
        for k, c in enumerate(poly):
            for j in range(i):
                out[k + j] += c
        poly = out
    tot = sum(poly)
    return np.array([c / tot for c in poly], float)


def _tail(pmf, value, alternative):
    """Exact p-value for a statistic with the given pmf on 0..m (integer or half-integer values are rounded to the lattice)."""
    cdf = np.cumsum(pmf)
    v = int(round(value))
    lower = float(cdf[min(max(v, 0), len(pmf) - 1)])
    upper = float(1.0 - (cdf[v - 1] if v >= 1 else 0.0)) if v <= len(pmf) - 1 else 0.0
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2 * min(lower, upper))


# ---------------------------------------------------------------------------------------------------------------
# B1. Mann-Whitney U
# ---------------------------------------------------------------------------------------------------------------
def mann_whitney(df, spec):
    y, g = need(spec, "y", "outcome variable"), need(spec, "group", "grouping variable")
    distinct(y, g)
    alpha = _alpha(spec)
    alt = spec.get("alternative") or "two-sided"
    method = spec.get("method") or "auto"
    cont = (spec.get("continuity") or "yes") == "yes"
    d, dropped = clean_frame(df, numeric=[y], factors=[g])
    lv = levels(d[g])
    if len(lv) != 2:
        raise OESError(f"The grouping variable must have exactly two groups; it has {len(lv)} ({', '.join(lv[:6])}). Use Kruskal-Wallis for more groups.")
    a = d.loc[d[g] == lv[0], y].to_numpy(float)
    b = d.loc[d[g] == lv[1], y].to_numpy(float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise OESError("Each group needs at least two observations.")
    N = n1 + n2
    allv = np.concatenate([a, b])
    ranks = stats.rankdata(allv)
    R1 = float(ranks[:n1].sum())
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    _, cnt = np.unique(allv, return_counts=True)
    ties = bool((cnt > 1).any())
    mu = n1 * n2 / 2
    tie_term = float(np.sum(cnt ** 3 - cnt)) / (N * (N - 1))
    sigma = np.sqrt(n1 * n2 / 12 * ((N + 1) - tie_term))
    use_exact = method == "exact" or (method == "auto" and not ties and max(n1, n2) <= EXACT_MAX)
    if use_exact and ties:
        raise OESError("The exact test needs data without ties. Use the asymptotic method.")
    if use_exact:
        p = _tail(u_distribution(n1, n2), U1, alt)
        z = (U1 - mu) / sigma
        how = "Exact null distribution of U"
    else:
        diff = U1 - mu
        c = 0.5 if cont else 0.0
        if alt == "two-sided":
            z = (abs(diff) - c) / sigma * (1 if diff >= 0 else -1) if abs(diff) > c else 0.0
            p = float(2 * stats.norm.sf(abs(z)))
        elif alt == "greater":
            z = (diff - c) / sigma
            p = float(stats.norm.sf(z))
        else:
            z = (diff + c) / sigma
            p = float(stats.norm.cdf(z))
        how = "Normal approximation with tie correction" + (" and continuity correction" if cont else "")
    p = min(1.0, p)
    rb = 2 * U1 / (n1 * n2) - 1
    cles = U1 / (n1 * n2)
    D = np.sort((a[:, None] - b[None, :]).ravel())
    hl = float(np.median(D))
    zc = stats.norm.ppf(1 - alpha / 2)
    C = int(np.floor(n1 * n2 / 2 - zc * np.sqrt(n1 * n2 * (N + 1) / 12.0)))
    C = max(C, 0)
    lo, hi = float(D[C]), float(D[n1 * n2 - 1 - C])
    # confidence interval of the rank-biserial correlation (Fisher transform with the large-sample variance of U)
    se_r = 2 * sigma / (n1 * n2)
    tabs = [table("Group descriptives", ["Group", "n", "Mean", "Median", "Mean rank", "Rank sum"],
                  [[lv[0], n1, f(a.mean()), f(np.median(a)), f(R1 / n1), f(R1)], [lv[1], n2, f(b.mean()), f(np.median(b)), f((ranks.sum() - R1) / n2), f(ranks.sum() - R1)]]),
            table("Mann-Whitney U test", ["Statistic", "Value"], [["U (first group)", f(U1)], ["U (second group)", f(U2)], ["Smaller U", f(min(U1, U2))], ["z", f(z, 4)], ["p-value", f(p, 6)], ["Alternative", alt], ["Method", how]]),
            table("Effect sizes and location shift", ["Measure", "Estimate", f"{int(round((1 - alpha) * 100))}% CI low", "CI high", "Reading"],
                  [["Rank-biserial correlation", f(rb, 4), f(max(-1.0, rb - zc * se_r), 4), f(min(1.0, rb + zc * se_r), 4), "positive: the first group tends to be larger"],
                   ["Probability of superiority P(X > Y) + 0.5 P(X = Y)", f(cles, 4), None, None, "0.5 means no tendency"],
                   [f"Hodges-Lehmann shift ({lv[0]} - {lv[1]})", f(hl), f(lo), f(hi), "median of all pairwise differences; interval from the order statistics of the differences"]],
                  "The Hodges-Lehmann interval uses the normal approximation of the rank-sum distribution, so it is approximate for small samples.")]
    diag = [["Ties in the pooled data", int((cnt > 1).sum()), None, "the exact test is skipped when ties are present" if ties else "no ties"]]
    a_sw = stats.shapiro(a) if 3 <= n1 <= 5000 else None
    b_sw = stats.shapiro(b) if 3 <= n2 <= 5000 else None
    if a_sw and b_sw:
        diag.append([f"Shapiro-Wilk normality, {lv[0]}", f(a_sw.statistic, 4), f(a_sw.pvalue, 4), "if normality holds the t test is more powerful"])
        diag.append([f"Shapiro-Wilk normality, {lv[1]}", f(b_sw.statistic, 4), f(b_sw.pvalue, 4), ""])
    lev = stats.levene(a, b, center="median")
    diag.append(["Levene equality of spreads", f(lev.statistic, 4), f(lev.pvalue, 4), "the test reads as a difference in location only if the shapes are alike"])
    tabs.append(tests_table(diag, "Assumption checks"))
    s_a, s_b = np.sort(a), np.sort(b)
    pr = (np.arange(1, 101) - 0.5) / 100
    charts = [chart("Group distributions (empirical quantiles)", [f"{q:.2f}" for q in pr[::5]], [(lv[0], np.quantile(a, pr[::5])), (lv[1], np.quantile(b, pr[::5]))], "Quantile", y),
              chart("Mean rank by group", lv, [("Mean rank", [R1 / n1, (ranks.sum() - R1) / n2])], "Group", "Mean rank", kind="bar")]
    notes = [f"{n1} and {n2} observations" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations from two groups and an ordinal or continuous outcome. The test compares the whole distributions; it reads as a test of medians only if the two shapes match."]
    summary = [stat("U", U1, 2), stat("z", z, 4), stat("p-value", p, 6), stat("Rank-biserial r", rb, 4), stat("Hodges-Lehmann shift", hl, 4)]
    return result("stat_mann_whitney", "Mann-Whitney U test", N, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# B2. Wilcoxon signed-rank
# ---------------------------------------------------------------------------------------------------------------
def wilcoxon(df, spec):
    x1 = need(spec, "x1", "first measurement")
    x2 = spec.get("x2") or None
    alpha = _alpha(spec)
    alt = spec.get("alternative") or "two-sided"
    method = spec.get("method") or "auto"
    zero = spec.get("zero_method") or "wilcox"
    cont = (spec.get("continuity") or "yes") == "yes"
    mu0 = float_option(spec, "mu0", 0.0, "Hypothesised median")
    cols = [x1] + ([x2] if x2 else [])
    distinct(*cols)
    d, dropped = clean_frame(df, numeric=cols)
    diff = (d[x1] - d[x2]).to_numpy(float) if x2 else d[x1].to_numpy(float) - mu0
    n_all = len(diff)
    nz = diff[diff != 0]
    n_zero = n_all - len(nz)
    if len(nz) < 2:
        raise OESError("Fewer than two non-zero differences remain, so the signed-rank test cannot be computed.")
    if zero == "wilcox":
        used = nz
        rk = stats.rankdata(np.abs(used))
        pos, neg = float(rk[used > 0].sum()), float(rk[used < 0].sum())
        count, mean_v = len(used), len(used) * (len(used) + 1) / 4
        var_v = len(used) * (len(used) + 1) * (2 * len(used) + 1) / 24
        tie_arr = np.abs(used)
    else:
        rk_all = stats.rankdata(np.abs(diff))
        pos = float(rk_all[diff > 0].sum())
        neg = float(rk_all[diff < 0].sum())
        count = n_all
        mean_v = n_all * (n_all + 1) / 4
        var_v = n_all * (n_all + 1) * (2 * n_all + 1) / 24
        tie_arr = np.abs(diff)
        if zero == "pratt":
            mean_v -= n_zero * (n_zero + 1) / 4
            var_v -= n_zero * (n_zero + 1) * (2 * n_zero + 1) / 24
        else:
            pos += float(rk_all[diff == 0].sum()) / 2
            neg += float(rk_all[diff == 0].sum()) / 2
    V = pos
    ad = tie_arr
    _, cnt = np.unique(tie_arr, return_counts=True)
    ties = bool((cnt > 1).any())
    cnt_t = cnt[cnt > 1]
    if zero == "pratt" and n_zero > 1:
        cnt_t = cnt_t[cnt_t != n_zero] if (cnt_t == n_zero).any() else cnt_t
    var_v -= float(np.sum(cnt_t ** 3 - cnt_t)) / 48
    sd = np.sqrt(var_v)
    nn = count
    use_exact = method == "exact" or (method == "auto" and not ties and n_zero == 0 and nn <= EXACT_MAX)
    if use_exact and (ties or n_zero):
        raise OESError("The exact test needs data without ties or zero differences. Use the normal approximation.")
    if use_exact:
        p = _tail(signrank_distribution(nn), V, alt)
        z = (V - mean_v) / sd
        how = "Exact null distribution of the signed-rank statistic"
    else:
        dv = V - mean_v
        c = 0.5 if cont else 0.0
        if alt == "two-sided":
            z = np.sign(dv) * max(abs(dv) - c, 0.0) / sd
            p = float(2 * stats.norm.sf(abs(z)))
        elif alt == "greater":
            z = (dv - c) / sd
            p = float(stats.norm.sf(z))
        else:
            z = (dv + c) / sd
            p = float(stats.norm.cdf(z))
        how = "Normal approximation with tie correction" + (" and continuity correction" if cont else "")
    p = min(1.0, p)
    rbc = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    # Hodges-Lehmann pseudomedian and interval from the Walsh averages
    dd = nz
    walsh = np.sort(((dd[:, None] + dd[None, :]) / 2)[np.triu_indices(len(dd))])
    M = len(walsh)
    hl = float(np.median(walsh))
    nd = len(dd)
    if nd <= EXACT_MAX and not ties:
        cdf = np.cumsum(signrank_distribution(nd))
        qu = int(np.searchsorted(cdf, alpha / 2 - 1e-12, side="left"))
        qu = max(qu, 1)
        how_ci = "exact"
    else:
        zc = stats.norm.ppf(1 - alpha / 2)
        qu = int(np.floor(nd * (nd + 1) / 4 - zc * np.sqrt(nd * (nd + 1) * (2 * nd + 1) / 24))) + 0
        qu = max(qu, 1)
        how_ci = "normal approximation"
    lo, hi = float(walsh[qu - 1]), float(walsh[M - qu])
    zc = stats.norm.ppf(1 - alpha / 2)
    label = f"{x1} - {x2}" if x2 else f"{x1} - {mu0:g}"
    tabs = [table("Differences", ["Item", "Value"], [["Pairs" if x2 else "Observations", n_all], ["Zero differences", n_zero], ["Positive differences", int((diff > 0).sum())], ["Negative differences", int((diff < 0).sum())],
                                                      ["Median difference", f(float(np.median(diff)))], ["Mean difference", f(float(diff.mean()))]]),
            table("Wilcoxon signed-rank test", ["Statistic", "Value"], [["V (sum of positive ranks)", f(V)], ["Sum of negative ranks", f(float(neg))], ["z", f(z, 4)], ["p-value", f(p, 6)], ["Alternative", alt], ["Zero handling", zero], ["Method", how]]),
            table("Effect size and location estimate", ["Measure", "Estimate", f"{int(round((1 - alpha) * 100))}% CI low", "CI high", "Reading"],
                  [["Matched-pairs rank-biserial correlation", f(rbc, 4), None, None, "(sum of positive ranks - sum of negative ranks) / total"],
                   ["Hodges-Lehmann pseudomedian of " + label, f(hl), f(lo), f(hi), f"interval from the Walsh averages ({how_ci})"]])]
    diag = [["Ties among the absolute differences", int((cnt > 1).sum()), None, "ties force the normal approximation" if ties else "no ties"]]
    if len(diff) >= 3 and len(diff) <= 5000 and np.ptp(diff) > 0:
        sw = stats.shapiro(diff)
        diag.append(["Shapiro-Wilk normality of the differences", f(sw.statistic, 4), f(sw.pvalue, 4), "if normality holds, the paired t test is more powerful"])
    sk = stats.skew(diff)
    diag.append(["Skewness of the differences", f(sk, 4), None, "the test assumes the differences are symmetric about their median"])
    tabs.append(tests_table(diag, "Assumption checks"))
    sd_ = np.sort(diff)
    charts = [chart("Sorted differences", list(range(1, len(sd_) + 1)), [("Difference", sd_)], "Rank of the difference", label, kind="scatter"),
              chart("Walsh averages (sorted)", list(range(1, M + 1))[::max(1, M // 150)], [("Walsh average", walsh[::max(1, M // 150)])], "Order", "Value")]
    notes = [f"{n_all} {'pairs' if x2 else 'observations'}" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent pairs and a distribution of the differences that is symmetric about its median."]
    summary = [stat("V", V, 2), stat("z", z, 4), stat("p-value", p, 6), stat("Rank-biserial r", rbc, 4), stat("Pseudomedian", hl, 4)]
    return result("stat_wilcoxon", "Wilcoxon signed-rank test", n_all, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# B3. Kruskal-Wallis
# ---------------------------------------------------------------------------------------------------------------
def kruskal(df, spec):
    y, g = need(spec, "y", "outcome variable"), need(spec, "group", "grouping variable")
    distinct(y, g)
    alpha = _alpha(spec)
    adj = spec.get("adjust") or "holm"
    d, dropped = clean_frame(df, numeric=[y], factors=[g])
    lv = levels(d[g])
    k = len(lv)
    if k < 3:
        raise OESError("Kruskal-Wallis needs at least three groups; use the Mann-Whitney test for two.")
    groups = [d.loc[d[g] == l, y].to_numpy(float) for l in lv]
    n_i = np.array([len(v) for v in groups])
    if (n_i < 2).any():
        raise OESError("Every group needs at least two observations.")
    N = int(n_i.sum())
    allv = np.concatenate(groups)
    rk = stats.rankdata(allv)
    _, cnt = np.unique(allv, return_counts=True)
    tie = 1 - float(np.sum(cnt ** 3 - cnt)) / (N ** 3 - N)
    if tie <= 0:
        raise OESError("All observations are tied, so the test is undefined.")
    idx = np.cumsum(np.r_[0, n_i])
    R = np.array([rk[idx[i]:idx[i + 1]].sum() for i in range(k)])
    H = (12.0 / (N * (N + 1)) * np.sum(R ** 2 / n_i) - 3 * (N + 1)) / tie
    dfh = k - 1
    p = float(stats.chi2.sf(H, dfh))
    eps2 = H * (N + 1) / (N ** 2 - 1)
    eta2h = max(0.0, (H - k + 1) / (N - k))
    mr = R / n_i
    tabs = [table("Group descriptives", ["Group", "n", "Median", "Mean", "Mean rank", "Rank sum"], [[l, int(n), f(np.median(v)), f(v.mean()), f(m), f(r)] for l, n, v, m, r in zip(lv, n_i, groups, mr, R)]),
            table("Kruskal-Wallis test", ["Statistic", "Value"], [["H (tie corrected)", f(H, 4)], ["df", dfh], ["p-value (chi-square)", f(p, 6)], ["Tie correction factor", f(tie, 5)]]),
            table("Effect sizes", ["Measure", "Estimate", "Reading"], [["Epsilon squared", f(eps2, 4), "share of the rank variance explained by the groups"], ["Eta squared based on H", f(eta2h, 4), "(H - k + 1) / (N - k)"]])]
    pairs = list(itertools.combinations(range(k), 2))
    sigma = np.sqrt(N * (N + 1) / 12.0 * tie)
    zs, ps = [], []
    for i, j in pairs:
        z = (mr[i] - mr[j]) / (sigma * np.sqrt(1.0 / n_i[i] + 1.0 / n_i[j]))
        zs.append(z)
        ps.append(float(2 * stats.norm.sf(abs(z))))
    ap = _adjust(ps, adj)
    rows = [[f"{lv[i]} - {lv[j]}", f(mr[i] - mr[j]), f(z, 4), f(p_, 6), f(a_, 6), "yes" if a_ < alpha else "no"] for (i, j), z, p_, a_ in zip(pairs, zs, ps, ap)]
    tabs.append(table(f"Dunn pairwise comparisons ({adj} adjustment)", ["Pair", "Mean-rank difference", "z", "p-value", "Adjusted p", f"Different at {alpha:g}"], rows,
                      "Dunn's test uses the pooled rank variance with the tie correction. The adjusted p-values control the family-wise error rate (Holm, Bonferroni) or the false discovery rate (Benjamini-Hochberg)."))
    diag = []
    lev = stats.levene(*groups, center="median")
    diag.append(["Levene equality of spreads", f(lev.statistic, 4), f(lev.pvalue, 4), "the test reads as a difference in medians only if the shapes are alike"])
    diag.append(["ANOVA F test (parametric comparison)", f(stats.f_oneway(*groups).statistic, 4), f(stats.f_oneway(*groups).pvalue, 6), ""])
    diag.append(["Smallest group size", int(n_i.min()), None, "the chi-square approximation is reliable when each group has at least 5 observations" + ("" if n_i.min() >= 5 else "; here it is smaller, so read the p-value with care")])
    tabs.append(tests_table(diag, "Assumption checks"))
    charts = [chart("Mean rank by group", lv, [("Mean rank", mr)], "Group", "Mean rank", kind="bar"),
              chart("Group medians", lv, [("Median", [float(np.median(v)) for v in groups])], "Group", y, kind="bar")]
    notes = [f"{k} groups, {N} observations" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Assumes independent observations and an ordinal or continuous outcome; with groups of similar shape it compares medians."]
    summary = [stat("H", H, 4), stat("p-value", p, 6), stat("Epsilon squared", eps2, 4), stat("Groups", k)]
    return result("stat_kruskal", "Kruskal-Wallis test", N, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# B4. Friedman
# ---------------------------------------------------------------------------------------------------------------
def friedman(df, spec):
    subj, w, y = need(spec, "subject", "subject (block) identifier"), need(spec, "treatment", "treatment or condition"), need(spec, "y", "outcome variable")
    distinct(subj, w, y)
    alpha = _alpha(spec)
    adj = spec.get("adjust") or "holm"
    d, dropped = clean_frame(df, numeric=[y], factors=[subj, w])
    lv = levels(d[w])
    k = len(lv)
    if k < 3:
        raise OESError("The Friedman test needs at least three conditions; use the Wilcoxon signed-rank test for two.")
    if (d.groupby([subj, w]).size() > 1).any():
        raise OESError("Some subjects have more than one value in the same condition. Average them first.")
    wide = d.pivot(index=subj, columns=w, values=y)[lv]
    comp = wide.dropna()
    lost = len(wide) - len(comp)
    n = len(comp)
    if n < 3:
        raise OESError("At least three subjects with a value in every condition are needed.")
    Y = comp.to_numpy(float)
    ranks = np.apply_along_axis(stats.rankdata, 1, Y)
    Rj = ranks.sum(0)
    ties_by_row = []
    for r in Y:
        _, c = np.unique(r, return_counts=True)
        ties_by_row.append(float(np.sum(c ** 3 - c)))
    C = 1 - sum(ties_by_row) / (n * k * (k ** 2 - 1))
    if C <= 0:
        raise OESError("Every subject has identical values in all conditions, so the test is undefined.")
    chi = (12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3 * n * (k + 1)) / C
    p = float(stats.chi2.sf(chi, k - 1))
    Wk = chi / (n * (k - 1))
    tabs = [table("Condition descriptives", ["Condition", "n", "Median", "Mean", "Mean rank", "Rank sum"], [[l, n, f(np.median(Y[:, i])), f(Y[:, i].mean()), f(Rj[i] / n), f(Rj[i])] for i, l in enumerate(lv)]),
            table("Friedman test", ["Statistic", "Value"], [["Chi-square (tie corrected)", f(chi, 4)], ["df", k - 1], ["p-value", f(p, 6)]]),
            table("Effect size", ["Measure", "Estimate", "Reading"], [["Kendall's W (coefficient of concordance)", f(Wk, 4), "0 = no agreement between subjects, 1 = perfect agreement"]])]
    pairs = list(itertools.combinations(range(k), 2))
    # Nemenyi: studentized range on the rank sums
    se_n = np.sqrt(k * (k + 1) / (6.0 * n))
    # Conover: t distribution with the Friedman rank-sum variance
    A1 = float(np.sum(ranks ** 2))
    B1 = float(np.sum(Rj ** 2)) / n
    se_c = np.sqrt(2 * n * (A1 - B1) / ((n - 1) * (k - 1)))
    rows, pn, pc = [], [], []
    for i, j in pairs:
        diff = Rj[i] - Rj[j]
        q = abs(diff) / n / se_n * np.sqrt(2)          # Demsar (2006): q = |mean-rank difference| / sqrt(k(k+1)/(6n)), referred to the studentized range / sqrt(2)
        pn.append(float(stats.studentized_range.sf(q, k, np.inf)))
        t = abs(diff) / se_c
        pc.append(float(2 * stats.t.sf(t, (n - 1) * (k - 1))))
    pcn = _adjust(pc, adj)
    for (i, j), a_, b_, c_ in zip(pairs, pn, pc, pcn):
        rows.append([f"{lv[i]} - {lv[j]}", f((Rj[i] - Rj[j]) / n), f(a_, 6), f(b_, 6), f(c_, 6), "yes" if c_ < alpha else "no"])
    tabs.append(table("Pairwise comparisons", ["Pair", "Mean-rank difference", "Nemenyi p (family-wise)", "Conover p", f"Conover p ({adj})", f"Different at {alpha:g}"], rows,
                      "Nemenyi's test uses the studentized range distribution on the rank sums; Conover's test uses a t distribution and is more powerful, so it needs the step-down adjustment shown."))
    diag = [["Subjects used", n, None, f"{lost} subject(s) with a missing condition were dropped" if lost else "balanced complete blocks"],
            ["Repeated-measures ANOVA F (parametric comparison)", f(_rm_f(Y), 4), None, "reported for comparison; assumes normal residuals and sphericity"]]
    tabs.append(tests_table(diag, "Assumption checks"))
    charts = [chart("Mean rank by condition", lv, [("Mean rank", Rj / n)], "Condition", "Mean rank", kind="bar"),
              chart("Condition medians", lv, [("Median", [float(np.median(Y[:, i])) for i in range(k)])], "Condition", y, kind="bar")]
    notes = [f"{n} subjects, {k} conditions" + (f"; {lost} subject(s) with a missing condition were dropped." if lost else "."), "Assumes independent subjects (blocks) and an ordinal or continuous outcome measured in every condition."]
    summary = [stat("Chi-square", chi, 4), stat("p-value", p, 6), stat("Kendall's W", Wk, 4), stat("Conditions", k)]
    return result("stat_friedman", "Friedman test", n * k, summary, tabs, charts, notes)


def _rm_f(Y):
    n, k = Y.shape
    gm = Y.mean()
    ss_c = n * np.sum((Y.mean(0) - gm) ** 2)
    ss_s = k * np.sum((Y.mean(1) - gm) ** 2)
    ss_e = np.sum((Y - gm) ** 2) - ss_c - ss_s
    return float((ss_c / (k - 1)) / (ss_e / ((n - 1) * (k - 1)))) if ss_e > 0 else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# B5. Kendall's tau
# ---------------------------------------------------------------------------------------------------------------
def _kendall_counts(x, y):
    n = len(x)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    prod = (dx * dy)[np.triu_indices(n, 1)]
    conc = int((prod > 0).sum())
    disc = int((prod < 0).sum())
    return conc, disc


def kendall(df, spec):
    x, y = need(spec, "x", "first variable"), need(spec, "y", "second variable")
    distinct(x, y)
    alpha = _alpha(spec)
    alt = spec.get("alternative") or "two-sided"
    method = spec.get("method") or "auto"
    d, dropped = clean_frame(df, numeric=[x, y])
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    n = len(xv)
    if n < 4:
        raise OESError("At least four complete pairs are needed.")
    conc, disc = _kendall_counts(xv, yv)
    n0 = n * (n - 1) / 2
    _, tx = np.unique(xv, return_counts=True)
    _, ty = np.unique(yv, return_counts=True)
    n1 = float(np.sum(tx * (tx - 1) / 2))
    n2 = float(np.sum(ty * (ty - 1) / 2))
    S = conc - disc
    if n0 - n1 <= 0 or n0 - n2 <= 0:
        raise OESError("One of the variables is constant, so the correlation is undefined.")
    tau_a = S / n0
    tau_b = S / np.sqrt((n0 - n1) * (n0 - n2))
    m = min(len(tx), len(ty))
    tau_c = 2 * S * m / (n ** 2 * (m - 1)) if m > 1 else float("nan")
    ties = bool(n1 or n2)
    v0 = n * (n - 1) * (2 * n + 5)
    vt = float(np.sum(tx * (tx - 1) * (2 * tx + 5)))
    vu = float(np.sum(ty * (ty - 1) * (2 * ty + 5)))
    v1 = float(np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1))) / (2.0 * n * (n - 1))
    v2 = float(np.sum(tx * (tx - 1) * (tx - 2)) * np.sum(ty * (ty - 1) * (ty - 2))) / (9.0 * n * (n - 1) * (n - 2))
    varS = (v0 - vt - vu) / 18.0 + v1 + v2
    use_exact = method == "exact" or (method == "auto" and not ties and n <= EXACT_MAX)
    if use_exact and ties:
        raise OESError("The exact test needs data without ties. Use the asymptotic method.")
    if use_exact:
        pmf = kendall_distribution(n)
        # discordant pairs D = (n0 - S) / 2; tau large <=> D small
        p = _tail(pmf, disc, "less" if alt == "greater" else "greater" if alt == "less" else "two-sided")
        z = S / np.sqrt(varS)
        how = "Exact null distribution of the number of discordant pairs"
    else:
        z = S / np.sqrt(varS)
        p = float(2 * stats.norm.sf(abs(z))) if alt == "two-sided" else float(stats.norm.sf(z)) if alt == "greater" else float(stats.norm.cdf(z))
        how = "Normal approximation with the tie-corrected variance of S"
    # confidence interval for tau-b: Fisher transform with the variance 0.437 / (n - 4) (Fieller, Hartley and Pearson 1957)
    zc = stats.norm.ppf(1 - alpha / 2)
    se_z = np.sqrt(0.437 / (n - 4)) if n > 4 else np.nan
    zt = np.arctanh(np.clip(tau_b, -0.999999, 0.999999))
    ci = (float(np.tanh(zt - zc * se_z)), float(np.tanh(zt + zc * se_z)))
    rho, rho_p = stats.spearmanr(xv, yv)
    r_, r_p = stats.pearsonr(xv, yv)
    tabs = [table("Kendall's tau", ["Measure", "Estimate", f"{int(round((1 - alpha) * 100))}% CI low", "CI high"],
                  [["Tau-b (adjusted for ties)", f(tau_b, 5), f(ci[0], 4), f(ci[1], 4)], ["Tau-a", f(tau_a, 5), None, None], ["Tau-c (Stuart)", f(tau_c, 5), None, None]],
                  "The interval transforms tau-b with the Fisher z transform and the variance 0.437 / (n - 4); it is approximate and most accurate without ties."),
            table("Test of independence", ["Statistic", "Value"], [["Concordant pairs", conc], ["Discordant pairs", disc], ["Pairs tied in x only / y only", f"{int(n1)} / {int(n2)}"], ["S = concordant - discordant", S], ["z", f(z, 4)], ["p-value", f(p, 6)], ["Alternative", alt], ["Method", how]]),
            table("Other correlation measures for comparison", ["Measure", "Estimate", "p-value"], [["Spearman's rho", f(rho, 5), f(rho_p, 6)], ["Pearson's r", f(r_, 5), f(r_p, 6)]])]
    diag = [["Tied values in x", int(sum(1 for c in tx if c > 1)), None, "tau-b corrects for ties" if ties else "no ties"], ["Tied values in y", int(sum(1 for c in ty if c > 1)), None, ""],
            ["Difference |Pearson r| - |tau-b|", f(abs(r_) - abs(tau_b), 4), None, "large gaps point to nonlinearity or outliers"]]
    tabs.append(tests_table(diag, "Checks"))
    order = np.argsort(xv)
    rx, ry = stats.rankdata(xv), stats.rankdata(yv)
    cx, cy = list(xv[order][:400]), list(yv[order][:400])
    charts = [chart(f"{y} against {x}", cx, [(y, cy)], x, y, kind="scatter"),
              chart("Ranks", list(rx[:400]), [("Rank of y", list(ry[:400]))], f"Rank of {x}", f"Rank of {y}", kind="scatter")]
    notes = [f"{n} complete pairs" + (f"; {dropped} row(s) with missing values were dropped." if dropped else "."),
             "Measures monotone association; assumes independent pairs. Tau is the difference between the probabilities of concordance and discordance."]
    summary = [stat("Tau-b", tau_b, 4), stat("z", z, 4), stat("p-value", p, 6), stat("Spearman rho", rho, 4)]
    return result("stat_kendall", "Kendall's tau correlation", n, summary, tabs, charts, notes)
