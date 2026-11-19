"""Category F of the Statistics Division: resampling and simulation-based methods.

The nonparametric bootstrap (percentile, basic, BCa and normal intervals), permutation tests (exact enumeration when the number
of rearrangements is small, Monte Carlo with the Phipson-Smyth correction otherwise) and power / sample-size calculations for
the common tests (t, ANOVA, correlation, two proportions, chi-square). The random number generator is seeded, so a run can be
reproduced exactly."""
import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import OESError, as_list, chart, distinct, f, float_option, int_option, need, result, stat, table, tests_table
from .stat_core import clean_frame, levels, thin

CHUNK = 4_000_000  # numbers held in memory at once when resampling


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


def _hist_chart(title, values, mark, xlabel, bins=30):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    labels = [f"{c:.4g}" for c in centres]
    return chart(title, labels, [("Frequency", counts.astype(float))], xlabel, "count", "bar")


# ---------------------------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------------------------
def _rowcorr(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (A * B).sum(1) / den


def _stat_rows(name, Y, X=None, G=None, trim=0.1):
    """Statistic computed on every row of the (m, n) arrays Y (and X, G). G holds 1 for the first group and 0 for the second."""
    if name == "mean":
        return Y.mean(1)
    if name == "median":
        return np.median(Y, axis=1)
    if name == "trimmed_mean":
        return stats.trim_mean(Y, trim, axis=1)
    if name == "sd":
        return Y.std(1, ddof=1)
    if name == "var":
        return Y.var(1, ddof=1)
    if name == "correlation":
        return _rowcorr(Y, X)
    if name == "spearman":
        return _rowcorr(stats.rankdata(Y, axis=1), stats.rankdata(X, axis=1))
    if name == "slope":
        Xc = X - X.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return ((Y - Y.mean(1, keepdims=True)) * Xc).sum(1) / (Xc * Xc).sum(1)
    if name == "mean_diff":
        g1, g0 = G == 1, G == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            return (Y * g1).sum(1) / g1.sum(1) - (Y * g0).sum(1) / g0.sum(1)
    if name == "median_diff":
        a = np.where(G == 1, Y, np.nan)
        b = np.where(G == 0, Y, np.nan)
        return np.nanmedian(a, axis=1) - np.nanmedian(b, axis=1)
    raise OESError("Unknown statistic.")


_BOOT_STATS = {
    "mean": "Mean", "median": "Median", "trimmed_mean": "10% trimmed mean", "sd": "Standard deviation", "var": "Variance",
    "correlation": "Pearson correlation", "spearman": "Spearman correlation", "slope": "Regression slope of y on x",
    "mean_diff": "Difference in means (first minus second group)", "median_diff": "Difference in medians (first minus second group)",
}


def _boot_resample(name, y, x, g, B, rng):
    """Bootstrap replicates of the statistic. Two-group statistics resample within each group (stratified)."""
    n = len(y)
    out = np.empty(B)
    if name in ("mean_diff", "median_diff"):
        i1, i0 = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
        step = max(1, CHUNK // n)
        done = 0
        while done < B:
            m = min(step, B - done)
            idx = np.concatenate([i1[rng.integers(0, len(i1), (m, len(i1)))], i0[rng.integers(0, len(i0), (m, len(i0)))]], axis=1)
            out[done:done + m] = _stat_rows(name, y[idx], None, g[idx])
            done += m
        return out
    step = max(1, CHUNK // n)
    done = 0
    while done < B:
        m = min(step, B - done)
        idx = rng.integers(0, n, (m, n))
        out[done:done + m] = _stat_rows(name, y[idx], x[idx] if x is not None else None, None)
        done += m
    return out


def _jackknife(name, y, x, g):
    n = len(y)
    out = np.empty(n)
    keep = ~np.eye(n, dtype=bool)
    step = max(1, CHUNK // n)
    for s in range(0, n, step):
        e = min(n, s + step)
        rows = np.array([np.flatnonzero(keep[i]) for i in range(s, e)])
        out[s:e] = _stat_rows(name, y[rows], x[rows] if x is not None else None, g[rows] if g is not None else None)
    return out


def bca_interval(theta, boot, jack, alpha):
    """Bias-corrected and accelerated interval from the bootstrap replicates and the jackknife values."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    below = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    if below <= 0 or below >= 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    z0 = stats.norm.ppf(below)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * np.sum((jm - jack) ** 2) ** 1.5
    a = num / den if den > 0 else 0.0
    out = []
    for p in (alpha / 2, 1 - alpha / 2):
        zp = stats.norm.ppf(p)
        adj = stats.norm.cdf(z0 + (z0 + zp) / (1 - a * (z0 + zp)))
        out.append(float(np.quantile(boot, adj, method="linear")))
    return out[0], out[1], float(z0), float(a)


def bootstrap(df, spec):
    alpha = _alpha(spec)
    name = need(spec, "statistic", "statistic")
    if name not in _BOOT_STATS:
        raise OESError("Unknown statistic.")
    B = int_option(spec, "n_boot", 2000, 200, 100000, "Number of bootstrap resamples")
    seed = int_option(spec, "seed", 12345, 0, 2 ** 31 - 1, "Random seed")
    ycol = need(spec, "y", "variable")
    xcol = spec.get("x") or None
    gcol = spec.get("group") or None
    need_x = name in ("correlation", "spearman", "slope")
    need_g = name in ("mean_diff", "median_diff")
    if need_x and not xcol:
        raise OESError("This statistic needs a second variable (x).")
    if need_g and not gcol:
        raise OESError("This statistic needs a grouping variable with two groups.")
    use_x = xcol if need_x else None
    use_g = gcol if need_g else None
    distinct(ycol, use_x, use_g)
    d, dropped = clean_frame(df, numeric=[ycol] + ([use_x] if use_x else []), factors=[use_g] if use_g else [])
    y = d[ycol].to_numpy(float)
    x = d[use_x].to_numpy(float) if use_x else None
    g = None
    if use_g:
        lv = levels(d[use_g])
        if len(lv) != 2:
            raise OESError(f"The grouping variable must have exactly two levels (it has {len(lv)}).")
        g = (d[use_g] == lv[0]).to_numpy().astype(int)
        if g.sum() < 2 or (1 - g).sum() < 2:
            raise OESError("Each group needs at least two observations.")
    n = len(y)
    if n < 5:
        raise OESError("At least five observations are needed.")
    if name in ("correlation", "spearman", "slope") and (np.ptp(y) == 0 or np.ptp(x) == 0):
        raise OESError("A variable is constant, so the statistic is undefined.")
    theta = float(_stat_rows(name, y[None], x[None] if x is not None else None, g[None] if g is not None else None)[0])
    rng = np.random.default_rng(seed)
    boot = _boot_resample(name, y, x, g, B, rng)
    ok = np.isfinite(boot)
    if ok.sum() < 0.9 * B:
        raise OESError("The statistic is undefined in many resamples (a constant variable or an empty group). Use a larger sample or another statistic.")
    boot = boot[ok]
    se = float(boot.std(ddof=1))
    bias = float(boot.mean() - theta)
    lo_p, hi_p = (float(np.quantile(boot, q)) for q in (alpha / 2, 1 - alpha / 2))
    lo_b, hi_b = 2 * theta - hi_p, 2 * theta - lo_p
    zc = stats.norm.ppf(1 - alpha / 2)
    lo_n, hi_n = theta - bias - zc * se, theta - bias + zc * se
    jack = _jackknife(name, y, x, g) if n <= 4000 else None
    if jack is not None and np.all(np.isfinite(jack)):
        lo_a, hi_a, z0, acc = bca_interval(theta, boot, jack, alpha)
    else:
        lo_a = hi_a = z0 = acc = float("nan")
    lab = f"{int(round((1 - alpha) * 100))}%"
    rows = [["Percentile", f(lo_p, 5), f(hi_p, 5)], ["Basic (reverse percentile)", f(lo_b, 5), f(hi_b, 5)], ["BCa (bias-corrected and accelerated)", f(lo_a, 5), f(hi_a, 5)], ["Normal (bias-corrected)", f(lo_n, 5), f(hi_n, 5)]]
    tabs = [table(f"Bootstrap estimate of the {_BOOT_STATS[name].lower()}", ["Quantity", "Value"],
                  [["Observed statistic", f(theta, 6)], ["Bootstrap standard error", f(se, 6)], ["Bootstrap bias (mean of replicates minus estimate)", f(bias, 6)], ["Bootstrap median", f(float(np.median(boot)), 6)],
                   ["Resamples", int(ok.sum())], ["Observations", n]]),
            table(f"{lab} bootstrap confidence intervals", ["Method", "Lower", "Upper"], rows,
                  "BCa corrects for bias and skewness and is usually the most accurate; percentile and basic intervals can be too narrow in small samples." + ("" if jack is not None else " BCa was not computed for n above 4000.")),
            table("BCa constants", ["Constant", "Value"], [["Bias correction z0", f(z0, 5)], ["Acceleration a (jackknife)", f(acc, 5)]])]
    summary = [stat("Observations", n), stat("Estimate", theta), stat("Bootstrap SE", se), stat(f"BCa {lab} low", lo_a), stat(f"BCa {lab} high", hi_a)]
    charts = [_hist_chart("Bootstrap distribution of the statistic", boot, theta, _BOOT_STATS[name])]
    notes = [f"{B} resamples drawn with replacement" + (" within each group" if use_g else "") + f" (seed {seed}); the same seed reproduces the same result.",
             "Resampling the observed cases treats the sample as the population; it works poorly for extreme statistics (maximum, minimum) and for very small samples."]
    if abs(bias) > 0.25 * se:
        notes.append("The bootstrap bias is large relative to the standard error; prefer the BCa interval.")
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_bootstrap", "Nonparametric bootstrap", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------------------------------------------
_PERM_TESTS = {"mean_diff": "Difference in means", "median_diff": "Difference in medians", "paired": "Paired mean difference (sign-flip test)", "correlation": "Pearson correlation", "anova_f": "One-way ANOVA F"}


def _perm_p(null, obs, alt):
    tol = 1e-9 * max(1.0, abs(obs))
    if alt == "two-sided":
        return np.sum(np.abs(null) >= abs(obs) - tol)
    if alt == "greater":
        return np.sum(null >= obs - tol)
    return np.sum(null <= obs + tol)


def permutation(df, spec):
    alpha = _alpha(spec)
    test = need(spec, "test", "test")
    if test not in _PERM_TESTS:
        raise OESError("Unknown test.")
    alt = spec.get("alternative") or "two-sided"
    B = int_option(spec, "n_perm", 10000, 500, 1000000, "Number of random permutations")
    cap = int_option(spec, "exact_max", 200000, 1000, 2000000, "Largest number of rearrangements to enumerate exactly")
    seed = int_option(spec, "seed", 12345, 0, 2 ** 31 - 1, "Random seed")
    ycol = need(spec, "y", "outcome variable")
    gcol = spec.get("group") or None
    xcol = spec.get("x") or None
    if test in ("mean_diff", "median_diff", "anova_f") and not gcol:
        raise OESError("This test needs a grouping variable.")
    if test in ("paired", "correlation") and not xcol:
        raise OESError("This test needs a second variable (x).")
    use_g = gcol if test in ("mean_diff", "median_diff", "anova_f") else None
    use_x = xcol if test in ("paired", "correlation") else None
    distinct(ycol, use_g, use_x)
    d, dropped = clean_frame(df, numeric=[ycol] + ([use_x] if use_x else []), factors=[use_g] if use_g else [])
    y = d[ycol].to_numpy(float)
    n = len(y)
    rng = np.random.default_rng(seed)
    exact = False
    note_ex = ""
    if test in ("mean_diff", "median_diff"):
        lv = levels(d[use_g])
        if len(lv) != 2:
            raise OESError(f"This test needs exactly two groups (the grouping variable has {len(lv)}). Use the ANOVA F statistic for more groups.")
        gi = (d[use_g] == lv[0]).to_numpy()
        n1, n2 = int(gi.sum()), int((~gi).sum())
        if n1 < 2 or n2 < 2:
            raise OESError("Each group needs at least two observations.")

        def stat_fn(idx_first):
            """statistic for the arrangements whose first-group members are the rows of idx_first (m, n1)"""
            m = idx_first.shape[0]
            mask = np.zeros((m, n), bool)
            np.put_along_axis(mask, idx_first, True, axis=1)
            if test == "mean_diff":
                return (np.where(mask, y, 0).sum(1) / n1) - (np.where(~mask, y, 0).sum(1) / n2)
            a = np.where(mask, y, np.nan)
            b = np.where(~mask, y, np.nan)
            return np.nanmedian(a, axis=1) - np.nanmedian(b, axis=1)

        obs = float(stat_fn(np.flatnonzero(gi)[None])[0])
        total = math.comb(n, n1)
        if total <= cap:
            exact = True
            combs = np.array(list(itertools.combinations(range(n), n1)), dtype=np.int32)
            null = np.concatenate([stat_fn(combs[s:s + 20000]) for s in range(0, len(combs), 20000)])
            b = _perm_p(null, obs, alt)
            p = b / len(null)
        else:
            nulls = []
            step = max(1, CHUNK // n)
            done = 0
            while done < B:
                m = min(step, B - done)
                nulls.append(stat_fn(np.argsort(rng.random((m, n)), axis=1)[:, :n1]))
                done += m
            null = np.concatenate(nulls)
            b = _perm_p(null, obs, alt)
            p = (b + 1) / (len(null) + 1)
        ref = stats.ttest_ind(y[gi], y[~gi], equal_var=False, alternative=alt)
        ref_row = ["Welch t-test", f(ref.statistic, 4), f(ref.pvalue, 6)]
        desc = f"{levels(d[use_g])[0]} minus {levels(d[use_g])[1]}"
    elif test == "paired":
        xv = d[use_x].to_numpy(float)
        diff = y - xv
        n = len(diff)
        if n < 3:
            raise OESError("At least three pairs are needed.")
        obs = float(diff.mean())
        if 2 ** n <= cap:
            exact = True
            signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2.0 - 1.0
            null = (signs * diff).mean(1)
            b = _perm_p(null, obs, alt)
            p = b / len(null)
        else:
            nulls = []
            step = max(1, CHUNK // n)
            done = 0
            while done < B:
                m = min(step, B - done)
                s = rng.integers(0, 2, (m, n)) * 2.0 - 1.0
                nulls.append((s * diff).mean(1))
                done += m
            null = np.concatenate(nulls)
            b = _perm_p(null, obs, alt)
            p = (b + 1) / (len(null) + 1)
        ref = stats.ttest_rel(y, xv, alternative=alt)
        ref_row = ["Paired t-test", f(ref.statistic, 4), f(ref.pvalue, 6)]
        desc = f"{ycol} minus {use_x}"
        y = diff
    elif test == "correlation":
        xv = d[use_x].to_numpy(float)
        if np.ptp(y) == 0 or np.ptp(xv) == 0:
            raise OESError("A variable is constant, so the correlation is undefined.")
        obs = float(np.corrcoef(y, xv)[0, 1])
        if math.factorial(n) <= cap:
            exact = True
            perms = np.array(list(itertools.permutations(range(n))), dtype=np.int32)
            null = _rowcorr(y[perms], np.broadcast_to(xv, perms.shape).copy())
            b = _perm_p(null, obs, alt)
            p = b / len(null)
        else:
            nulls = []
            step = max(1, CHUNK // n)
            done = 0
            xb = None
            while done < B:
                m = min(step, B - done)
                idx = np.argsort(rng.random((m, n)), axis=1)
                nulls.append(_rowcorr(y[idx], np.broadcast_to(xv, (m, n)).copy()))
                done += m
            null = np.concatenate(nulls)
            b = _perm_p(null, obs, alt)
            p = (b + 1) / (len(null) + 1)
        ref = stats.pearsonr(y, xv, alternative=alt)
        ref_row = ["Pearson correlation test", f(ref.statistic, 4), f(ref.pvalue, 6)]
        desc = f"correlation of {ycol} and {use_x}"
    else:
        lv = levels(d[use_g])
        k = len(lv)
        if k < 3:
            raise OESError("The F statistic needs at least three groups; use the difference in means for two.")
        gidx = d[use_g].map({v: i for i, v in enumerate(lv)}).to_numpy()
        sizes = np.bincount(gidx, minlength=k).astype(float)
        if (sizes < 2).any():
            raise OESError("Each group needs at least two observations.")
        alt = "greater"

        def fstat(Yr):
            # Yr: (m, n) outcome arrangements against the fixed labels gidx
            tot = Yr.mean(1, keepdims=True)
            sums = np.stack([Yr[:, gidx == j].sum(1) for j in range(k)], axis=1)
            means = sums / sizes
            ssb = (sizes * (means - tot) ** 2).sum(1)
            ssw = ((Yr - tot) ** 2).sum(1) - ssb
            return (ssb / (k - 1)) / (ssw / (n - k))

        obs = float(fstat(y[None])[0])
        nulls = []
        step = max(1, CHUNK // n)
        done = 0
        while done < B:
            m = min(step, B - done)
            idx = np.argsort(rng.random((m, n)), axis=1)
            nulls.append(fstat(y[idx]))
            done += m
        null = np.concatenate(nulls)
        b = _perm_p(null, obs, "greater")
        p = (b + 1) / (len(null) + 1)
        ref = stats.f_oneway(*[y[gidx == j] for j in range(k)])
        ref_row = ["One-way ANOVA F test", f(ref.statistic, 4), f(ref.pvalue, 6)]
        desc = "between-group F ratio"
        note_ex = "The F test is one-sided by construction (large F = evidence against equal means)."
    m_used = len(null)
    se_p = float(np.sqrt(max(p * (1 - p), 0) / m_used)) if not exact else 0.0
    tabs = [table("Permutation test", ["Quantity", "Value"],
                  [["Test", _PERM_TESTS[test]], ["Observed statistic (" + desc + ")", f(obs, 6)], ["Alternative", alt], ["Rearrangements used", m_used], ["Exact enumeration", "Yes" if exact else "No (Monte Carlo)"],
                   ["Permutation p-value", f(p, 6)], ["Monte Carlo standard error of p", f(se_p, 6) if not exact else "none (exact)"], ["Random seed", seed if not exact else "not used"]],
                  "Exact p-value = share of all rearrangements at least as extreme as the observed one. Monte Carlo p-value = (b + 1) / (m + 1), where b counts the random rearrangements at least as extreme (Phipson and Smyth 2010); it is never zero and is a valid p-value."),
            table("Reference parametric test", ["Test", "Statistic", "p-value"], [ref_row], "Shown for comparison only; the permutation p-value does not rely on a normal or equal-variance assumption."),
            table("Quantiles of the permutation distribution", ["Quantile", "Value"], [[f"{q * 100:g}%", f(float(np.quantile(null, q)), 5)] for q in (0.025, 0.25, 0.5, 0.75, 0.975)])]
    charts = [_hist_chart("Permutation distribution of the statistic", null, obs, _PERM_TESTS[test])]
    summary = [stat("Observations", n), stat("Observed statistic", obs), stat("Permutation p-value", p, 6), stat("Rearrangements", m_used)]
    notes = ["The test assumes the observations are exchangeable under the null hypothesis (random assignment, or independent identically distributed data)." if test != "paired" else "The sign-flip test assumes the pair differences are symmetric about zero under the null hypothesis."]
    if note_ex:
        notes.append(note_ex)
    if not exact and se_p > 0.1 * min(p, 1 - p) and p < 0.5:
        notes.append("The Monte Carlo error is large compared with the p-value; increase the number of permutations.")
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return result("stat_permutation", "Permutation test", n, summary, tabs, charts, notes)


# ---------------------------------------------------------------------------------------------------------------
# Power and sample size
# ---------------------------------------------------------------------------------------------------------------
def _tails(alt):
    return 1 if alt in ("greater", "less") else 2


def power_t(d, n1, alpha, alt, kind="two-sample", ratio=1.0):
    if kind == "two-sample":
        n2 = n1 * ratio
        df = n1 + n2 - 2
        nc = d * np.sqrt(n1 * n2 / (n1 + n2))
    else:
        df = n1 - 1
        nc = d * np.sqrt(n1)
    if df < 1:
        return float("nan")
    if abs(nc) > 35:
        return 1.0  # the noncentral t is numerically unreliable here and the power is 1 to many decimals
    if _tails(alt) == 2:
        tc = stats.t.isf(alpha / 2, df)
        return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
    tc = stats.t.isf(alpha, df)
    return float(stats.nct.sf(tc, df, abs(nc)))


def power_anova(fe, n_total, k, alpha):
    df1, df2 = k - 1, n_total - k
    if df2 < 1:
        return float("nan")
    fc = stats.f.isf(alpha, df1, df2)
    return float(stats.ncf.sf(fc, df1, df2, fe ** 2 * n_total))


def power_corr(r, n, alpha, alt):
    """Power of the test of a correlation (the Fisher-z method with the bias correction used by the pwr package)."""
    if n < 4:
        return float("nan")
    tt = _tails(alt)
    tcrit = stats.t.isf(alpha / tt, n - 2)
    rc = np.sqrt(tcrit ** 2 / (tcrit ** 2 + n - 2))
    zr = np.arctanh(abs(r)) + abs(r) / (2 * (n - 1))
    zrc = np.arctanh(rc)
    pw = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
    if tt == 2:
        pw += stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
    return float(pw)


def power_prop(h, n, alpha, alt, ratio=1.0):
    neff = n * ratio / (1 + ratio)
    zc = stats.norm.isf(alpha / _tails(alt))
    mu = abs(h) * np.sqrt(neff)
    pw = stats.norm.sf(zc - mu)
    if _tails(alt) == 2:
        pw += stats.norm.cdf(-zc - mu)
    return float(pw)


def power_chisq(w, n_total, df, alpha):
    cc = stats.chi2.isf(alpha, df)
    return float(stats.ncx2.sf(cc, df, n_total * w ** 2))


def _power_fn(test, spec_vals):
    alpha, alt, k, df, ratio = spec_vals["alpha"], spec_vals["alt"], spec_vals["k"], spec_vals["df"], spec_vals["ratio"]
    if test == "t_two":
        return lambda es, n: power_t(es, n, alpha, alt, "two-sample", ratio)
    if test == "t_one":
        return lambda es, n: power_t(es, n, alpha, alt, "one-sample")
    if test == "anova":
        return lambda es, n: power_anova(es, n * k, k, alpha)
    if test == "corr":
        return lambda es, n: power_corr(es, n, alpha, alt)
    if test == "prop":
        return lambda es, n: power_prop(es, n, alpha, alt, ratio)
    return lambda es, n: power_chisq(es, n, df, alpha)


_POWER_TESTS = {
    "t_two": ("Two-sample t-test", "Cohen's d", "per group (the first group when the ratio is not 1)", 0.2, 0.5, 0.8, 2),
    "t_one": ("One-sample or paired t-test", "Cohen's d", "pairs / subjects", 0.2, 0.5, 0.8, 2),
    "anova": ("One-way ANOVA", "Cohen's f", "per group", 0.1, 0.25, 0.4, 2),
    "corr": ("Test of a correlation", "correlation r", "pairs", 0.1, 0.3, 0.5, 4),
    "prop": ("Two proportions (arcsine method)", "Cohen's h", "per group (the first group when the ratio is not 1)", 0.2, 0.5, 0.8, 2),
    "chisq": ("Chi-square test (goodness of fit or independence)", "Cohen's w", "total observations", 0.1, 0.3, 0.5, 2),
}


def power_analysis(df, spec):
    test = need(spec, "test", "test")
    if test not in _POWER_TESTS:
        raise OESError("Unknown test.")
    solve = spec.get("solve_for") or "n"
    alpha = _alpha(spec)
    alt = spec.get("alternative") or "two-sided"
    k = int_option(spec, "k", 3, 2, 50, "Number of groups")
    dfc = int_option(spec, "df", 2, 1, 200, "Degrees of freedom")
    ratio = float_option(spec, "ratio", 1.0, "Allocation ratio")
    if ratio <= 0:
        raise OESError("The allocation ratio must be positive.")
    name, es_label, n_label, e_s, e_m, e_l, nmin = _POWER_TESTS[test]
    if test in ("anova", "chisq") and alt != "two-sided":
        alt = "two-sided"
    # an effect size may be given directly, or as two proportions
    p1, p2 = spec.get("p1"), spec.get("p2")
    es = spec.get("effect")
    if test == "prop" and p1 not in (None, "") and p2 not in (None, ""):
        p1, p2 = float(p1), float(p2)
        if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
            raise OESError("The two proportions must lie strictly between 0 and 1 and differ.")
        es = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        if alt == "greater" and es < 0 or alt == "less" and es > 0:
            es = abs(es) if alt == "greater" else -abs(es)
    vals = dict(alpha=alpha, alt=alt, k=k, df=dfc, ratio=ratio)
    pf = _power_fn(test, vals)
    tgt = float_option(spec, "power", 0.8, "Target power")
    n_in = spec.get("n")
    sgn = -1.0 if (alt == "less" and test in ("t_two", "t_one", "corr", "prop")) else 1.0
    if solve in ("n", "power") and es in (None, ""):
        raise OESError("Enter the effect size.")
    if solve in ("power", "effect") and n_in in (None, ""):
        raise OESError("Enter the sample size.")
    if solve in ("n", "effect") and not 0.05 <= tgt < 0.9999:
        raise OESError("The target power must be between 0.05 and 0.9999.")
    if es not in (None, ""):
        es = float(es)
        if es == 0 and solve != "effect":
            raise OESError("The effect size must not be zero.")
        if test == "corr" and not -0.999 < es < 0.999:
            raise OESError("The correlation must lie between -1 and 1.")
    ef = lambda v: abs(v)
    if solve == "n":
        lo = nmin
        hi = 10
        while pf(ef(es), hi) < tgt:
            hi *= 2
            if hi > 5e7:
                raise OESError("The sample size needed is above 50 million; the effect size is too small for a practical study.")
        # smallest integer n reaching the power (power is not strictly monotone for tiny n, so scan upward from a bracket)
        a, b = lo, hi
        while b - a > 1:
            mid = (a + b) // 2
            if pf(ef(es), mid) >= tgt:
                b = mid
            else:
                a = mid
        n_res = b if pf(ef(es), a) < tgt else a
        pw = pf(ef(es), n_res)
        res_row = [["Required sample size " + n_label, n_res], ["Achieved power", f(pw, 5)]]
        n_use = n_res
        es_use = es
    elif solve == "power":
        n_use = float(n_in)
        if n_use < nmin:
            raise OESError(f"The sample size must be at least {nmin}.")
        pw = pf(ef(es), n_use)
        res_row = [["Power", f(pw, 5)]]
        es_use = es
    else:
        n_use = float(n_in)
        if n_use < nmin:
            raise OESError(f"The sample size must be at least {nmin}.")
        top = 0.999 if test == "corr" else 20.0
        hi = 0.05
        while hi < top and not pf(hi, n_use) >= tgt:
            hi = min(hi * 1.5, top)
        if not pf(hi, n_use) >= tgt:
            raise OESError("Even the largest effect size cannot reach the target power with this sample size.")
        es_use = optimize.brentq(lambda e: pf(e, n_use) - tgt, 1e-6, hi, xtol=1e-10)
        pw = tgt
        res_row = [["Smallest detectable " + es_label, f(es_use, 5)], ["Power", f(tgt, 5)]]
    tabs = [table("Power analysis", ["Quantity", "Value"], [["Test", name], ["Significance level", alpha], ["Alternative", alt]] + ([["Groups", k]] if test == "anova" else []) + ([["Degrees of freedom", dfc]] if test == "chisq" else []) +
                  ([["Allocation ratio (second group / first)", ratio]] if test in ("t_two", "prop") and ratio != 1 else []) + [[es_label, f(es_use, 5)]] + ([["Sample size " + n_label, n_use]] if solve != "n" else []) + res_row,
                  "Power is computed from the exact noncentral t, F and chi-square distributions (correlation: Fisher z with the pwr-package bias correction; proportions: Cohen's arcsine h with the normal approximation).")]
    # power at the three conventional effect sizes
    n_ref = n_use
    tabs.append(table("Power at conventional effect sizes (same sample size)", [es_label, "Size", "Power"],
                      [[e, lab, f(pf(e, n_ref), 4)] for e, lab in ((e_s, "small"), (e_m, "medium"), (e_l, "large"))],
                      "Cohen's benchmarks; the right effect size is the smallest one that matters in your field."))
    ns = np.unique(np.round(np.geomspace(max(nmin, 3), max(n_ref * 2.5, 20), 60)).astype(int))
    curve = [pf(ef(es_use), int(v)) for v in ns]
    charts = [chart("Power as a function of the sample size", [float(v) for v in ns], [("Power", curve), ("Target / achieved", [pw] * len(ns))], "sample size " + n_label, "power", "line")]
    summary = [stat("Test", name)]
    if solve == "n":
        summary += [stat("Required n", int(n_res)), stat("Achieved power", pw)]
    elif solve == "power":
        summary += [stat("Power", pw)]
    else:
        summary += [stat("Detectable effect", es_use)]
    notes = ["Plan with a conservative effect size and allow for dropout: divide the required n by the expected retention rate.",
             "Post-hoc power computed from an observed effect carries no information beyond the p-value; use this tool before collecting data."]
    return result("stat_power", "Power and sample size", int(n_use) if np.isfinite(n_use) else 0, summary, tabs, charts, notes)
