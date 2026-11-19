"""Category G of the Statistics Division: meta-analysis.

Fixed-effect (inverse-variance) and random-effects pooling of study results. Effects can be supplied directly (an effect size with a
standard error or variance) or computed from raw study data: log risk ratio, log odds ratio and risk difference from event counts,
or Hedges' g from means and standard deviations. Between-study variance estimators: DerSimonian-Laird, Hedges, Paule-Mandel, ML
and REML, with the Q-profile interval, Wald or Hartung-Knapp-Sidik-Jonkman inference, prediction interval, heterogeneity measures,
leave-one-out analysis and the Egger and Begg tests of small-study effects."""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from .core import OESError, as_list, chart, distinct, f, float_option, need, result, stat, table, tests_table
from .stat_core import clean_frame

_MEASURES = {"lrr": "log risk ratio", "lor": "log odds ratio", "rd": "risk difference", "smd": "standardised mean difference (Hedges g)", "generic": "effect size"}


def _alpha(spec):
    a = float_option(spec, "alpha", 0.05, "Significance level")
    if not 0 < a < 0.5:
        raise OESError("The significance level must be between 0 and 0.5.")
    return a


# ---------------------------------------------------------------------------------------------------------------
# effect sizes
# ---------------------------------------------------------------------------------------------------------------
def effect_sizes(df, spec):
    """Return (labels, y, v, measure, notes) from the chosen input mode."""
    mode = spec.get("input") or "generic"
    lab = spec.get("study") or None
    notes = []
    if mode == "generic":
        ycol = need(spec, "yi", "effect size column")
        se, vc = spec.get("sei") or None, spec.get("vi") or None
        if not se and not vc:
            raise OESError("Choose the standard error or the variance column.")
        cols = [ycol] + ([se] if se else []) + ([vc] if vc and not se else [])
        distinct(ycol, se, vc if not se else None)
        d, dropped = clean_frame(df, numeric=cols, keep=[lab] if lab else [])
        y = d[ycol].to_numpy(float)
        v = d[se].to_numpy(float) ** 2 if se else d[vc].to_numpy(float)
        measure = "generic"
    elif mode == "binary":
        c = [need(spec, k, lbl) for k, lbl in (("ev1", "events in the treatment group"), ("n1", "size of the treatment group"), ("ev2", "events in the control group"), ("n2", "size of the control group"))]
        distinct(*c)
        measure = spec.get("measure") or "lrr"
        if measure not in ("lrr", "lor", "rd"):
            raise OESError("Choose log risk ratio, log odds ratio or risk difference.")
        d, dropped = clean_frame(df, numeric=c, keep=[lab] if lab else [])
        a, n1, cc, n2 = (d[k].to_numpy(float) for k in c)
        if (a < 0).any() or (cc < 0).any() or (a > n1).any() or (cc > n2).any() or (n1 <= 0).any() or (n2 <= 0).any():
            raise OESError("Event counts must lie between 0 and the group size, and group sizes must be positive.")
        b, dd = n1 - a, n2 - cc
        zero = (a == 0) | (b == 0) | (cc == 0) | (dd == 0)
        if zero.any() and measure != "rd":
            notes.append(f"{int(zero.sum())} study(ies) have a zero cell; 0.5 was added to every cell of those studies (the usual continuity correction).")
        elif zero.any():
            pass
        k_ = np.where(zero & (measure != "rd"), 0.5, 0.0)
        a2, b2, c2, d2 = a + k_, b + k_, cc + k_, dd + k_
        n1c, n2c = a2 + b2, c2 + d2
        if measure == "lrr":
            y = np.log((a2 / n1c) / (c2 / n2c))
            v = 1 / a2 - 1 / n1c + 1 / c2 - 1 / n2c
        elif measure == "lor":
            y = np.log(a2 * d2 / (b2 * c2))
            v = 1 / a2 + 1 / b2 + 1 / c2 + 1 / d2
        else:
            y = a / n1 - cc / n2
            v = a * b / n1 ** 3 + cc * dd / n2 ** 3
            if (v <= 0).any():
                raise OESError("The risk difference has zero variance in some study (no events or all events in both groups); use a ratio measure.")
    elif mode == "means":
        c = [need(spec, k, lbl) for k, lbl in (("m1", "mean of the treatment group"), ("sd1", "SD of the treatment group"), ("n1", "size of the treatment group"),
                                               ("m2", "mean of the control group"), ("sd2", "SD of the control group"), ("n2", "size of the control group"))]
        distinct(*c)
        measure = "smd"
        d, dropped = clean_frame(df, numeric=c, keep=[lab] if lab else [])
        m1, s1, n1, m2, s2, n2 = (d[k].to_numpy(float) for k in c)
        if (s1 <= 0).any() or (s2 <= 0).any() or (n1 < 2).any() or (n2 < 2).any():
            raise OESError("Standard deviations must be positive and group sizes at least 2.")
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
        y = J * (m1 - m2) / sp
        v = 1 / n1 + 1 / n2 + y ** 2 / (2 * (n1 + n2))
    else:
        raise OESError("Unknown input type.")
    if len(d) < 2:
        raise OESError("At least two studies are needed.")
    if (v <= 0).any() or not np.all(np.isfinite(v)):
        raise OESError("Every study needs a positive, finite variance (standard error).")
    labels = d[lab].astype(str).tolist() if lab else [f"Study {i + 1}" for i in range(len(d))]
    if dropped:
        notes.append(f"{dropped} row(s) with missing values were dropped.")
    return labels, np.asarray(y, float), np.asarray(v, float), measure, notes


# ---------------------------------------------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------------------------------------------
def _pool(y, w):
    sw = w.sum()
    th = float((w * y).sum() / sw)
    return th, float(1 / np.sqrt(sw))


def _Q(y, v, tau2):
    w = 1 / (v + tau2)
    th = (w * y).sum() / w.sum()
    return float((w * (y - th) ** 2).sum())


def tau2_estimate(y, v, method):
    k = len(y)
    w = 1 / v
    q = _Q(y, v, 0.0)
    if method == "fixed":
        return 0.0
    if method == "dl":
        c = w.sum() - (w ** 2).sum() / w.sum()
        return max(0.0, (q - (k - 1)) / c)
    if method == "he":
        return max(0.0, float(np.var(y, ddof=1) - v.mean()))
    upper = max(10 * float(np.var(y, ddof=1)), 10 * float(v.max()), 1e-3)
    if method == "pm":
        if q <= k - 1:
            return 0.0
        g = lambda t: _Q(y, v, t) - (k - 1)
        while g(upper) > 0:
            upper *= 4
            if upper > 1e12:
                raise OESError("The Paule-Mandel estimator did not converge.")
        return float(optimize.brentq(g, 0.0, upper, xtol=1e-12, rtol=1e-12))
    if method in ("ml", "reml"):
        def nll(t):
            ww = 1 / (v + t)
            th = (ww * y).sum() / ww.sum()
            val = np.log(v + t).sum() + (ww * (y - th) ** 2).sum()
            if method == "reml":
                val += np.log(ww.sum())
            return val
        r = optimize.minimize_scalar(nll, bounds=(0.0, upper), method="bounded", options=dict(xatol=1e-12))
        cands = [(nll(0.0), 0.0), (r.fun, float(r.x))]
        # polish with a Fisher-scoring style fixed-point iteration from the best candidate
        t = min(cands)[1]
        for _ in range(200):
            ww = 1 / (v + t)
            th = (ww * y).sum() / ww.sum()
            num = (ww ** 2 * ((y - th) ** 2 - v)).sum()
            den = (ww ** 2).sum()
            if method == "reml":
                num += (ww ** 2).sum() / ww.sum()
            tn = max(0.0, t + num / den) if den > 0 else t
            if abs(tn - t) < 1e-13 * (1 + t):
                t = tn
                break
            t = tn
        return float(t) if nll(t) <= min(cands)[0] + 1e-10 else min(cands)[1]
    raise OESError("Unknown between-study variance estimator.")


def _q_profile(y, v, alpha):
    k = len(y)
    hi_q, lo_q = stats.chi2.ppf(1 - alpha / 2, k - 1), stats.chi2.ppf(alpha / 2, k - 1)
    q0 = _Q(y, v, 0.0)
    up = max(100 * float(np.var(y, ddof=1)), 100 * float(v.max()), 1.0)

    def solve(target):
        if q0 < target:
            return 0.0
        g = lambda t: _Q(y, v, t) - target
        u = up
        while g(u) > 0:
            u *= 4
            if u > 1e14:
                return float("inf")
        return float(optimize.brentq(g, 0.0, u, xtol=1e-12, rtol=1e-12))

    return solve(hi_q), solve(lo_q)


def _fit(y, v, method, ci):
    """Pooled estimate under the chosen model. method = 'fixed' or a tau^2 estimator."""
    k = len(y)
    t2 = tau2_estimate(y, v, method)
    w = 1 / (v + t2)
    th, se = _pool(y, w)
    q_star = float((w * (y - th) ** 2).sum() / (k - 1)) if k > 1 else 1.0
    if method != "fixed" and ci in ("hksj", "hksj_mod"):
        factor = q_star if ci == "hksj" else max(1.0, q_star)
        se = se * np.sqrt(factor)
        return dict(theta=th, se=se, tau2=t2, w=w, dist="t", df=k - 1, hk=q_star)
    return dict(theta=th, se=se, tau2=t2, w=w, dist="z", df=None, hk=q_star)


def _ci(fit, alpha):
    c = stats.t.ppf(1 - alpha / 2, fit["df"]) if fit["dist"] == "t" else stats.norm.ppf(1 - alpha / 2)
    stat_ = fit["theta"] / fit["se"]
    p = 2 * (stats.t.sf(abs(stat_), fit["df"]) if fit["dist"] == "t" else stats.norm.sf(abs(stat_)))
    return fit["theta"] - c * fit["se"], fit["theta"] + c * fit["se"], stat_, p


def _tr(est, measure):
    return float(np.exp(est)) if measure in ("lrr", "lor") else float(est)


def _analyse(df, spec, random):
    alpha = _alpha(spec)
    labels, y, v, measure, notes = effect_sizes(df, spec)
    k = len(y)
    method = (spec.get("tau_method") or "reml") if random else "fixed"
    ci = (spec.get("ci_method") or "wald") if random else "wald"
    if method not in ("fixed", "dl", "he", "pm", "ml", "reml"):
        raise OESError("Unknown between-study variance estimator.")
    fit = _fit(y, v, method, ci)
    lo, hi, zt, p = _ci(fit, alpha)
    # heterogeneity (always from the fixed-effect weights)
    wf = 1 / v
    q = _Q(y, v, 0.0)
    dfq = k - 1
    pq = float(stats.chi2.sf(q, dfq))
    i2 = max(0.0, (q - dfq) / q) if q > 0 else 0.0
    h2 = q / dfq
    t2_dl = tau2_estimate(y, v, "dl")
    s2 = dfq * wf.sum() / (wf.sum() ** 2 - (wf ** 2).sum())
    lab = f"{int(round((1 - alpha) * 100))}% CI"
    log_scale = measure in ("lrr", "lor")
    cols = ["Model", "Estimate", "SE", "z" if fit["dist"] == "z" else "t", "p-value", f"{lab} low", "CI high"] + (["exp(Estimate)", "exp(CI low)", "exp(CI high)"] if log_scale else [])
    mname = "Random-effects" if random else "Fixed-effect (common effect)"
    row = [mname + (f" ({method.upper()}, {'Hartung-Knapp-Sidik-Jonkman' if ci != 'wald' else 'Wald'})" if random else ""), f(fit["theta"], 5), f(fit["se"], 5), f(zt, 4), f(p, 6), f(lo, 5), f(hi, 5)]
    if log_scale:
        row += [f(_tr(fit["theta"], measure), 5), f(_tr(lo, measure), 5), f(_tr(hi, measure), 5)]
    tabs = [table("Pooled effect", cols, [row], f"Effect measure: {_MEASURES[measure]}." + (" Ratio measures are pooled on the log scale and shown as ratios in the last columns." if log_scale else ""))]
    # study table
    se_i = np.sqrt(v)
    z = stats.norm.ppf(1 - alpha / 2)
    wpct = 100 * fit["w"] / fit["w"].sum()
    srow = []
    for i in range(k):
        r_ = [labels[i], f(y[i], 5), f(se_i[i], 5), f(y[i] - z * se_i[i], 5), f(y[i] + z * se_i[i], 5), f(wpct[i], 2)]
        if log_scale:
            r_ += [f(_tr(y[i], measure), 4)]
        srow.append(r_)
    tabs.append(table("Study results", ["Study", "Effect", "SE", f"{lab} low", "CI high", "Weight (%)"] + (["exp(Effect)"] if log_scale else []), srow,
                      "Weights are inverse-variance" + (" including the between-study variance." if random else ".")))
    # heterogeneity table
    het = [["Cochran's Q", f(q, 4), f"df = {dfq}", f(pq, 6)], ["I-squared (share of variability from heterogeneity)", f(100 * i2, 2), "%", None], ["H-squared (Q / df)", f(h2, 4), None, None]]
    if random or True:
        tlo, thi = _q_profile(y, v, alpha)
        het.append([f"tau-squared ({method.upper() if random else 'DL'})", f(fit["tau2"] if random else t2_dl, 6), None, None])
        het.append(["tau (between-study SD)", f(np.sqrt(fit["tau2"] if random else t2_dl), 5), None, None])
        het.append([f"Q-profile {lab} for tau-squared", f(tlo, 6), f(thi, 6) if np.isfinite(thi) else "unbounded", None])
        i2lo = tlo / (tlo + s2)
        i2hi = thi / (thi + s2) if np.isfinite(thi) else 1.0
        het.append(["I-squared from tau-squared, tau-squared / (tau-squared + typical variance)", f(100 * (fit["tau2"] if random else t2_dl) / ((fit["tau2"] if random else t2_dl) + s2), 2), "%", None])
        het.append([f"I-squared from the tau-squared interval ({lab})", f(100 * i2lo, 2), f(100 * i2hi, 2), "%"])
    tabs.append(table("Heterogeneity", ["Measure", "Value", "Detail / upper", "p-value"], het,
                      "Q tests the null hypothesis that all studies share one true effect; it has low power with few studies. I-squared of about 25%, 50% and 75% is conventionally low, moderate and high."))
    notes_extra = []
    summary = [stat("Studies", k), stat("Pooled effect", fit["theta"]), stat(f"{lab} low", lo), stat("CI high", hi), stat("p-value", p, 6), stat("I-squared (%)", 100 * i2, 2)]
    if log_scale:
        summary.insert(2, stat("Pooled ratio", _tr(fit["theta"], measure)))
    if random:
        summary.append(stat("tau-squared", fit["tau2"], 5))
        if k >= 3:
            tc = stats.t.ppf(1 - alpha / 2, k - 2)
            half = tc * np.sqrt(fit["tau2"] + (fit["se"] if fit["dist"] == "z" else float(np.sqrt(1 / fit["w"].sum()))) ** 2)
            pl, ph = fit["theta"] - half, fit["theta"] + half
            prow = [f"{int(round((1 - alpha) * 100))}% prediction interval for a new study", f(pl, 5), f(ph, 5)]
            if log_scale:
                prow += [f(_tr(pl, measure), 4), f(_tr(ph, measure), 4)]
            tabs.append(table("Prediction interval", ["Quantity", "Low", "High"] + (["exp(Low)", "exp(High)"] if log_scale else []), [prow],
                              "Range expected to contain the true effect of a new, similar study (Higgins, Thompson and Spiegelhalter 2009); it is wider than the confidence interval whenever tau-squared is positive."))
        else:
            notes_extra = ["A prediction interval needs at least three studies."]
    # leave-one-out
    if k >= 3:
        loo = []
        for i in range(k):
            m = np.arange(k) != i
            try:
                fi = _fit(y[m], v[m], method, ci)
                l_, h_, _, p_ = _ci(fi, alpha)
                loo.append([labels[i], f(fi["theta"], 5), f(l_, 5), f(h_, 5), f(fi["tau2"], 5) if random else None] + ([f(_tr(fi["theta"], measure), 4)] if log_scale else []))
            except OESError:
                continue
        tabs.append(table("Leave-one-out analysis (pooled effect without each study)", ["Omitted study", "Estimate", f"{lab} low", "CI high", "tau-squared" if random else "(fixed-effect: none)"] + (["exp(Estimate)"] if log_scale else []), loo,
                          "A large change when one study is omitted flags an influential study."))
    # small-study effects
    sm_rows = []
    if k >= 3:
        s_i = np.sqrt(v)
        Xe = np.column_stack([np.ones(k), 1 / s_i])
        ye = y / s_i
        bh = np.linalg.lstsq(Xe, ye, rcond=None)[0]
        res = ye - Xe @ bh
        s2e = (res ** 2).sum() / (k - 2) if k > 2 else float("nan")
        cov = s2e * np.linalg.inv(Xe.T @ Xe)
        te = bh[0] / np.sqrt(cov[0, 0])
        pe = 2 * stats.t.sf(abs(te), k - 2)
        sm_rows.append(["Egger regression test (intercept)", f(bh[0], 4), f(te, 4), f(pe, 4)])
        thf, _ = _pool(y, 1 / v)
        vstar = v - 1 / (1 / v).sum()
        if (vstar > 0).all():
            ystar = (y - thf) / np.sqrt(vstar)
            kt = stats.kendalltau(ystar, v)
            sm_rows.append(["Begg rank correlation (Kendall tau)", f(kt.statistic, 4), None, f(kt.pvalue, 4)])
        tabs.append(tests_table(sm_rows, "Small-study effects (funnel-plot asymmetry)", "These tests have little power with fewer than about 10 studies, and asymmetry can come from heterogeneity as well as publication bias."))
    # charts
    charts = [chart("Study effects with the pooled estimate", labels, [("Effect", y), ("Pooled estimate", [fit["theta"]] * k)], "study", "effect", "bar")]
    pad = [None] * k
    charts.append(chart("Funnel plot (effect against standard error)", [float(a) for a in y] + [fit["theta"], fit["theta"]],
                        [("Studies", [float(a) for a in np.sqrt(v)] + [None, None]), ("Pooled estimate", pad + [0.0, float(np.sqrt(v).max()) * 1.05])], "effect", "standard error", "scatter"))
    out = result("stat_meta_random" if random else "stat_meta_fixed", "Random-effects meta-analysis" if random else "Fixed-effect meta-analysis", k, summary, tabs, charts, notes + (notes_extra if k < 3 else []))
    out["forest"] = dict(labels=labels, est=[f(a, 5) for a in y], lo=[f(a - z * s, 5) for a, s in zip(y, se_i)], hi=[f(a + z * s, 5) for a, s in zip(y, se_i)], weight=[f(a, 2) for a in wpct],
                         pooled=dict(label=mname, est=f(fit["theta"], 5), lo=f(lo, 5), hi=f(hi, 5)), log=bool(log_scale), measure=_MEASURES[measure], null=0.0)
    if random and k >= 3:
        out["forest"]["pred"] = [f(pl, 5), f(ph, 5)]
    if random and fit["tau2"] == 0:
        out["notes"].append("The between-study variance is estimated as zero, so the random-effects and fixed-effect results coincide; this does not prove the studies are homogeneous.")
    if k < 5:
        out["notes"].append(f"Only {k} studies: estimates of between-study variance are very imprecise" + (", and Wald intervals can be too narrow; consider the Hartung-Knapp-Sidik-Jonkman method." if random and ci == "wald" else "."))
    return out


def meta_fixed(df, spec):
    out = _analyse(df, spec, False)
    out["notes"].append("The fixed-effect model assumes one common true effect; if the studies differ in population, dose or design, a random-effects model answers a more realistic question.")
    return out


def meta_random(df, spec):
    return _analyse(df, spec, True)
