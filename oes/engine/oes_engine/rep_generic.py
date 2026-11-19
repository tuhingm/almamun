"""Generic report for every model that has no dedicated reporter yet.

It arranges the engine's own result (summary statistics, tables, charts, notes) into the Report > Step structure so that the
same page, copy buttons, figure downloads and ZIP work for all 100 models. It computes NO new statistic and changes no number:
every table cell and plotted value is the engine's. What it does not have, compared with the dedicated OLS and ANOVA reports,
is a researcher-ordered diagnostic workflow, extra graphs and model-specific interpretation; the report says so.
"""
import math

from . import rep_tables as T
from . import rep_text as X
from .rep_model import Figure, Paragraph, Report, Step
from .rep_tables import fnum, make_table
from .stat_core import thin

MAX_ROWS = 60
VALIDATED = ("numbers are the engine's own output, reproduced unchanged and compared cell by cell (test_generic_reports); the statistics themselves are covered "
             "by the engine's existing tests and benchmarks; this layer adds no statistic")


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt_col(vals):
    nums = [abs(v) for v in vals if _isnum(v) and v != 0 and not float(v).is_integer()]
    d = T.decimals_for(nums, 2, 6) if nums else 0
    return lambda v: ("" if v is None else fnum(v, d) if _isnum(v) and not float(v).is_integer() else T.fint(v) if _isnum(v) else str(v))


def _is_test_table(t):
    cols = [c.lower() for c in t["columns"]]
    return any(c in ("p-value", "p", "prob", "p value") for c in cols) and (any(w in t["title"].lower() for w in ("test", "diagnostic")) or cols[0] in ("test", "null hypothesis"))


def _table(key, t):
    cols = t["columns"]
    rows = t["rows"][:MAX_ROWS]
    fmts = [_fmt_col([r[j] for r in rows if j < len(r)]) for j in range(len(cols))]
    low = [c.lower() for c in cols]
    pj = next((j for j, c in enumerate(low) if c in ("p-value", "p", "prob")), None)
    cj = next((j for j, c in enumerate(low) if c in ("coef.", "coefficient", "estimate")), None)
    out = []
    for r in rows:
        cells = [fmts[j](r[j]) if j < len(r) else "" for j in range(len(cols))]
        if pj is not None and cj is not None and _isnum(r[pj]) and cj < len(r) and _isnum(r[cj]):
            cells[cj] += T.stars(r[pj])
        out.append(cells)
    notes = ([t["note"]] if t.get("note") else [])
    if len(t["rows"]) > MAX_ROWS:
        notes.append(f"The first {MAX_ROWS} of {len(t['rows'])} rows are shown; the CSV download of the standard result holds all rows.")
    if pj is not None and cj is not None:
        notes.append(T.STAR_NOTE)
    return make_table(key, t["title"], list(cols), out, notes, raw={"columns": list(cols), "rows": rows})


def _figure(i, c):
    x = c["x"]
    series = c["series"][:6]
    numeric = all(_isnum(v) for v in x) if x else False
    kind = c.get("kind", "line")
    n = len(x)
    pos = [float(v) for v in x] if numeric else [float(j) for j in range(n)]
    xt = None
    if not numeric and n:
        idx = sorted(set(int(round(j)) for j in range(0, n, max(1, (n - 1) // 6 or 1)))) + ([n - 1] if n > 1 else [])
        xt = [[float(j), str(x[j])] for j in sorted(set(idx))]
    layers = []
    if c.get("band"):
        layers.append({"type": "band", "x": pos, "lo": c["band"]["lo"], "hi": c["band"]["hi"]})
    ns = len(series)
    for k, s in enumerate(series):
        y = s["y"]
        if kind == "bar":
            step = (pos[1] - pos[0]) if n > 1 else 1.0
            w = 0.8 * step / ns
            layers.append({"type": "bar", "x": [p + (k - (ns - 1) / 2) * w for p in pos], "y": y, "width": w, "fill": ["#8a8a8a", "#4d4d4d", "#c8c8c8", "#6e6e6e", "#a5a5a5", "#333333"][k % 6]})
        elif kind == "scatter":
            px, py = thin(pos, y, 400)
            layers.append({"type": "scatter", "x": px, "y": py, "k": k, "hollow": k > 0, "r": 1.7})
        else:
            layers.append({"type": "line", "x": pos, "y": y, "k": k, "lw": 0.9})
        if ns > 1:
            layers[-1]["label"] = s["name"]
    panel = {"xlabel": c.get("xlabel", ""), "ylabel": c.get("ylabel", ""), "layers": layers, "legend": ns > 1, "zero": kind == "bar"}
    if xt:
        panel["xticks"] = xt
    note = "Drawn from the values returned by the engine." + (f" Only the first 6 of {len(c['series'])} series are drawn." if len(c["series"]) > 6 else "")
    return Figure(f"f_{i}", c["title"], {"ncols": 1, "panels": [panel]}, note, "Source: author's calculations.", 6.0, 3.2)


def _coef_figure(i, t):
    """Coefficient plot from a table that already has an estimate and its confidence interval (the engine's numbers, nothing recomputed)."""
    low = [c.lower() for c in t["columns"]]
    ci = next((j for j, c in enumerate(low) if c in ("coef.", "coefficient", "estimate")), None)
    lo = next((j for j, c in enumerate(low) if "ci low" in c or c in ("lower", "ci lower")), None)
    hi = next((j for j, c in enumerate(low) if "ci high" in c or c in ("upper", "ci upper")), None)
    if None in (ci, lo, hi):
        return None
    rows = [r for r in t["rows"] if len(r) > max(ci, lo, hi) and all(_isnum(r[j]) for j in (ci, lo, hi)) and str(r[0]).lower() not in ("const", "constant", "intercept")][:20]
    if len(rows) < 2:
        return None
    m = len(rows)
    ypos = [float(m - 1 - k) for k in range(m)]
    panel = {"xlabel": "Estimate with confidence interval", "yticks": [[ypos[k], str(rows[k][0])[:24]] for k in range(m)], "ylim": [-0.7, m - 0.3],
             "ml": 20 + 5.2 * min(24, max(len(str(r[0])) for r in rows)),
             "layers": [{"type": "errh", "y": ypos, "x": [float(r[ci]) for r in rows], "lo": [float(r[lo]) for r in rows], "hi": [float(r[hi]) for r in rows]}, {"type": "vline", "x": 0.0}]}
    return Figure(f"f_coef{i}", f"Estimates with confidence intervals: {t['title']}", {"ncols": 1, "panels": [panel]},
                  "Points and bars are the estimates and confidence intervals reported in the corresponding table (the intercept is omitted); the dashed line marks zero.",
                  "Source: author's calculations.", 5.2, max(2.2, 0.45 * m + 1.2))


def build_from_result(out):
    from . import __version__
    mid = out.get("model", "model")
    rep = Report(mid, out.get("title", mid), references={})
    rep.meta = dict(level="generic", nobs=out.get("nobs"), engine=__version__)
    summ = [s for s in out.get("summary", []) if s.get("value") is not None]

    s1 = Step("summary", "Summary of the run", "The headline numbers of the estimation as returned by the engine.", validated=VALIDATED)
    if summ:
        s1.tables.append(make_table("t_summary", "Summary statistics", ["Statistic", "Value"],
                                    [[s["label"], (fnum(s["value"], 4) if _isnum(s["value"]) and not float(s["value"]).is_integer() else T.fint(s["value"]) if _isnum(s["value"]) else str(s["value"]))] for s in summ],
                                    raw={"rows": summ}))
        pick = summ[:6]
        facts, parts = {}, []
        for i, s in enumerate(pick):
            v = s["value"]
            if _isnum(v):
                facts[f"v{i}"] = float(v)
                parts.append(f"{s['label']} = " + (f"{v:.4f}" if not float(v).is_integer() else f"{int(v)}"))
            else:
                facts[f"v{i}"] = str(v)
                parts.append(f"{s['label']} = {v}")
        s1.paragraphs.append(Paragraph("result", f"The run uses {out.get('nobs')} observations. Reported statistics: " + "; ".join(parts) + ".", dict(facts, n=out.get("nobs"))))
    else:
        s1.paragraphs.append(Paragraph("result", f"The run uses {out.get('nobs')} observations.", {"n": out.get("nobs")}))
    rep.steps.append(s1)

    tabs = out.get("tables", [])
    res_t = [(i, t) for i, t in enumerate(tabs) if not _is_test_table(t)]
    tst_t = [(i, t) for i, t in enumerate(tabs) if _is_test_table(t)]
    if res_t:
        s2 = Step("results", "Estimates and results", "The result tables of the model as returned by the engine.", validated=VALIDATED)
        for i, t in res_t:
            s2.tables.append(_table(f"t_{i + 1}", t))
        rep.steps.append(s2)
    if tst_t:
        s3 = Step("tests", "Tests and diagnostics", "Specification and diagnostic tests as returned by the engine; the null hypothesis of each test is in the table.", validated=VALIDATED)
        for i, t in tst_t:
            s3.tables.append(_table(f"t_{i + 1}", t))
            cols = [c.lower() for c in t["columns"]]
            pj = next(j for j, c in enumerate(cols) if c in ("p-value", "p", "prob", "p value"))
            rows = [r for r in t["rows"] if len(r) > pj and _isnum(r[pj])]
            rej = [str(r[0]) for r in rows if r[pj] < 0.05]
            if rows:
                if rej:
                    s3.paragraphs.append(Paragraph("warning", f"In '{t['title']}', the null hypothesis is rejected at the 5% level for {len(rej)} of {len(rows)} tests: " + "; ".join(rej) +
                                                   ". What a rejection means depends on the null hypothesis of each test (see the table); check it before relying on the estimates.", dict(rejected=len(rej), total=len(rows))))
                else:
                    s3.paragraphs.append(Paragraph("result", f"In '{t['title']}', none of the {len(rows)} tests rejects its null hypothesis at the 5% level. This does not prove the assumptions hold.", dict(total=len(rows))))
        rep.steps.append(s3)
    ch = out.get("charts", [])
    if ch:
        s4 = Step("figures", "Figures", "The charts of the model as returned by the engine, redrawn in publication style.", validated=VALIDATED)
        for i, c in enumerate(ch, 1):
            if c.get("x") and c.get("series"):
                s4.figures.append(_figure(i, c))
        if s4.figures:
            rep.steps.append(s4)
    cf = [f for f in (_coef_figure(i + 1, t) for i, t in enumerate(tabs)) if f]
    if cf:
        s5 = Step("coefplots", "Coefficient plots", "Estimates and confidence intervals of the result tables, drawn from the numbers the engine reported.", validated=VALIDATED)
        s5.figures = cf
        rep.steps.append(s5)

    rep.methods.append(Paragraph("method", f"{out.get('title', mid)} was estimated with the Open Econometric Systems engine (version {__version__}) using {out.get('nobs')} observations. "
                                 "The tables and figures of this report are the engine's output, reproduced without change.", {"n": out.get("nobs"), "ver": __version__}))
    for n in out.get("notes", []):
        rep.limitations.append(Paragraph("note", str(n)))
    rep.limitations.append(Paragraph("note", "This model has no dedicated step-by-step report yet: the workflow of diagnostics, additional graphs and model-specific interpretation available for OLS and one-way ANOVA is not implemented here, and no interpretation of the estimates is generated."))
    return rep.finalize()
