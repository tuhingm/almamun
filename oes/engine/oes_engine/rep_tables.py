"""Publication-style tables: number formatting, stars, builders and the four renderers (text, HTML, LaTeX, CSV).

Conventions (AER / Stata `esttab` / APA): coefficients with stars (*** p<0.01, ** p<0.05, * p<0.10), standard errors in
parentheses under the coefficient, a rule structure of top rule, header rule and bottom rule (booktabs), the sample
size and fit statistics below a mid rule, and a note line under the table.
"""
import csv
import html
import io
import math

import numpy as np

STAR_NOTE = "*** p<0.01, ** p<0.05, * p<0.10."
MINUS = "−"


def isnum(x):
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fnum(x, d=3, minus=True):
    """Fixed decimals; the typographic minus sign; blank for missing values."""
    if not isnum(x):
        return ""
    s = f"{float(x):.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", MINUS) if minus else s


def fint(x):
    return f"{int(round(float(x))):,}" if isnum(x) else ""


def fp(p, d=3):
    """p-value as printed in a table: 0.012, or <0.001."""
    if not isnum(p):
        return ""
    return f"<{10 ** -d:.{d}f}" if p < 10 ** -d else f"{p:.{d}f}"


def fp_text(p, d=3):
    """p-value inside a sentence: p = 0.012, p < 0.001."""
    if not isnum(p):
        return "p = n.a."
    return f"p < {10 ** -d:.{d}f}" if p < 10 ** -d else f"p = {p:.{d}f}"


def stars(p):
    if not isnum(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def decimals_for(values, lo=2, hi=6, sig=3):
    """Common number of decimals so that the smallest non-zero magnitude keeps `sig` significant digits."""
    v = [abs(float(x)) for x in values if isnum(x) and float(x) != 0]
    if not v:
        return lo
    m = min(v)
    return int(min(hi, max(lo, sig - 1 - math.floor(math.log10(m)))))


def _row(cells, kind="body"):
    return {"cells": [str(c) for c in cells], "kind": kind}


def make_table(key, title, header, rows, notes=(), spans=None, align=None, raw=None):
    from .rep_model import Table
    header = [list(h) for h in (header if header and isinstance(header[0], (list, tuple)) else [header])]
    ncol = len(header[-1])
    if align is None:
        align = ["l"] + ["r"] * (ncol - 1)
    rr = [r if isinstance(r, dict) else _row(r) for r in rows]
    return Table(key=key, title=title, header=header, rows=rr, notes=list(notes), spans=spans, align=align, raw=raw)


# ------------------------------------------------------------------------------------------------- builders
def regression_table(key, title, columns, names, labels=None, stat_rows=(), dep_label=None, notes=(), show="se", level=0.95):
    """Column-wise regression layout.

    columns   list of dicts: {"label": "(1)", "sub": "OLS", "params": Series-like, "se": ..., "p": ..., "stats": {name: value}}
    names     order of the variables (rows); a variable missing from a column is left blank
    stat_rows list of (label, key, kind) with kind in "int" | "num3" | "num2" | "p"
    show      "se" (parentheses under the estimate), "t" or "p"
    """
    labels = labels or {}
    allb = [columns[j]["params"][n] for j in range(len(columns)) for n in names if n in columns[j]["params"]]
    d = decimals_for(allb)
    header = [["", *[c["label"] for c in columns]]]
    if any(c.get("sub") for c in columns):
        header.append(["", *[c.get("sub", "") for c in columns]])
    rows = []
    for n in names:
        top, below = [labels.get(n, n)], [""]
        for c in columns:
            if n in c["params"]:
                p = c["p"][n]
                top.append(fnum(c["params"][n], d) + stars(p))
                if show == "t":
                    below.append("[" + fnum(c["params"][n] / c["se"][n], 2) + "]")
                elif show == "p":
                    below.append("[" + fp(p) + "]")
                else:
                    below.append("(" + fnum(c["se"][n], d) + ")")
            else:
                top.append("")
                below.append("")
        rows.append(_row(top))
        rows.append(_row(below, "se"))
    first = True
    for lab, k, kind in stat_rows:
        cells = [lab]
        for c in columns:
            v = c.get("stats", {}).get(k)
            cells.append(fint(v) if kind == "int" else fp(v) if kind == "p" else fnum(v, 2 if kind == "num2" else 3) if isnum(v) else (str(v) if v is not None else ""))
        rows.append(_row(cells, "stat" if first else "body"))
        first = False
    dep = [dep_label] if dep_label else []
    nts = list(notes) + [STAR_NOTE] if show == "se" else list(notes) + [STAR_NOTE]
    se_txt = {"se": "Standard errors in parentheses.", "t": "t-statistics in brackets.", "p": "p-values in brackets."}[show]
    raw = {"names": list(names), "columns": [{"label": c["label"], "params": {n: float(c["params"][n]) for n in names if n in c["params"]},
                                              "se": {n: float(c["se"][n]) for n in names if n in c["se"]},
                                              "p": {n: float(c["p"][n]) for n in names if n in c["p"]}} for c in columns]}
    return make_table(key, title, header, rows, dep + [se_txt] + nts, align=["l"] + ["c"] * len(columns), raw=raw)


def descriptives_table(key, df, names, title="Descriptive statistics", labels=None, notes=()):
    from scipy import stats as st
    labels = labels or {}
    head = ["Variable", "N", "Mean", "Std. dev.", "Min", "P25", "Median", "P75", "Max", "Skewness", "Kurtosis"]
    rows, raw = [], []
    for n in names:
        x = df[n].astype(float).dropna().to_numpy()
        q = np.percentile(x, [25, 50, 75])
        vals = [len(x), x.mean(), x.std(ddof=1), x.min(), q[0], q[1], q[2], x.max(), st.skew(x, bias=False), st.kurtosis(x, fisher=False, bias=False)]
        d = decimals_for([vals[1], vals[2]], 2, 4)
        rows.append([labels.get(n, n), fint(vals[0])] + [fnum(v, d) for v in vals[1:8]] + [fnum(vals[8], 2), fnum(vals[9], 2)])
        raw.append(dict(zip(["n", "mean", "sd", "min", "p25", "median", "p75", "max", "skew", "kurtosis"], [float(v) for v in vals]), name=n))
    nt = ["Std. dev. uses n-1. Skewness and kurtosis are the bias-adjusted sample moments (the normal distribution has kurtosis 3)."]
    return make_table(key, title, head, rows, nt + list(notes), raw={"rows": raw})


def correlation_table(key, df, names, title="Correlation matrix", labels=None):
    from scipy import stats as st
    labels = labels or {}
    k = len(names)
    rows, raw = [], {}
    for i in range(k):
        cells = [labels.get(names[i], names[i])]
        for j in range(k):
            if j > i:
                cells.append("")
            elif j == i:
                cells.append("1.000")
            else:
                r, p = st.pearsonr(df[names[i]].astype(float), df[names[j]].astype(float))
                cells.append(fnum(r, 3) + stars(p))
                raw[f"{names[i]}|{names[j]}"] = [float(r), float(p)]
        rows.append(cells)
    return make_table(key, title, ["", *[labels.get(n, n) for n in names]], rows,
                      ["Pearson correlations, lower triangle; N = " + fint(len(df)) + ".", STAR_NOTE], align=["l"] + ["r"] * k, raw=raw)


def tests_table(key, title, tests, notes=()):
    """Specification-test table: statistic, p-value, decision at 5 per cent."""
    head = ["Test", "Statistic", "d.f.", "p-value", "Null hypothesis", "Decision (5%)"]
    rows = []
    for t in tests:
        dec = "n.a." if t.reject is None else ("Reject" if t.reject else "Do not reject")
        rows.append([t.name, fnum(t.statistic, 3), t.df or "", fp(t.p_value), t.null, dec])
    return make_table(key, title, head, rows, list(notes), align=["l", "r", "r", "r", "l", "l"])


# ------------------------------------------------------------------------------------------------- renderers
def _strip(s):
    return str(s)


def to_text(t):
    """Fixed-width plain text, ready to paste into any editor or e-mail."""
    ncol = len(t.header[-1])
    grid = [[_strip(c) for c in r] for r in t.header] + [[_strip(c) for c in r["cells"]] for r in t.rows]
    grid = [r + [""] * (ncol - len(r)) for r in grid]
    w = [max(len(r[j]) for r in grid) for j in range(ncol)]
    total = sum(w) + 2 * (ncol - 1)

    def line(r):
        out = []
        for j, c in enumerate(r):
            a = (t.align or ["l"] * ncol)[j]
            out.append(c.ljust(w[j]) if a == "l" else c.center(w[j]) if a == "c" else c.rjust(w[j]))
        return "  ".join(out).rstrip()
    out = [t.caption(), "=" * total]
    for h in t.header:
        out.append(line(h + [""] * (ncol - len(h))))
    out.append("-" * total)
    prev_stat = False
    for r in t.rows:
        if r["kind"] == "stat":
            out.append("-" * total)
        out.append(line(r["cells"] + [""] * (ncol - len(r["cells"]))))
    out.append("=" * total)
    out += ["Note: " + n if i == 0 else n for i, n in enumerate(t.notes)] if t.notes else []
    return "\n".join(out)


_HTML_FONT = "font-family:'Times New Roman',Times,serif;font-size:10pt;"


def to_html(t, standalone=False):
    """HTML table with inline styles only (no classes), so it pastes into Word and Google Docs unchanged.

    Three-rule layout: 1.5 pt top and bottom rules and a 0.75 pt rule under the header.
    """
    ncol = len(t.header[-1])
    al = {"l": "left", "r": "right", "c": "center"}
    cell = _HTML_FONT + "padding:2pt 6pt;line-height:1.25;"
    out = [f'<p style="{_HTML_FONT}margin:12pt 0 4pt 0;"><b>Table {t.number}.</b> {html.escape(t.title)}</p>' if t.number else f'<p style="{_HTML_FONT}margin:12pt 0 4pt 0;"><b>{html.escape(t.title)}</b></p>']
    out.append(f'<table style="border-collapse:collapse;border-top:1.5pt solid #000;border-bottom:1.5pt solid #000;{_HTML_FONT}" cellspacing="0" cellpadding="0">')
    for hi, h in enumerate(t.header):
        last = hi == len(t.header) - 1
        out.append("<tr>")
        if hi == 0 and t.spans:
            pos = 0
            for (a, b, lab) in t.spans:
                while pos < a:
                    out.append(f'<td style="{cell}"></td>')
                    pos += 1
                out.append(f'<td colspan="{b - a + 1}" style="{cell}text-align:center;border-bottom:0.75pt solid #000;">{html.escape(lab)}</td>')
                pos = b + 1
            while pos < ncol:
                out.append(f'<td style="{cell}"></td>')
                pos += 1
        else:
            for j in range(ncol):
                c = h[j] if j < len(h) else ""
                bd = "border-bottom:0.75pt solid #000;" if last else ""
                out.append(f'<td style="{cell}text-align:{"left" if j == 0 else "center"};{bd}"><i>{html.escape(c)}</i></td>' if False else
                           f'<td style="{cell}text-align:{"left" if j == 0 else "center"};{bd}">{html.escape(c)}</td>')
        out.append("</tr>")
    for r in t.rows:
        out.append("<tr>")
        top = "border-top:0.75pt solid #000;" if r["kind"] == "stat" else ""
        for j in range(ncol):
            c = r["cells"][j] if j < len(r["cells"]) else ""
            extra = "padding-top:0;" if r["kind"] == "se" else ""
            style = f'{cell}{extra}text-align:{al[(t.align or ["l"] * ncol)[j]]};{top}'
            out.append(f'<td style="{style}">{html.escape(c)}</td>')
        out.append("</tr>")
    out.append("</table>")
    if t.notes:
        out.append(f'<p style="{_HTML_FONT}font-size:9pt;margin:3pt 0 12pt 0;"><i>Note:</i> {html.escape(" ".join(t.notes))}</p>')
    return "".join(out)


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
        MINUS: "$-$", "<": r"$<$", ">": r"$>$", "²": r"$^2$", "β": r"$\beta$", "χ": r"$\chi$", "≤": r"$\leq$", "×": r"$\times$", "α": r"$\alpha$", "σ": r"$\sigma$", "ρ": r"$\rho$", "±": r"$\pm$", "–": "--"}


def tex(s):
    s = str(s)
    out = []
    i = 0
    while i < len(s):
        # stars become superscripts
        if s[i] == "*":
            j = i
            while j < len(s) and s[j] == "*":
                j += 1
            out.append("$^{" + "*" * (j - i) + "}$")
            i = j
            continue
        out.append(_TEX.get(s[i], s[i]))
        i += 1
    return "".join(out)


def to_latex(t):
    """booktabs + threeparttable; needs \\usepackage{booktabs,threeparttable}."""
    ncol = len(t.header[-1])
    spec = "".join({"l": "l", "r": "r", "c": "c"}[a] for a in (t.align or ["l"] + ["r"] * (ncol - 1)))
    L = ["\\begin{table}[htbp]\\centering", "\\begin{threeparttable}", f"\\caption{{{tex(t.title)}}}\\label{{tab:{t.key}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    for hi, h in enumerate(t.header):
        if hi == 0 and t.spans:
            cells, pos, rules = [], 0, []
            for (a, b, lab) in t.spans:
                while pos < a:
                    cells.append("")
                    pos += 1
                cells.append(f"\\multicolumn{{{b - a + 1}}}{{c}}{{{tex(lab)}}}")
                rules.append(f"\\cmidrule(lr){{{a + 1}-{b + 1}}}")
                pos = b + 1
            L.append(" & ".join(cells) + " \\\\ " + "".join(rules))
        else:
            L.append(" & ".join(tex(h[j]) if j < len(h) else "" for j in range(ncol)) + " \\\\")
    L.append("\\midrule")
    for r in t.rows:
        if r["kind"] == "stat":
            L.append("\\midrule")
        L.append(" & ".join(tex(r["cells"][j]) if j < len(r["cells"]) else "" for j in range(ncol)) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if t.notes:
        L += ["\\begin{tablenotes}\\footnotesize", f"\\item \\textit{{Note:}} {tex(' '.join(t.notes))}", "\\end{tablenotes}"]
    L += ["\\end{threeparttable}", "\\end{table}"]
    return "\n".join(L)


def to_csv(t):
    """The displayed cells as CSV (UTF-8, ASCII minus sign) so Excel reads numbers as numbers."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for h in t.header:
        w.writerow([c.replace(MINUS, "-") for c in h])
    for r in t.rows:
        cells = [c.replace(MINUS, "-") for c in r["cells"]]
        if r["kind"] == "se":   # Excel reads "(0.123)" as a negative number: write the bare value on a labelled row
            cells = ["(std. err. / statistic)"] + [c.strip("()[]") for c in cells[1:]]
        w.writerow(cells)
    return buf.getvalue()
