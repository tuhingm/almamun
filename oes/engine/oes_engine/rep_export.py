"""Exporters of the reporting layer: JSON for the page, standalone HTML, Word-compatible HTML, LaTeX document, plain-text
manuscript and a file bundle. Everything is a pure function of a finished Report; nothing here changes a number.

The browser adds what only a browser can draw (PNG at 300 dpi, the ZIP container, the Word MHTML wrapper) from the pieces
produced here, so the whole flow works with no server.
"""
import base64
import html as _html
import re

from . import rep_tables as T
from .rep_tables import to_csv, to_html, to_latex, to_text

CSS = """
body{font-family:'Times New Roman',Times,serif;font-size:11pt;line-height:1.45;color:#000;max-width:7.2in;margin:24px auto;padding:0 16px}
h1{font-size:17pt;margin:0 0 4pt}h2{font-size:13.5pt;margin:22pt 0 4pt;border-bottom:0.75pt solid #000;padding-bottom:2pt}
h3{font-size:11pt;margin:12pt 0 3pt;font-style:italic}p{margin:4pt 0 6pt}.cap{font-size:10pt;margin:3pt 0 0}.fignote{font-size:9pt;margin:2pt 0 14pt}
.warn{border-left:3px solid #b00;padding-left:8px}.meta{font-size:9.5pt;color:#333}.toc li{margin:1pt 0}
svg{max-width:100%;height:auto;display:block}img.fig{max-width:100%;height:auto}.nv{font-size:9pt;color:#555}
@media print{h2{page-break-after:avoid}table,svg,img{page-break-inside:avoid}}
"""


def esc(s):
    return _html.escape(str(s))


# ------------------------------------------------------------------------------------------------- pieces
def paragraphs_text(pars):
    return "\n\n".join(p.text for p in pars)


def figure_png_name(fig):
    return f"figure_{fig.number:02d}_{fig.key}"


def table_name(t):
    return f"table_{t.number:02d}_{t.key}"


def manuscript_text(rep):
    """Methods, one results block per step, limitations and references as plain text for the manuscript."""
    out = ["METHODS", paragraphs_text(rep.methods), ""]
    out.append("RESULTS")
    for s in rep.steps:
        res = [p for p in s.paragraphs if p.kind in ("result", "warning", "note")]
        if not res:
            continue
        out += [s.title, paragraphs_text(res), ""]
    out += ["LIMITATIONS", paragraphs_text(rep.limitations), ""]
    out += ["REFERENCES"] + rep.used_references()
    return "\n".join(out)


def step_html(s, rep, images=False, fig_src=None):
    o = [f'<h2 id="step-{esc(s.key)}">{esc(s.title)}</h2>']
    if s.purpose:
        o.append(f'<p class="meta"><i>{esc(s.purpose)}</i></p>')
    for t in s.tables:
        o.append(to_html(t))
    for f in s.figures:
        if images:
            o.append(f'<p class="cap"><b>Figure {f.number}.</b> {esc(f.title)}</p><img class="fig" src="cid:{esc(figure_png_name(f))}" width="{int(f.width_in * 96)}">')
        else:
            o.append(f'<p class="cap"><b>Figure {f.number}.</b> {esc(f.title)}</p>{f.svg}')
        o.append(f'<p class="fignote"><i>Note:</i> {esc(f.note)} {esc(f.source)}</p>')
    for p in s.paragraphs:
        cls = ' class="warn"' if p.kind == "warning" else ""
        o.append(f"<p{cls}>{esc(p.text)}</p>")
    if s.validated == "not validated":
        o.append('<p class="nv">This step is not validated against a reference implementation.</p>')
    else:
        o.append(f'<p class="nv">Validation: {esc(s.validated)}</p>')
    return "".join(o)


def html_report(rep, images=False):
    toc = "".join(f'<li><a href="#step-{esc(s.key)}">{i}. {esc(s.title)}</a></li>' for i, s in enumerate(rep.steps, 1))
    meth = "".join(f"<p>{esc(p.text)}</p>" for p in rep.methods)
    lim = "".join(f"<p>{esc(p.text)}</p>" for p in rep.limitations)
    refs = "".join(f"<p>{esc(r)}</p>" for r in rep.used_references())
    body = "".join(step_html(s, rep, images) for s in rep.steps)
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{esc(rep.title)}</title><style>{CSS}</style></head><body>'
            f'<h1>{esc(rep.title)}</h1><p class="meta">Open Econometric Systems report. Every number is computed from the data supplied; nothing is typed by hand.</p>'
            f'<ol class="toc">{toc}</ol><h2>Methods</h2>{meth}{body}<h2>Limitations</h2>{lim}<h2>References</h2>{refs}</body></html>')


# ------------------------------------------------------------------------------------------------- LaTeX
def latex_document(rep):
    from .rep_tables import tex
    L = ["\\documentclass[11pt]{article}", "\\usepackage[margin=1in]{geometry}", "\\usepackage{booktabs,threeparttable,graphicx,float}", "\\usepackage{times}",
         f"\\title{{{tex(rep.title)}}}", "\\date{}", "\\begin{document}", "\\maketitle"]
    L += ["\\section*{Methods}", "\n\n".join(tex(p.text) for p in rep.methods)]
    for s in rep.steps:
        L.append(f"\\section{{{tex(s.title)}}}")
        for t in s.tables:
            L.append(to_latex(t))
        for f in s.figures:
            L += ["\\begin{figure}[htbp]\\centering",
                  f"\\includegraphics[width=\\textwidth]{{figures/{figure_png_name(f)}.pdf}}",
                  f"\\caption{{{tex(f.title)}}}\\label{{fig:{f.key}}}",
                  f"\\par\\footnotesize\\textit{{Note:}} {tex(f.note)} {tex(f.source)}", "\\end{figure}"]
        for p in s.paragraphs:
            L.append(tex(p.text) + "\n")
    L += ["\\section*{Limitations}", "\n\n".join(tex(p.text) for p in rep.limitations), "\\section*{References}", "\\begin{itemize}"]
    L += [f"\\item {tex(r)}" for r in rep.used_references()]
    L += ["\\end{itemize}", "\\end{document}"]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- JSON for the page
def _tests_rows(s):
    return [dict(name=t.name, statistic=t.statistic, p_value=t.p_value, df=t.df, null=t.null, reject=t.reject, problem=t.problem, reference=t.reference) for t in s.tests]


def to_json(rep):
    steps = []
    for s in rep.steps:
        steps.append(dict(
            key=s.key, title=s.title, purpose=s.purpose, validated=s.validated,
            paragraphs=[dict(kind=p.kind, text=p.text) for p in s.paragraphs],
            tests=_tests_rows(s),
            tables=[dict(key=t.key, number=t.number, title=t.title, caption=t.caption(), name=table_name(t), notes=t.notes,
                         text=to_text(t), html=to_html(t), latex=to_latex(t), csv=to_csv(t), header=t.header, rows=t.rows, raw=t.raw) for t in s.tables],
            figures=[dict(key=f.key, number=f.number, title=f.title, caption=f.caption(), name=figure_png_name(f), note=f.note, source=f.source,
                          full_caption=f"{f.caption()} {f.note} {f.source}".strip(), svg=f.svg, width_in=f.width_in, height_in=f.height_in,
                          pdf_b64=_pdf_b64(f)) for f in s.figures]))
    return dict(model_id=rep.model_id, title=rep.title, meta=rep.meta,
                methods=[p.text for p in rep.methods], limitations=[p.text for p in rep.limitations], references=rep.used_references(),
                steps=steps, unvalidated=rep.unvalidated_steps(), manuscript=manuscript_text(rep),
                exports=dict(html_images=html_report(rep, images=True), tex=latex_document(rep)))


def _pdf_b64(f):
    from . import rep_fig
    return base64.b64encode(rep_fig.render(f.spec, f.width_in, f.height_in).pdf()).decode()


# ------------------------------------------------------------------------------------------------- file bundle (used by tests and scripts)
def bundle(rep):
    """{path: bytes} of every table (csv, tex, html), figure (svg, pdf), the report as HTML and LaTeX, and the manuscript text."""
    from . import rep_fig
    files = {}
    for s in rep.steps:
        for t in s.tables:
            n = table_name(t)
            files[f"tables/{n}.csv"] = to_csv(t).encode("utf-8-sig")
            files[f"tables/{n}.tex"] = to_latex(t).encode()
            files[f"tables/{n}.html"] = to_html(t).encode()
        for f in s.figures:
            n = figure_png_name(f)
            files[f"figures/{n}.svg"] = f.svg.encode()
            files[f"figures/{n}.pdf"] = rep_fig.render(f.spec, f.width_in, f.height_in).pdf()
    files["report.html"] = html_report(rep).encode()
    files["report.tex"] = latex_document(rep).encode()
    files["manuscript.txt"] = manuscript_text(rep).encode()
    return files
