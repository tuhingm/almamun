#!/usr/bin/env python3
"""Build the step-by-step method guides in ml-guide/ from tools/ml_selector/guides/*.md.

Same format and machinery as tools/model_selector/build_guides.py (the Time-Series Guide).

Each guide is Markdown with a short header (key: value lines, then ---). Every ```python block is executed, in order and
in one shared namespace, from the ml-guide/examples folder; its printed output and any matplotlib figure are
inserted under the code, so every number on the page is produced by the code shown. Use ```python norun for code that
is shown but not executed. usage: python3 tools/ml_selector/build_ml_guides.py [slug ...]
"""
import contextlib, html, io, json, os, pathlib, re, sys, warnings

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "ml-guide"
sys.path.insert(0, str(ROOT / "tools"))
from build_courses import md_to_html  # same Markdown and math settings as the course sites

ORDER = ["lasso", "ridge", "enet", "rf", "xgb", "dml-plr", "dml-irm", "dml-iv", "causal-forest"]
BLOCK = re.compile(r"```python( norun)?\n(.*?)\n```", re.S)


def run_blocks(slug, body):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np, pandas as pd
    plt.rcParams.update({"figure.figsize": (8, 3.4), "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.prop_cycle": plt.cycler(color=["#0033a0", "#c46000", "#0f7a4f", "#6ec1e4", "#8a2a17"]),
                         "axes.grid": True, "grid.alpha": .25, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
    pd.set_option("display.width", 110); pd.set_option("display.max_columns", 12); pd.set_option("display.precision", 4)
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    ns, cells = {"__name__": "__main__"}, []
    (OUT / "img").mkdir(exist_ok=True)
    cwd = os.getcwd(); os.chdir(OUT / "examples")
    if str(OUT / "examples") not in sys.path:
        sys.path.insert(0, str(OUT / "examples"))   # so guides can import helper files kept with the data

    def sub(m):
        code = m.group(2)
        cell = f'<div class="cell"><div class="cell-label">Python</div><pre class="code"><code>{html.escape(code)}</code></pre>'
        if not m.group(1):
            buf = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
                warnings.simplefilter("ignore")
                exec(compile(code, f"<{slug} cell {len(cells) + 1}>", "exec"), ns)
            out = buf.getvalue().rstrip()
            if out:
                cell += f'<div class="cell-label out-l">Output</div><pre class="out">{html.escape(out)}</pre>'
            for k, num in enumerate(plt.get_fignums()):
                name = f"{slug}-{len(cells) + 1}{'abc'[k] if k else ''}.png"
                plt.figure(num).savefig(OUT / "img" / name, bbox_inches="tight", facecolor="white")
                cell += f'<img class="fig" src="img/{name}" alt="Figure produced by the code above" loading="lazy">'
            plt.close("all")
        cells.append(cell + "</div>")
        return f"\n\n@@CELL{len(cells) - 1}@@\n\n"
    try:
        body = BLOCK.sub(sub, body)
    finally:
        os.chdir(cwd)
    return body, cells


def build(slug, metas):
    text = (HERE / "guides" / f"{slug}.md").read_text()
    head, body = text.split("\n---\n", 1)
    meta = dict(l.split(":", 1) for l in head.strip().splitlines())
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    body, cells = run_blocks(slug, body)
    page = md_to_html(body)
    for i, c in enumerate(cells):
        page = page.replace(f"<p>@@CELL{i}@@</p>", c)
    assert "@@CELL" not in page, "unplaced cell"
    toc, n = [], 0

    def h2(m):
        nonlocal n
        n += 1
        title = re.sub(r"<[^>]+>", "", m.group(1))
        toc.append(f'<a href="#s{n}"><span>{n:02d}</span>{title}</a>')
        return f'<h2 id="s{n}"><span class="stepno">Step {n}</span>{m.group(1)}</h2>'
    page = re.sub(r"<h2[^>]*>(.*?)</h2>", h2, page)
    others = "".join(f'<a class="{"cur" if s == slug else ""}" href="{s}.html">{metas[s]["short"]}</a>' for s in ORDER if s in metas)
    tpl = (HERE / "guide_template.html").read_text()
    for k, v in dict(TITLE=meta["title"], SHORT=meta["short"], WHEN=meta["when"], DATA=meta["data"], LEAD=meta["lead"], SLUG=slug, TOOLS=meta.get("tools", "Python · scikit-learn"),
                     STEPS=str(n), TOC="".join(toc), BODY=page, OTHERS=others).items():
        tpl = tpl.replace("{{" + k + "}}", v)
    (OUT / f"{slug}.html").write_text(tpl)
    print(f"{slug}: {n} steps, {len(cells)} code cells")
    return meta


if __name__ == "__main__":
    metas = {}
    for s in ORDER:
        f = HERE / "guides" / f"{s}.md"
        if f.exists():
            head = f.read_text().split("\n---\n", 1)[0]
            metas[s] = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}
    todo = [a for a in sys.argv[1:]] or list(metas)
    for s in todo:
        build(s, metas)
