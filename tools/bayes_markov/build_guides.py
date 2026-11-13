#!/usr/bin/env python3
"""Build the Bayesian and Markov-switching research guides from tools/bayes_markov/guides/<section>/*.md.

    python3 tools/bayes_markov/build_guides.py bayesian [slug ...]
    python3 tools/bayes_markov/build_guides.py markov [slug ...]

Same machinery as the other guides (tools/ml_selector/build_ml_guides.py): every ```python block is executed in order,
in one namespace, from the section's examples folder; its printed output and matplotlib figures are inserted under the
code. The estimators come from the Open Econometric Systems engine (oes/engine/oes_engine), so the numbers on the guide
pages are the numbers the OES page gives. Run with the Pyodide 0.27.2 package versions (see tools/oes_sync.py).
Then run tools/site_polish.py, tools/share_buttons.py, tools/og_tags.py and tools/make_sitemap.py.
"""
import contextlib, html, io, os, pathlib, re, shutil, sys, warnings

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "oes" / "engine"))
sys.dont_write_bytecode = True
from build_courses import md_to_html  # noqa: E402

SECTIONS = {
    "bayesian": dict(folder="bayesian-guide", name="Bayesian Research Guide", short="Bayesian Guide",
                     order=["bvar", "bayes-reg", "favar", "steady-state", "panel-var", "tvp-var", "sv-var", "mixed-frequency", "threshold-var"],
                     data=["time-series-guide/examples/money.csv", "oes/examples/us_macro.csv", "oes/examples/mixed_frequency.csv",
                           "panel-data-guide/examples/garrett.csv", "ml-guide/examples/wage2.csv"]),
    "markov": dict(folder="markov-guide", name="Markov-Switching Research Guide", short="Markov Guide",
                   order=["ms-ar", "ms-regression", "ms-var", "ms-panel-var"],
                   data=["oes/examples/hamilton_gnp.csv", "oes/examples/fedfunds.csv", "oes/examples/us_macro.csv",
                         "panel-data-guide/examples/garrett.csv"]),
    # local projections: two more guides in the existing Time-Series and Panel-Data guides (hubs are built by their own tools)
    "lp": dict(folder="time-series-guide", name="Empirical Research Guide (Time Series)", short="Time-Series Guide",
               order=["ols", "ardl", "nardl", "var", "vecm", "gmm", "lp", "svar", "bp"], static=dict(ols="OLS", ardl="ARDL", nardl="NARDL", var="VAR", vecm="VECM", gmm="2SLS / GMM"),
               data=[]),
    "plp": dict(folder="panel-data-guide", name="Panel Data Research Guide", short="Panel-Data Guide",
                order=["pols", "re", "fe", "gmm", "fedk", "mg", "diff", "pmg", "csardl", "plp", "pthresh"],
                static=dict(pols="Pooled OLS", re="Random effects", fe="Fixed effects", gmm="GMM", fedk="FE-DK", mg="MG / CCE", diff="Δ model", pmg="PMG", csardl="CS-ARDL"),
                data=[]),
}
BLOCK = re.compile(r"```python( norun)?\n(.*?)\n```", re.S)


def run_blocks(out, slug, body):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np, pandas as pd
    plt.rcParams.update({"figure.figsize": (8, 3.4), "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.prop_cycle": plt.cycler(color=["#0033a0", "#c46000", "#0f7a4f", "#6ec1e4", "#8a2a17"]),
                         "axes.grid": True, "grid.alpha": .25, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
    pd.set_option("display.width", 120); pd.set_option("display.max_columns", 14); pd.set_option("display.precision", 4)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    ns, cells = {"__name__": "__main__"}, []
    (out / "img").mkdir(exist_ok=True)
    cwd = os.getcwd(); os.chdir(out / "examples")

    def sub(m):
        code = m.group(2)
        cell = f'<div class="cell"><div class="cell-label">Python</div><pre class="code"><code>{html.escape(code)}</code></pre>'
        if not m.group(1):
            buf = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
                warnings.simplefilter("ignore")
                exec(compile(code, f"<{slug} cell {len(cells) + 1}>", "exec"), ns)
            o = buf.getvalue().rstrip()
            if o:
                cell += f'<div class="cell-label out-l">Output</div><pre class="out">{html.escape(o)}</pre>'
            for k, num in enumerate(plt.get_fignums()):
                name = f"{slug}-{len(cells) + 1}{'abcdefghijkl'[k] if k else ''}.png"
                plt.figure(num).savefig(out / "img" / name, bbox_inches="tight", facecolor="white")
                cell += f'<img class="fig" src="img/{name}" alt="Figure produced by the code above" loading="lazy">'
            plt.close("all")
        cells.append(cell + "</div>")
        return f"\n\n@@CELL{len(cells) - 1}@@\n\n"
    try:
        body = BLOCK.sub(sub, body)
    finally:
        os.chdir(cwd)
    return body, cells


def meta_of(path):
    head = path.read_text().split("\n---\n", 1)[0]
    return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}


def build(key, slug, metas):
    S = SECTIONS[key]
    out = ROOT / S["folder"]
    text = (HERE / "guides" / key / f"{slug}.md").read_text()
    head, body = text.split("\n---\n", 1)
    meta = metas[slug]
    body, cells = run_blocks(out, slug, body)
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
    oes = f'<p class="oes-run"><a href="../oes/index.html#{meta["oes"]}">Run this model on your own data in Open Econometric Systems →</a></p>' if meta.get("oes") else ""
    others = "".join(f'<a class="{"cur" if s == slug else ""}" href="{s}.html">{(metas[s]["short"] if s in metas else S["static"][s])}</a>' for s in S["order"] if s in metas or s in S.get("static", {}))
    tpl = (HERE / "guide_template.html").read_text()
    for k, v in dict(TITLE=meta["title"], SHORT=meta["short"], WHEN=meta["when"], DATA=meta["data"], LEAD=meta["lead"], SLUG=slug,
                     TOOLS=meta.get("tools", "Python · OES engine · statsmodels"), STEPS=str(n), TOC="".join(toc), BODY=oes + page, OTHERS=others,
                     SECTION=S["name"], SECTION_SHORT=S["short"], SECTION_KEY=key, FOLDER=S["folder"]).items():
        tpl = tpl.replace("{{" + k + "}}", v)
    tpl = tpl.replace("</style>", ".oes-run{margin:0 0 18px}.oes-run a{display:inline-block;font-size:12px;font-weight:800;color:#fff!important;background:var(--navy);"
                                  "padding:9px 14px;border-radius:8px;text-decoration:none!important}\n</style>", 1)
    (out / f"{slug}.html").write_text(tpl)
    print(f"{key}/{slug}: {n} steps, {len(cells)} code cells")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "bayesian"
    S = SECTIONS[key]
    out = ROOT / S["folder"]
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for f in S["data"]:
        shutil.copy(ROOT / f, out / "examples" / pathlib.Path(f).name)
    metas = {s: meta_of(HERE / "guides" / key / f"{s}.md") for s in S["order"] if (HERE / "guides" / key / f"{s}.md").exists()}
    for s in (sys.argv[2:] or list(metas)):
        build(key, s, metas)


if __name__ == "__main__":
    main()
