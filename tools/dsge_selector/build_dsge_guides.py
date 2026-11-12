#!/usr/bin/env python3
"""Build the step-by-step Dynare guides in dsge-guide/ from tools/dsge_selector/guides/*.md.

Same page format as the Time-Series and Panel guides, but every code cell is Dynare or Octave/MATLAB and is executed:

    ```dynare rbc.mod          the block is written to dsge-guide/models/<slug>/rbc.mod and run with Dynare
    ```dynare rbc.mod norun    written and shown, not run (e.g. a variant the next cell runs)
    ```octave                  Octave/MATLAB code run in the same session (Dynare's oo_, M_, options_ are available)
    ```octave hide             run but not shown (used to export results for the main page)

Cell options: from=REGEX / to=REGEX keep the part of the output between the first match of each; max=N keeps at
most N lines. Every cell of a guide runs, in order, in one Octave session (Dynare 6, xvfb for figures). Figures are
saved by the cells themselves with print(..., 'name.png'); a figure appears under the cell whose code names it.
Requires: apt install dynare octave xvfb, pip install markdown pymdown-extensions jinja2 pyyaml.
usage: python3 tools/dsge_selector/build_dsge_guides.py [slug ...]
"""
import html, json, os, pathlib, re, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "dsge-guide"
EXPORT = HERE / "exports"
sys.path.insert(0, str(ROOT / "tools"))
from build_courses import md_to_html

ORDER = ["rbc", "nk", "soe", "estimation", "medium", "identification", "optimal", "zlb", "higher"]
CELL = re.compile(r"```(dynare|octave)([^\n]*)\n(.*?)\n```", re.S)
NOISE = re.compile(r"^(Starting Dynare|Using |Configuring Dynare|Starting preprocessing|Found \d+|Evaluating expressions|"
                   r"Computing static|Computing dynamic|Processing outputs|Preprocessing completed|Starting MATLAB|"
                   r"Total computing time|Note: warning|Note: \d+ warning|warning: |Warning: |QStandardPaths|"
                   r"Writing |Substitution of|Computing (s|t|d)|Preprocessing (of|done|time)|  - |Loading |WARNING: |If you are |"
                   r"Calling Dynare|Normalizing|Finding the optimal|\d+ block\(s\)|\d+ recursive block|the largest simultaneous|"
                   r"and \d+ feedback|Some of the parameters have no value|X13|called from|_run_guide|Could not normalize|Normalization failed|"
                   r"The (original Ramsey|static|dynamic) model cannot be normalized|Ramsey Problem: added|Computing (original|planner)|"
                   r"RW Metropolis|Estimation::mcmc|.*% done|Loading |Period \d+ of)")
PREAMBLE = r"""
graphics_toolkit qt; set(0, 'defaultfigurevisible', 'off');
set(0, 'defaultaxesfontsize', 9, 'defaultlinelinewidth', 1.8, 'defaultaxesbox', 'off');
set(0, 'defaultaxescolororder', [0 0.2 0.627; 0.769 0.376 0; 0.059 0.478 0.31; 0.431 0.757 0.894; 0.541 0.165 0.09]);
addpath /usr/lib/dynare/matlab;
warning('off', 'all'); more off; format short g;
"""


def opts(s):
    o = {}
    for tok in re.findall(r'(\w+)(?:=("[^"]*"|\S+))?', s):
        o[tok[0]] = tok[1].strip('"') if tok[1] else True
    return o


def parse(text):
    head, body = text.split("\n---\n", 1)
    meta = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in head.strip().splitlines())}
    cells = []
    def sub(m):
        kind, rest, code = m.group(1), m.group(2).strip(), m.group(3)
        parts = rest.split(None, 1) if kind == "dynare" else [None, rest]
        fname = parts[0] if kind == "dynare" else None
        o = opts(parts[1] if len(parts) > 1 else "")
        cells.append(dict(kind=kind, file=fname, opt=o, code=code))
        return f"\n\n@@CELL{len(cells) - 1}@@\n\n"
    body = CELL.sub(sub, body)
    return meta, body, cells


def run_cells(slug, cells):
    work = OUT / "models" / slug
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in (OUT / "data").glob("*"):
        shutil.copy(f, work / f.name)
    EXPORT.mkdir(exist_ok=True)
    lines = [PREAMBLE, f"cd('{work}');", f"export_dir = '{EXPORT}';"]
    for i, c in enumerate(cells):
        if c["kind"] == "dynare":
            (work / c["file"]).write_text(c["code"] + "\n")
            if c["opt"].get("norun"):
                continue
            body = f"dynare {c['file'][:-4]} noclearall nograph nolog console"
        else:
            body = c["code"]
        lines.append(f"disp('@@START {i}@@');\ntry\n{body}\ncatch err\n  disp(['@@ERROR ' err.message]);\nend\ndisp('@@END {i}@@');")
    script = work / "_run_guide.m"
    script.write_text("\n".join(lines) + "\n")
    p = subprocess.run(["xvfb-run", "-a", "octave", "--no-gui", "--quiet", str(script)], capture_output=True, text=True, cwd=work, timeout=7200)
    log = p.stdout + "\n" + p.stderr
    (work / "_run_guide.log").write_text(log)
    pathlib.Path("/tmp/dsge_logs").mkdir(exist_ok=True); pathlib.Path(f"/tmp/dsge_logs/{slug}.log").write_text(log)
    outs = {}
    for i in range(len(cells)):
        m = re.search(rf"@@START {i}@@\n(.*?)@@END {i}@@", log, re.S)
        if m:
            outs[i] = m.group(1)
    errs = re.findall(r"@@ERROR (.*)", log)
    if errs:
        raise RuntimeError(f"{slug}: Octave/Dynare error: {errs[0]}\n(see {work / '_run_guide.log'})")
    script.unlink()
    for f in list(work.glob("+*")) + list(work.glob("*_run_guide*")):
        pass
    return work, outs


def clean(out, o):
    keep = [l.rstrip() for l in out.splitlines() if not NOISE.match(l.strip())]
    text = "\n".join(keep)
    if o.get("from"):
        m = re.search(o["from"], text)
        if m:
            text = text[m.start():]
    if o.get("to"):
        m = re.search(o["to"], text)
        if m:
            text = text[:m.start()]
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    if o.get("max"):
        ls = text.splitlines()
        n = int(o["max"])
        if len(ls) > n:
            text = "\n".join(ls[:n] + ["…"])
    return text


def build(slug, metas):
    meta, body, cells = parse((HERE / "guides" / f"{slug}.md").read_text())
    work, outs = run_cells(slug, cells)
    (OUT / "img").mkdir(exist_ok=True)
    rendered = []
    for i, c in enumerate(cells):
        if c["opt"].get("hide"):
            rendered.append("")
            continue
        label = f"Dynare model file · {c['file']}" if c["kind"] == "dynare" else "Octave / MATLAB"
        cell = f'<div class="cell"><div class="cell-label">{html.escape(label)}</div><pre class="code"><code>{html.escape(c["code"])}</code></pre>'
        if c["kind"] == "dynare" and not c["opt"].get("norun"):
            cell += f'<div class="cell-label out-l">Command</div><pre class="out">dynare {c["file"][:-4]}</pre>'
        out = clean(outs.get(i, ""), c["opt"])
        if out and not c["opt"].get("noout"):
            cell += f'<div class="cell-label out-l">Output</div><pre class="out">{html.escape(out)}</pre>'
        for name in re.findall(r"['\"]([\w\-]+\.png)['\"]", c["code"]):
            src = work / name
            if src.exists():
                dst = f"{slug}-{name}"
                shutil.copy(src, OUT / "img" / dst)
                cell += f'<img class="fig" src="img/{dst}" alt="Figure produced by the code above" loading="lazy">'
        files = [c["file"]] if c["kind"] == "dynare" else []
        if files:
            cell += f'<div class="dl"><a href="models/{slug}/{c["file"]}" download>Download {html.escape(c["file"])}</a></div>'
        rendered.append(cell + "</div>")
    page = md_to_html(body)
    for i, c in enumerate(rendered):
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
    for k, v in dict(TITLE=meta["title"], SHORT=meta["short"], WHEN=meta["when"], DATA=meta["data"], LEAD=meta["lead"], SLUG=slug,
                     STEPS=str(n), TOC="".join(toc), BODY=page, OTHERS=others).items():
        tpl = tpl.replace("{{" + k + "}}", v)
    (OUT / f"{slug}.html").write_text(tpl)
    # keep only the model files and data in the published folder
    for f in work.iterdir():
        if f.is_dir() or f.suffix not in (".mod", ".csv", ".m") or f.name.startswith("_"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    used = " ".join(c["code"] for c in cells)
    for f in work.glob("*.csv"):
        if f.name not in used:
            f.unlink()
    for f in work.glob("*.m"):
        if f.stem not in {c["file"][:-4] for c in cells if c["kind"] == "dynare"} and not (OUT / "data" / f.name).exists():
            f.unlink()
    print(f"{slug}: {n} steps, {len(cells)} cells")
    return meta


if __name__ == "__main__":
    metas = {}
    for s in ORDER:
        f = HERE / "guides" / f"{s}.md"
        if f.exists():
            metas[s] = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in f.read_text().split("\n---\n", 1)[0].strip().splitlines())}
    todo = sys.argv[1:] or list(metas)
    for s in todo:
        build(s, metas)
