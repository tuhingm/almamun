#!/usr/bin/env python3
"""Build the index pages of the Bayesian and Markov-switching guides (bayesian-guide/index.html, markov-guide/index.html).

    python3 tools/bayes_markov/build_hubs.py

Each hub has a 'which method fits your data' table, one card per guide (title, when to use it, data, steps) and a link to the
same model in Open Econometric Systems. Titles, 'when' lines and data come from the header of each guide's .md file.
Then run tools/site_polish.py, tools/share_buttons.py, tools/og_tags.py and tools/make_sitemap.py.
"""
import html, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_guides import SECTIONS, meta_of  # noqa: E402

HUBS = {
    "bayesian": dict(
        title="Which Bayesian VAR fits your question?",
        lead="Bayesian methods let you estimate models with many parameters from short macroeconomic samples, because the prior shrinks what the data cannot pin down. "
             "Find your situation in the table, then follow the guide: each one takes a real dataset through pre-estimation tests, estimation, convergence checks, "
             "diagnostics, robustness and the text for the paper.",
        table=[("A single regression with many regressors or a small sample", "Bayesian linear regression", "bayes-reg"),
               ("Several macroeconomic series, a short sample, a VAR that would otherwise have too many parameters", "Bayesian VAR (Minnesota, Normal-Wishart)", "bvar"),
               ("Dozens of indicators that move together; you want a few common factors in the VAR", "Factor-augmented VAR (FAVAR)", "favar"),
               ("The long-run means are known from theory or policy (an inflation target, trend growth)", "Steady-state BVAR", "steady-state"),
               ("A panel of countries or firms with a short time dimension; you want a VAR for the whole panel", "Panel VAR", "panel-var"),
               ("Relationships drift slowly over time (the effect of policy changes across decades)", "Time-varying parameter VAR", "tvp-var"),
               ("The size of shocks changes over time (the Great Moderation, 2008, 2020)", "Stochastic-volatility VAR", "sv-var"),
               ("Monthly indicators arrive before quarterly GDP and you want a nowcast", "Mixed-frequency BVAR", "mixed-frequency"),
               ("Dynamics differ above and below a threshold of a variable you observe (inflation, the interest rate)", "Bayesian threshold VAR", "threshold-var")],
        other=("The regimes are <b>hidden</b> rather than defined by an observed variable?", "Use the <a href=\"../markov-guide/index.html\">Markov-Switching Guide</a>.")),
    "markov": dict(
        title="Which Markov-switching model fits your data?",
        lead="Markov-switching models let the data date hidden regimes, such as expansions and recessions or calm and turbulent periods, and tell you how long each lasts. "
             "Find your situation in the table, then follow the guide: each one covers the number of regimes, the linearity test, regime dates, diagnostics, "
             "the checks for a collapsed regime and the text for the paper.",
        table=[("One series (GDP growth, an interest rate) whose mean shifts between states", "Markov-switching autoregression (Hamilton)", "ms-ar"),
               ("A regression whose intercept, slopes or error variance shift between states", "Markov-switching regression", "ms-regression"),
               ("Several variables whose means, volatilities or dynamics shift together", "Markov-switching VAR", "ms-var"),
               ("A panel of countries that share the same regimes (a global slowdown)", "Markov-switching panel VAR", "ms-panel-var")],
        other=("The regime is <b>observed</b> (a variable above or below a threshold)?", "Use the <a href=\"../bayesian-guide/threshold-var.html\">Bayesian threshold VAR</a>.")),
}

CSS = """
.hubtab{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13.5px;margin:12px 0 8px}
.hubtab th{font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#7f8aa1;text-align:left;padding:10px 14px;background:#f6f8fd;border-bottom:1px solid var(--line)}
.hubtab td{padding:11px 14px;border-bottom:1px solid #edf1f7;vertical-align:top}.hubtab tr:last-child td{border-bottom:0}.hubtab td a{color:var(--navy);font-weight:700;text-decoration:underline}
.hubnote{background:#fff;border:1px solid var(--line);border-left:4px solid var(--sky2);border-radius:10px;padding:12px 16px;font-size:13.5px;margin:14px 0 6px}
.hubhead{margin:44px 0 6px}.hubhead h2{font:800 23px/1.25 Manrope;letter-spacing:-.5px;margin:6px 0 6px}.hubhead p{margin:0;color:var(--muted);max-width:760px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;margin-top:18px}
.card{display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 20px 16px;box-shadow:0 5px 18px rgba(0,32,91,.06);transition:transform .15s,box-shadow .15s}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(0,32,91,.12)}.card .label{margin-bottom:8px}.card h3{font:800 17px/1.3 Manrope;margin:0 0 8px}.card p{font-size:13px;line-height:1.6;color:#41506a;margin:0 0 10px}
.card .data{font-size:11.5px;color:var(--muted);margin-top:auto;padding-top:10px;border-top:1px solid #edf1f7}.card .go{font-size:12px;font-weight:800;color:var(--navy);margin-top:10px}
.runbox{margin-top:30px;background:var(--navy);color:#fff;border-radius:14px;padding:20px 24px;display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.runbox p{margin:0;max-width:640px;font-size:13.5px;color:#dbe6f6}.runbox a{background:#fff;color:var(--navy);font-weight:800;font-size:12px;padding:10px 16px;border-radius:9px;text-decoration:none}
"""


def build(key):
    S, H = SECTIONS[key], HUBS[key]
    folder = ROOT / S["folder"]
    metas = {s: meta_of(HERE / "guides" / key / f"{s}.md") for s in S["order"]}
    steps = {}
    for s in S["order"]:
        pg = (folder / f"{s}.html")
        steps[s] = len(re.findall(r'<h2 id="s\d+">', pg.read_text())) if pg.exists() else None
    tpl = (HERE / "guide_template.html").read_text()
    head, rest = tpl.split('<section class="hero">', 1)
    foot = rest.split("</main>", 1)[1]
    desc = re.sub(r"<[^>]+>", "", H["lead"]).split(". ")[0]
    for k, v in dict(TITLE=H["title"], WHEN=desc, SLUG="index", SECTION=S["name"], SECTION_KEY=key, FOLDER=S["folder"],
                     SECTION_SHORT=S["short"]).items():
        head = head.replace("{{" + k + "}}", v)
        foot = foot.replace("{{" + k + "}}", v)
    head = head.replace("?: ", "? ")
    head = head.replace(": a step-by-step guide to writing the empirical paper, with Python code, real output and sample text. ", ": ")
    head = head.replace("</style>", CSS + "</style>", 1)
    n = len(S["order"])
    hero = (f'<section class="hero"><div class="wrap"><div class="crumbs">Empirical Research Guide<span>/</span>{S["short"]}</div>'
            f'<div class="kicker">Empirical Research Guide · {S["short"].replace(" Guide", "")}</div><h1>{H["title"]}</h1><p class="lead">{H["lead"]}</p>'
            f'<div class="meta"><span><strong>{n}</strong> method guides</span><span>Real example data, Python code and output</span>'
            f'<span>Runs on your own data in Open Econometric Systems</span></div></div></section>')
    rows = "".join(f'<tr><td>{html.escape(q)}</td><td><a href="{slug}.html">{html.escape(m)}</a></td></tr>' for q, m, slug in H["table"])
    cards = ""
    for s in S["order"]:
        m = metas[s]
        cards += (f'<a class="card" href="{s}.html"><span class="label">{html.escape(m["short"])}</span><h3>{html.escape(m["title"])}</h3>'
                  f'<p>{html.escape(m["when"])}.</p><div class="data">Data: {html.escape(m["data"])}'
                  + (f' · {steps[s]} steps' if steps[s] else "") + f'</div><span class="go">Open the guide →</span></a>')
    body = (f'<main class="wrap"><article style="max-width:none">'
            f'<h2 style="margin-top:0;padding-top:0;border:0">Which method fits?</h2>'
            f'<table class="hubtab"><tr><th>Your situation</th><th>Method and guide</th></tr>{rows}</table>'
            f'<div class="hubnote">{H["other"][0]} {H["other"][1]}</div>'
            f'<div class="hubhead"><span class="label">Method guides</span><h2>{n} routes to a finished paper</h2>'
            f'<p>Each guide takes one real dataset through pre-estimation tests, estimation, regime or convergence checks, diagnostics, robustness, and sample text for the paper.</p></div>'
            f'<div class="cards">{cards}</div>'
            f'<div class="runbox"><p>Every model in this guide runs in your browser on your own CSV file, with the same code and the same numbers as the guides. No installation and no server.</p>'
            f'<a href="../oes/index.html">Open Econometric Systems →</a></div></article></main>')
    (folder / "index.html").write_text(head + hero + body + foot)
    print(f"{key}: hub written ({n} guides, steps {steps})")


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(HUBS)):
        build(k)
