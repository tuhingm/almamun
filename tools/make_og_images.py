#!/usr/bin/env python3
"""Render the 1200x630 link-preview (Open Graph) images listed in tools/og_cards.json, plus the three top-level pages.

Run after tools/build_courses.py. Needs Playwright with Chromium; fonts are loaded from Google Fonts unless
OG_FONT_CSS points to a local @font-face stylesheet. Output: assets/og/**.jpg.
"""
import asyncio, json, os, pathlib
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
cards = json.loads((ROOT / "tools" / "og_cards.json").read_text())
cards += [
    dict(out="assets/og-image.jpg", kicker="Economist · Quantitative Researcher", title="Tuhin Al Mamun, Ph.D.",
         sub="Monetary and fiscal policy · Environmental economics · Econometrics", nowho=True),
    dict(out="assets/og/working-papers.jpg", kicker="Research portfolio · 2023–present", title="Working Papers",
         sub="Fifteen manuscripts on decarbonization, energy, inflation, monetary institutions and macro-finance.",
         tags=["Climate & energy", "Inflation", "Monetary institutions", "Macro-finance"]),
    dict(out="assets/og/time-series-guide.jpg", kicker="Empirical Research Guide", title="Which model fits your data?",
         sub="Test your data and follow the path to OLS, VAR, VECM, ARDL, NARDL or GMM. Real examples and Python code.",
         tags=["Unit-root tests", "Cointegration", "Model selection", "Python"]),
    dict(out="assets/og/oes/index.jpg", kicker="Open Econometric Systems", title="Run every model on your data",
         sub="51 models: the four research guides plus Bayesian VARs and Markov switching, free in your browser.",
         tags=["Time series", "Panel data", "Causal inference", "Machine learning"]),
    dict(out="assets/og/teaching.jpg", kicker="Teaching portfolio", title="Courses & Learning Resources",
         sub="Six full course sites with lecture notes, coding exercises, problem sets and exams.",
         tags=["Econometrics", "Statistics", "Calculus", "Macroeconomics", "Finance", "Monetary"]),
]
for _g in (ROOT / "tools" / "model_selector" / "guide_cards.json", ROOT / "tools" / "og_cards_site.json"):
    if _g.exists():
        cards += json.loads(_g.read_text())
if os.environ.get("OG_ONLY"):  # comma-separated path prefixes, e.g. OG_ONLY=assets/og/dsge-guide
    cards = [c for c in cards if c["out"].startswith(tuple(os.environ["OG_ONLY"].split(",")))]
FONTS = os.environ.get("OG_FONT_CSS")
HTML = """<html><head>{HEAD}<style>
/* Designed for centre-cropping: phone apps often show only the middle 630x630 square, so all text lives there. */
body{margin:0;width:1200px;height:630px;overflow:hidden;font-family:'DM Sans',sans-serif;color:#fff;
 background:radial-gradient(circle at 50% -10%,#6ec1e450,transparent 45%),linear-gradient(125deg,#0033A0,#00205B);position:relative}
.ring{position:absolute;border:1px solid #ffffff1c;border-radius:50%}
.r1{width:560px;height:560px;left:-300px;top:-180px}.r2{width:560px;height:560px;right:-300px;bottom:-200px}.r3{width:300px;height:300px;right:-80px;top:-120px;border-color:#ffffff12}
.net{position:absolute;opacity:.45}
.col{position:absolute;left:300px;width:600px;top:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.brand{font:800 24px Manrope,sans-serif;letter-spacing:.08em;margin-bottom:26px}.brand span{color:#6EC1E4}
.eye{font-size:19px;font-weight:800;letter-spacing:.14em;color:#9fdaf2;text-transform:uppercase;line-height:1.35;max-width:560px}
h1{font:800 62px/1.08 Manrope,sans-serif;letter-spacing:-.035em;margin:16px 0 16px;max-width:580px;overflow-wrap:anywhere}
p{font-size:25px;line-height:1.35;color:#d9e5f7;margin:0;max-width:560px}
.who{margin-top:24px;font-size:18px;font-weight:700;color:#c9d8ef;padding:8px 18px;border:1px solid #ffffff40;border-radius:40px;background:#ffffff12}
</style></head><body><div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div>
<svg class="net" width="240" height="200" viewBox="0 0 240 200" style="left:30px;bottom:40px"><g fill="#a0daf2"><circle cx="20" cy="170" r="4"/><circle cx="80" cy="120" r="4"/><circle cx="140" cy="140" r="4"/><circle cx="200" cy="70" r="5"/><circle cx="110" cy="50" r="3"/></g><g stroke="#a0daf2" stroke-width="1.5" fill="none"><polyline points="20,170 80,120 140,140 200,70"/><line x1="80" y1="120" x2="110" y2="50" opacity=".5"/><line x1="110" y1="50" x2="200" y2="70" opacity=".5"/></g></svg>
<svg class="net" width="240" height="200" viewBox="0 0 240 200" style="right:30px;top:60px"><g fill="#a0daf2"><circle cx="30" cy="40" r="4"/><circle cx="90" cy="90" r="5"/><circle cx="160" cy="60" r="3"/><circle cx="210" cy="150" r="4"/><circle cx="120" cy="170" r="3"/></g><g stroke="#a0daf2" stroke-width="1.5" fill="none"><polyline points="30,40 90,90 160,60"/><line x1="90" y1="90" x2="210" y2="150" opacity=".5"/><line x1="90" y1="90" x2="120" y2="170" opacity=".5"/></g></svg>
<div class="col"><div class="brand">TGM <span>AL MAMUN</span></div><div class="eye" id="k"></div><h1 id="t"></h1><p id="s"></p><div class="who" id="w">Tuhin Al Mamun, Ph.D.</div></div></body></html>"""


async def main():
    head = f'<link rel="stylesheet" href="{FONTS}">' if FONTS else (
        '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700;800&family=Manrope:wght@800&display=swap" rel="stylesheet">')
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=os.environ.get("CHROME"), args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1200, "height": 630})
        if os.environ.get("OG_ROUTE"):  # optional offline font routing used in the build container
            exec(open(os.environ["OG_ROUTE"]).read(), globals()); await pg.route("**/*", ROUTE)  # noqa: F821
        await pg.set_content(HTML.replace("{HEAD}", head)); await pg.wait_for_timeout(800)
        for c in cards:
            await pg.evaluate("""c=>{k.textContent=c.kicker;t.textContent=c.title;s.textContent=c.sub||'';w.style.display=c.nowho?'none':'';
              const col=document.querySelector('.col');const used=()=>{const v=[...col.children].filter(e=>e.style.display!=='none');return v[v.length-1].getBoundingClientRect().bottom-v[0].getBoundingClientRect().top};s.style.display='';let f=64;t.style.fontSize=f+'px';
              const lines=()=>Math.round(t.getBoundingClientRect().height/(f*1.08));
              while((lines()>3||used()>560)&&f>34){f-=2;t.style.fontSize=f+'px'}
              if(used()>560)s.style.display='none';
              let g=25;s.style.fontSize=g+'px';while(s.style.display!=='none'&&s.getBoundingClientRect().height>g*1.35*2.2&&g>19){g-=1;s.style.fontSize=g+'px'}
              if(s.getBoundingClientRect().height>g*1.35*2.2||used()>560)s.style.display='none'}""", c)
            out = ROOT / c["out"]; out.parent.mkdir(parents=True, exist_ok=True)
            await pg.screenshot(path=str(out), type="jpeg", quality=82)
        await b.close()
    print(f"{len(cards)} preview images")

asyncio.run(main())
