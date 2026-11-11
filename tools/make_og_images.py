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
    dict(out="assets/og/working-papers.jpg", kicker="Research portfolio · 2023–present", title="Working Papers",
         sub="Fifteen manuscripts on decarbonization, energy, inflation, monetary institutions and macro-finance.",
         tags=["Climate & energy", "Inflation", "Monetary institutions", "Macro-finance"]),
    dict(out="assets/og/research-guide.jpg", kicker="Empirical Research Guide", title="Which model fits your data?",
         sub="Test your data and follow the path to OLS, VAR, VECM, ARDL, NARDL or GMM. Real examples and Python code.",
         tags=["Unit-root tests", "Cointegration", "Model selection", "Python"]),
    dict(out="assets/og/teaching.jpg", kicker="Teaching portfolio", title="Courses & Learning Resources",
         sub="Six full course sites with lecture notes, coding exercises, problem sets and exams.",
         tags=["Econometrics", "Statistics", "Calculus", "Macroeconomics", "Finance", "Monetary"]),
]
FONTS = os.environ.get("OG_FONT_CSS")
HTML = """<html><head>{HEAD}<style>
body{margin:0;width:1200px;height:630px;overflow:hidden;font-family:'DM Sans',sans-serif;color:#fff;
 background:radial-gradient(circle at 84% 12%,#6ec1e455,transparent 32%),linear-gradient(125deg,#0033A0,#00205B);position:relative}
.ring{position:absolute;width:620px;height:620px;border:1px solid #ffffff1c;border-radius:50%;right:-200px;bottom:-380px}
.ring2{position:absolute;width:420px;height:420px;border:1px solid #ffffff14;border-radius:50%;right:-60px;top:-260px}
.top{position:absolute;left:80px;right:80px;top:64px;display:flex;justify-content:space-between;align-items:center}
.brand{font:800 22px Manrope,sans-serif;letter-spacing:.06em}.brand span{color:#6EC1E4}.who{font-size:18px;color:#c9d8ef}
.mid{position:absolute;left:80px;right:110px;top:150px;bottom:130px;display:flex;flex-direction:column;justify-content:center}
.eye{font-size:19px;font-weight:800;letter-spacing:.17em;color:#9fdaf2;text-transform:uppercase}
.eye:before{content:"";display:inline-block;width:44px;height:2px;background:#6EC1E4;vertical-align:middle;margin-right:16px}
h1{font:800 72px/1.06 Manrope,sans-serif;letter-spacing:-.045em;margin:20px 0 20px}
p{font-size:26px;line-height:1.4;color:#d9e5f7;margin:0}
.tags{position:absolute;left:80px;right:80px;bottom:62px;display:flex;gap:12px;flex-wrap:nowrap;overflow:hidden}
.tags span{white-space:nowrap;font-size:17px;font-weight:700;padding:9px 18px;border:1px solid #ffffff55;border-radius:40px;background:#ffffff14}
</style></head><body><div class="ring"></div><div class="ring2"></div>
<div class="top"><div class="brand">TGM <span>AL MAMUN</span></div><div class="who">Tuhin Al Mamun, Ph.D.</div></div>
<div class="mid"><div class="eye" id="k"></div><h1 id="t"></h1><p id="s"></p></div><div class="tags" id="g"></div></body></html>"""


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
            await pg.evaluate("""c=>{k.textContent=c.kicker;t.textContent=c.title;s.textContent=c.sub;
              g.innerHTML='';c.tags.forEach(x=>{const e=document.createElement('span');e.textContent=x;g.appendChild(e)});
              let f=76;t.style.fontSize=f+'px';while(t.getBoundingClientRect().height>2*f*1.12&&f>48){f-=4;t.style.fontSize=f+'px'}
              s.style.display='';if(document.querySelector('.mid').scrollHeight>document.querySelector('.mid').clientHeight)s.style.display='none';
              [...g.children].forEach(e=>{if(e.offsetLeft+e.offsetWidth>g.clientWidth)e.remove()})}""", c)
            out = ROOT / c["out"]; out.parent.mkdir(parents=True, exist_ok=True)
            await pg.screenshot(path=str(out), type="jpeg", quality=82)
        await b.close()
    print(f"{len(cards)} preview images")

asyncio.run(main())
