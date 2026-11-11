#!/usr/bin/env python3
"""Render a narrated slide video from a script module.  usage: render.py <lecture_dir> [--frames-only]"""
import asyncio, importlib.util, json, os, pathlib, subprocess, sys, html, re
import numpy as np, soundfile as sf

D = pathlib.Path(sys.argv[1]).resolve()
FRAMES_ONLY = "--frames-only" in sys.argv
spec = importlib.util.spec_from_file_location("script", D / "script.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W, H, FPS, XF = 1920, 1080, 30, 0.4
FONTS = "/home/claude/scratch/node_modules/@fontsource-variable"
MJ = "/home/claude/scratch/node_modules/mathjax/es5"
OUT = D / "build"; OUT.mkdir(exist_ok=True)

CSS = """
@font-face{font-family:'DM Sans';font-weight:100 1000;src:url(fonts/dm-sans/files/dm-sans-latin-wght-normal.woff2)}
@font-face{font-family:'Manrope';font-weight:200 800;src:url(fonts/manrope/files/manrope-latin-wght-normal.woff2)}
:root{--blue:#0033A0;--deep:#00205B;--sky:#6EC1E4;--ink:#17233b;--muted:#596780;--body:#3b4961;--line:#dbe3ef;--gold:#a8650f;--green:#1f7a4d}
*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;overflow:hidden}
body{font:38px/1.5 'DM Sans',sans-serif;color:var(--body);background:#fff}
.slide{position:absolute;inset:0;padding:70px 120px 0;display:flex;flex-direction:column}
.top{display:flex;justify-content:space-between;align-items:center;font:800 19px 'DM Sans';letter-spacing:.2em;text-transform:uppercase;color:var(--blue)}
.top .r{color:#9aa6bd}
.top .k:before{content:"";display:inline-block;width:46px;height:2px;background:var(--blue);vertical-align:middle;margin-right:18px}
h1{font:800 70px/1.1 Manrope;letter-spacing:-.045em;color:var(--ink);margin:26px 0 34px}
.content{flex:1}
.bar{position:absolute;left:0;right:0;bottom:0;height:64px;background:var(--deep);color:#c5d2e7;display:flex;align-items:center;justify-content:space-between;padding:0 120px;font:600 19px 'DM Sans';letter-spacing:.06em}
.bar b{color:#fff;font-family:Manrope}.bar .u{color:var(--sky)}
.prog{position:absolute;left:0;bottom:64px;height:6px;background:var(--sky)}
[data-s]{transition:none}.hide{opacity:0}
.big li{margin:14px 0;font-size:38px}ol.big,ul.big{padding-left:56px}
li::marker{color:var(--blue);font-weight:800}
.def,.thm{background:#f3f6fb;border-left:7px solid var(--blue);border-radius:16px;padding:22px 36px;margin:0 0 26px}
.def{border-left-color:var(--green);background:#f2f8f5}
.proof{border-left:4px solid var(--line);padding:6px 0 6px 36px;margin:0 0 24px}
.eq-big{font-size:48px;margin:4px 0 22px}
.note{background:#fbf7ef;border-radius:14px;padding:18px 30px;border-left:7px solid var(--gold);margin:10px 0}
.facts li{margin:16px 0}
.split{display:grid;grid-template-columns:1.05fr 1fr;gap:60px;align-items:center}
.fig{width:100%;max-height:690px}.side p{margin:0 0 24px}
.tbl{border-collapse:collapse;width:100%;font-size:35px}
.tbl th{font:800 19px 'DM Sans';letter-spacing:.14em;text-transform:uppercase;color:var(--deep);background:#f3f6fb;text-align:left;padding:18px 24px}
.tbl td{padding:18px 24px;border-bottom:2px solid var(--line)}
.tbl tr.warn td{background:#fbf7ef;color:var(--ink);font-weight:600}
.next{margin-top:30px;background:var(--blue);color:#fff;border-radius:16px;padding:24px 36px;font:700 34px Manrope;display:inline-block}
.title{position:absolute;inset:0;color:#fff;padding:0 140px;display:flex;flex-direction:column;justify-content:center;
 background:radial-gradient(circle at 82% 15%,#6ec1e455,transparent 30%),linear-gradient(125deg,var(--blue),var(--deep))}
.title:after{content:"";position:absolute;width:900px;height:900px;border:2px solid #ffffff18;border-radius:50%;right:-300px;bottom:-620px}
.title .e{font:800 24px 'DM Sans';letter-spacing:.22em;text-transform:uppercase;color:#9fdaf2}
.title .e:before{content:"";display:inline-block;width:60px;height:2px;background:var(--sky);vertical-align:middle;margin-right:22px}
.title h1{color:#fff;font-size:104px;letter-spacing:-.055em;margin:30px 0 26px;max-width:1500px}
.title .s{font-size:38px;color:#d9e5f7}
.title .who{position:absolute;left:140px;bottom:90px;font:700 26px Manrope;color:#fff}.title .who span{display:block;font:500 22px 'DM Sans';color:#a8def2;margin-top:6px}
mjx-container{font-size:100%!important}
"""

def slide_html(i, sl):
    n = len(S.SLIDES)
    prog = f'<div class="prog" style="width:{(i+1)/n*100:.2f}%"></div>'
    bar = f'<div class="bar"><span><b>{S.COURSE}</b> · {S.LECTURE} · {html.escape(S.TITLE)}</span><span class="u">tuhingm.github.io/almamun</span></div>'
    if sl.get("kind") == "title":
        return f'<div class="title"><div class="e">{S.COURSE} · {S.LECTURE}</div><h1>{S.TITLE}</h1><div class="s">Graduate course · lecture notes, problem set and lab online</div><div class="who">Tuhin G. M. Al Mamun, Ph.D.<span>tuhingm.github.io/almamun</span></div></div>'
    return (f'<div class="slide"><div class="top"><span class="k">{sl["kicker"]}</span><span class="r">{S.COURSE} · {S.LECTURE}</span></div>'
            f'<h1>{sl["title"]}</h1><div class="content">{sl["body"]}</div></div>{prog}{bar}')

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},chtml:{{scale:1}},startup:{{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{{document.body.dataset.ready=1}})}}}};</script>
<script src="mj/tex-chtml.js"></script></head><body>{body}</body></html>"""

async def draw():
    from playwright.async_api import async_playwright
    for name, target in (("fonts", FONTS), ("mj", MJ)):
        p = D / name
        if not p.exists(): p.symlink_to(target)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox", "--allow-file-access-from-files"])
        pg = await b.new_page(viewport={"width": W, "height": H})
        for i, sl in enumerate(S.SLIDES):
            f = D / f"_slide{i:02d}.html"
            f.write_text(PAGE.format(css=CSS, body=slide_html(i, sl)))
            await pg.goto(f.as_uri()); await pg.wait_for_selector("body[data-ready]", timeout=20000)
            await pg.wait_for_timeout(150)
            for k in range(len(sl["say"])):
                await pg.evaluate(f"document.querySelectorAll('[data-s]').forEach(e=>e.classList.toggle('hide', +e.dataset.s>{k}))")
                await pg.screenshot(path=str(OUT / f"s{i:02d}_{k}.png"))
        await b.close()

def tts():
    from kokoro_onnx import Kokoro
    k = Kokoro("/home/claude/video/tts/kokoro-v1.0.onnx", "/home/claude/video/tts/voices-v1.0.bin")
    meta = []
    for i, sl in enumerate(S.SLIDES):
        for j, text in enumerate(sl["say"]):
            wav = OUT / f"a{i:02d}_{j}.wav"
            if not wav.exists():
                a, sr = k.create(text, voice=getattr(S, "VOICE", "am_michael"), speed=getattr(S, "SPEED", 1.0), lang="en-us")
                pre, post = 0.35, 0.45 if j < len(sl["say"]) - 1 else 0.85
                a = np.concatenate([np.zeros(int(pre*sr)), a, np.zeros(int(post*sr))]).astype(np.float32)
                sf.write(wav, a, sr)
            info = sf.info(wav); meta.append((i, j, info.frames / info.samplerate, text))
    return meta

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)

def encode(meta):
    files, prev = [], None
    for i, j, dur, _ in meta:
        cur = OUT / f"s{i:02d}_{j}.png"; out = OUT / f"v{i:02d}_{j}.mp4"; tmp = OUT / f"tmp_{i:02d}_{j}.mp4"
        frames = round(dur * FPS); d = frames / FPS
        enc = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]
        if prev is None:
            fc = f"[0]format=yuv420p,loop=-1:1,trim=duration={d},setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.6[v]"
            run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(cur), "-filter_complex", fc, "-map", "[v]", "-frames:v", str(frames), *enc, str(tmp)])
        else:
            fc = (f"[0]format=yuv420p,loop=-1:1,trim=duration={XF+0.1},setpts=PTS-STARTPTS[a];"
                  f"[1]format=yuv420p,loop=-1:1,trim=duration={d},setpts=PTS-STARTPTS[b];[a][b]xfade=transition=fade:duration={XF}:offset=0[v]")
            run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(prev), "-framerate", str(FPS), "-i", str(cur), "-filter_complex", fc, "-map", "[v]", "-frames:v", str(frames), *enc, str(tmp)])
        os.replace(tmp, out); files.append((out, d)); prev = cur
    # concat video
    (OUT / "list.txt").write_text("".join(f"file '{f.name}'\n" for f, _ in files))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(OUT / "list.txt"), "-c", "copy", str(OUT / "video_only.mp4")])
    # audio padded to exact video durations
    sr = sf.info(OUT / "a00_0.wav").samplerate; chunks = []
    for (i, j, _, _), (_, d) in zip(meta, files):
        a, _ = sf.read(OUT / f"a{i:02d}_{j}.wav", dtype="float32"); n = round(d * sr)
        chunks.append(np.pad(a, (0, max(0, n - len(a))))[:n])
    sf.write(OUT / "narration.wav", np.concatenate(chunks), sr)
    final = D / f"{S.OUTNAME}.mp4"; tmp = D / f"{S.OUTNAME}.tmp.mp4"
    run(["ffmpeg", "-y", "-i", str(OUT / "video_only.mp4"), "-i", str(OUT / "narration.wav"), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-af", "loudnorm=I=-16:TP=-1.5", "-movflags", "+faststart", "-shortest", str(tmp)])
    os.replace(tmp, final)
    # WebVTT captions, one cue per sentence, timed by character share within each step
    def ts(t): h=int(t//3600); m=int(t%3600//60); s=t%60; return f"{h:02d}:{m:02d}:{s:06.3f}"
    cues, t = [], 0.0
    for (i, j, _, text), (_, d) in zip(meta, files):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        span = d - 0.35 - 0.45; total = sum(len(s) for s in sents); u = t + 0.35
        for s in sents:
            L = span * len(s) / total; cues.append((u, u + L, s)); u += L
        t += d
    (D / f"{S.OUTNAME}.vtt").write_text("WEBVTT\n\n" + "\n".join(f"{ts(a)} --> {ts(b)}\n{s}\n" for a, b, s in cues))
    print("duration", round(t, 1), "s ->", final)

asyncio.run(draw())
if FRAMES_ONLY: sys.exit(0)
encode(tts())
