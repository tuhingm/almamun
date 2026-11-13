"""Tests for the site steward on a small synthetic site (run: python3 -m pytest -q tools/steward)."""
import json, pathlib, sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import steward  # noqa: E402

PAGE = """<!doctype html><html lang="en"><head><title>{t}</title><meta name="description" content="{d}">
<link rel="canonical" href="https://tuhingm.github.io/almamun/{u}"></head><body><h1>{t}</h1>{extra}{body}</body></html>"""
DESC = "A page description that is comfortably within the recommended length for search results."


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(steward, "ROOT", tmp_path)
    monkeypatch.setattr(steward, "HERE", tmp_path / "tools" / "steward")
    (tmp_path / "tools" / "steward").mkdir(parents=True)
    g = tmp_path / "guide"
    g.mkdir()
    for i in range(12):
        extra = '<aside class="author-note">note</aside>' if i != 11 else ""
        body = '<a href="p1.html#top">ok</a><img src="x.png" alt="x">' if i else '<a href="missing.html">bad</a><a href="p1.html#nope">anchor</a><img src="x.png">'
        (g / f"p{i}.html").write_text(PAGE.format(t=f"Page {i}", d=DESC, u=f"guide/p{i}.html", extra=extra, body=body))
    (g / "p1.html").write_text(PAGE.format(t="Page 1", d=DESC, u="guide/p1.html", extra='<aside class="author-note">n</aside>', body='<h2 id="top">t</h2><img src="x.png" alt="x">'))
    (g / "x.png").write_bytes(b"png")
    (tmp_path / "sitemap.xml").write_text('<urlset><url><loc>https://tuhingm.github.io/almamun/guide/p0.html</loc></url><url><loc>https://tuhingm.github.io/almamun/gone.html</loc></url></urlset>')
    return tmp_path


def kinds(findings):
    return {(f[0], f[2].split("/")[-1], f[3][:30]) for f in findings}


def test_hard_checks(site):
    F = steward.audit(steward.Site())
    checks = {f[0] for f in F}
    assert {"broken link", "broken anchor", "accessibility", "sitemap"} <= checks
    assert any(f[0] == "broken link" and "missing.html" in f[3] for f in F)
    assert any(f[0] == "broken anchor" and "#nope" in f[3] for f in F)
    assert any(f[0] == "accessibility" and "alt" in f[3] for f in F)
    assert any(f[0] == "sitemap" and "gone.html" in f[3] for f in F)
    assert any(f[0] == "sitemap" and "missing from" in f[3] for f in F)


def test_learned_convention_flags_the_outlier(site):
    F = steward.audit(steward.Site())
    conv = [f for f in F if f[0] == "convention"]
    assert len(conv) == 1 and conv[0][2] == "guide/p11.html" and "author's note" in conv[0][3]


def test_memory_tracks_new_persistent_resolved(site):
    today = "2026-01-01"
    F1 = steward.audit(steward.Site())
    mem, new, persistent, resolved, n = steward.learn(F1, today)
    assert n == 1 and len(new) == len(F1) and not persistent and not resolved
    mem, new, persistent, resolved, n = steward.learn(F1, today)
    assert n == 2 and not new and len(persistent) == len(F1) and all(r["runs_seen"] == 2 for _, r in persistent)
    (site / "guide" / "p11.html").write_text(PAGE.format(t="Page 11", d=DESC, u="guide/p11.html", extra='<aside class="author-note">n</aside>', body=""))
    F3 = steward.audit(steward.Site())
    mem, new, persistent, resolved, n = steward.learn(F3, today)
    assert any(r["check"] == "convention" for r in resolved)
    assert steward.priority("error", 5) > steward.priority("error", 1) > steward.priority("warning", 1)


def test_og_fix_repairs_the_naming_form(site):
    (site / "assets" / "og").mkdir(parents=True)
    (site / "assets" / "og" / "guide-p0.jpg").write_bytes(b"jpg")
    p = site / "guide" / "p0.html"
    p.write_text(p.read_text().replace("</head>", '<meta property="og:image" content="https://tuhingm.github.io/almamun/assets/og/guide/p0.jpg"></head>'))
    assert any(f[0] == "og image" for f in steward.audit(steward.Site()))
    fixed = steward.fix_og(steward.Site())
    assert fixed and "guide-p0.jpg" in p.read_text()
    assert not any(f[0] == "og image" for f in steward.audit(steward.Site()))


def test_report_is_written(site):
    F = steward.audit(steward.Site())
    mem, new, persistent, resolved, n = steward.learn(F, "2026-01-01")
    steward.report(F, mem, new, persistent, resolved, n, "2026-01-01", ["example fix"])
    txt = (site / "tools" / "steward" / "REPORT.md").read_text()
    assert "Fix first" in txt and "example fix" in txt and json.loads((site / "tools" / "steward" / "memory.json").read_text())["runs"]
