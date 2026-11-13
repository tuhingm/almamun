"""Tests for the site search: index building, tokenizer parity between Python and JavaScript, ranking, curated answers."""
import json, pathlib, subprocess, sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_index as bi  # noqa: E402

JS = pathlib.Path(bi.ROOT) / "assets" / "ai" / "search.js"
PAGE = """<html><head><title>{t} | Site</title><meta name="description" content="{d}"></head><body><header>menu menu</header>
<main><h1>{t}</h1>{b}</main><script>var secret=1</script><footer>foot</footer></body></html>"""


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "ROOT", tmp_path)
    monkeypatch.setattr(bi, "OUT", tmp_path / "assets" / "ai")
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "tar.html").write_text(PAGE.format(t="Threshold regression", d="Estimate a threshold in a time series with a bootstrap test of linearity.",
        b='<h2 id="s1">The model</h2><p>The threshold regression splits the sample at an estimated level of the threshold variable. ' + "Each regime has its own coefficients. " * 6 + '</p><h2 id="s2">Breaks</h2><p>' + "Structural breaks happen at dates and not levels of a variable in the sample. " * 5 + '</p><pre>CODE_ONLY_TOKEN</pre>'))
    (tmp_path / "g" / "bp.html").write_text(PAGE.format(t="Breaks", d="Bai and Perron find several structural breaks at unknown dates in a series.", b='<h2 id="a">Dates</h2><p>' + "The Bai Perron procedure estimates break dates by dynamic programming over all partitions. " * 5 + "</p>"))
    (tmp_path / "old.html").write_text('<html><head><meta http-equiv="refresh" content="0;url=g/bp.html"></head><body>moved</body></html>')
    (tmp_path / "assets" / "ai").mkdir(parents=True)
    return tmp_path


def run_js(idx, query, k=3):
    code = "const S=require(%r);const idx=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(S.search(idx,%s,%d).map(h=>[h.url,h.faq])))" % (str(JS), json.dumps(query), k)
    r = subprocess.run(["node", "-e", code], input=json.dumps(idx), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def test_extraction_skips_menus_code_scripts_and_redirects(site):
    idx = bi.build()
    text = " ".join(c[3] for c in idx["chunks"])
    assert "menu" not in text and "CODE_ONLY_TOKEN" not in text and "secret" not in text and "foot" not in text and "moved" not in text
    assert {c[0] for c in idx["chunks"]} >= {"g/tar.html#s1", "g/tar.html#s2", "g/bp.html#a", "g/tar.html", "g/bp.html"}
    assert any(c[2] == "About this page" for c in idx["chunks"])


def test_inverted_index_is_consistent(site):
    idx = bi.build()
    assert idx["n"] == len(idx["chunks"]) == len(idx["len"])
    for term, post in idx["inv"].items():
        assert len(post) % 2 == 0 and all(0 <= post[i] < idx["n"] and post[i + 1] > 0 for i in range(0, len(post), 2))


@pytest.mark.skipif(subprocess.run(["which", "node"], capture_output=True).returncode != 0, reason="node not installed")
def test_tokenizer_parity_between_python_and_javascript():
    samples = ["Bai-Perron structural breaks", "Non-stationary series, I(1) and I(0)?", "Estimations of thresholds & the p-value 0.081", "log_ip and gdp.growth", "Teräsvirta's tests", "ESTAR/LSTAR models"]
    code = "const S=require(%r);console.log(JSON.stringify(%s.map(s=>S.tokens(s))))" % (str(JS), json.dumps(samples))
    js = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    assert js == [bi.tokens(s) for s in samples]


@pytest.mark.skipif(subprocess.run(["which", "node"], capture_output=True).returncode != 0, reason="node not installed")
def test_search_ranks_the_right_page_first(site):
    idx = bi.build()
    assert run_js(idx, "How do I estimate a threshold regression?")[0][0].startswith("g/tar.html")
    assert run_js(idx, "Bai Perron break dates")[0][0].startswith("g/bp.html")
    assert run_js(idx, "zzzz qqqq") == []
    assert run_js(idx, "") == []


@pytest.mark.skipif(subprocess.run(["which", "node"], capture_output=True).returncode != 0, reason="node not installed")
def test_curated_answers_outrank_page_text(site):
    (bi.OUT / "faq.json").write_text(json.dumps([{"q": "What is the bootstrap test for?", "a": "It tests linearity against a threshold.", "u": "g/tar.html", "k": "bootstrap"}]))
    idx = bi.build()
    assert idx["faq_from"] < idx["n"]
    top = run_js(idx, "what is the bootstrap test for")[0]
    assert top[1] is True


@pytest.mark.skipif(subprocess.run(["which", "node"], capture_output=True).returncode != 0, reason="node not installed")
def test_best_sentences_picks_matching_sentences():
    code = "const S=require(%r);console.log(JSON.stringify(S.bestSentences('Rain fell all day. The threshold is 4.23 percent. Nothing else matters here.','what is the threshold',1)))" % str(JS)
    assert json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout) == ["The threshold is 4.23 percent."]
