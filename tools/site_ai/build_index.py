#!/usr/bin/env python3
"""Build the search index that powers the "Ask this site" box (assets/ai/index.json).

No server and no external AI: the index is a static file with an inverted index (BM25), built from the text of every page.
The visitor's browser downloads it the first time the box is opened and answers by quoting the best passages with links.
Curated answers in assets/ai/faq.json (written by hand or exported from the personal assistant with `personal-ai export-faq`)
are added as high-priority passages, so corrected answers beat raw page text.

    python3 tools/site_ai/build_index.py
"""
import collections, json, pathlib, re, sys

from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "ai"
SKIP_DIRS = {"tools", ".git", ".github", "node_modules", "__pycache__", ".pytest_cache"}
STOP = set("a an the of and or to in on for with is are was were be been it its this that these those as at by from we you i not but if then so than which who what when where how do does did can could should would will may might has have had into about over under also".split())
WORD = re.compile(r"[a-z0-9]+(?:[_.][a-z0-9]+)*")
CHUNK_WORDS = 110


def stem(w):
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokens(text):
    return [stem(w) for w in WORD.findall(text.lower()) if w not in STOP]


def pages():
    for p in sorted(ROOT.rglob("*.html")):
        rel = p.relative_to(ROOT).as_posix()
        if p.parts[len(ROOT.parts)] in SKIP_DIRS or rel.startswith("oes/engine/") or "/examples/" in rel or rel.startswith("google"):
            continue
        yield p, rel


def page_chunks(path, rel):
    s = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    if s.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}):
        return []
    for t in s(["script", "style", "nav", "header", "footer", "pre", "noscript", "svg", "form", "button", "aside"]):
        t.decompose()
    title = re.sub(r"\s*\|.*$", "", (s.title.string or rel).strip()) if s.title else rel
    url = rel[:-len("index.html")] if rel.endswith("index.html") else rel
    desc = s.find("meta", attrs={"name": "description"})
    sections, cur = [], dict(h="", id="", words=[])
    root = s.find("main") or s.body or s
    for el in root.find_all(["h1", "h2", "h3", "p", "li", "td", "th", "dd", "blockquote", "figcaption"]):
        if el.name in ("h1", "h2", "h3"):
            if cur["words"]:
                sections.append(cur)
            txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            cur = dict(h=re.sub(r"^Step \d+\s*", "", txt), id=el.get("id") or "", words=[])
        elif not el.find(["p", "li"]) or el.name in ("p", "li"):
            txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if txt:
                cur["words"].append(txt)
    if cur["words"]:
        sections.append(cur)
    out = []
    for sec in sections:
        buf, n = [], 0
        parts = []
        for para in sec["words"]:
            buf.append(para)
            n += len(para.split())
            if n >= CHUNK_WORDS:
                parts.append(" ".join(buf)); buf, n = [], 0
        if buf and (n >= 8 or not parts):
            parts.append(" ".join(buf))
        elif buf:
            parts[-1] += " " + " ".join(buf)
        for t in parts:
            if len(t.split()) >= 6:
                out.append([url + (f"#{sec['id']}" if sec["id"] else ""), title, sec["h"], t])
    if desc and desc.get("content") and len(desc["content"].split()) >= 6:
        out.insert(0, [url, title, "About this page", desc["content"].strip()])
    return out


def build():
    chunks, seen = [], set()
    for p, rel in pages():
        for c in page_chunks(p, rel):
            key = c[3][:160]
            if key in seen:
                continue
            seen.add(key)
            chunks.append(c)
    faq_path = OUT / "faq.json"
    faq = json.loads(faq_path.read_text(encoding="utf-8")) if faq_path.exists() else []
    first_faq = len(chunks)
    for f in faq:   # curated answers: stored as chunks flagged by position (>= first_faq)
        chunks.append([f.get("u", ""), "Answer", f["q"], f["a"], f.get("k", "")])
    inv = collections.defaultdict(list)
    lens = []
    for i, c in enumerate(chunks):
        extra = (c[4] if len(c) > 4 else "")
        toks = tokens(c[3]) + tokens(c[2]) * 2 + tokens(c[1]) + tokens(extra) * 2
        lens.append(len(toks))
        for term, tf in collections.Counter(toks).items():
            inv[term].append(i)
            inv[term].append(tf)
    return dict(v=1, n=len(chunks), faq_from=first_faq, avg=round(sum(lens) / max(len(lens), 1), 2), len=lens,
                chunks=[c[:4] for c in chunks], inv=inv)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = build()
    (OUT / "index.json").write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = (OUT / "index.json").stat().st_size
    print(f"site index: {idx['n']} passages ({len(idx['inv'])} terms), {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
