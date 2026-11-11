#!/usr/bin/env python3
"""Build the course websites under teaching/ from their Markdown sources.

Each course folder (teaching/<course>/) holds Markdown files: index.md, syllabus.md,
problem-sets-solutions.md, examinations.md, optional outline.md, and lectures/NN-*.md.
This script renders every one of them to an .html file next to it, in an
OpenCourseWare-style layout with rendered math, and generates three more pages per
course: calendar.html, lecture-notes.html and readings.html.

Usage:   python3 tools/build_courses.py
Needs:   pip install markdown pymdown-extensions jinja2 pyyaml

Writing math in the Markdown: inline \\( ... \\), display \\[ ... \\] or
\\begin{align} ... \\end{align}. Inside a table cell write \\mid instead of a bare |.
"""
from __future__ import annotations

import json
import datetime
import html
import pathlib
import re
import sys

import jinja2
import markdown
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = "https://tuhingm.github.io/almamun/"
TEACH = ROOT / "teaching"
TOOLS = ROOT / "tools"

MD_EXT = [
    "tables", "attr_list", "md_in_html", "footnotes", "fenced_code", "def_list",
    "sane_lists", "toc", "pymdownx.arithmatex",
]
MD_CFG = {
    "pymdownx.arithmatex": {"generic": True, "inline_syntax": ["round"], "block_syntax": ["square", "begin"]},
    "toc": {"permalink": False},
}

BOX_RULES = [
    ("objectives", re.compile(r"^(learning )?objectives|^core question|^motivation", re.I)),
    ("example", re.compile(r"^worked (example|illustration|problem)", re.I)),
    ("problems", re.compile(r"^(problem set|problems|exercises)", re.I)),
    ("lab", re.compile(r"^(computational )?lab\b|^computational lab|^coding exercises", re.I)),
    ("reading", re.compile(r"^(further |suggested |required )?readings?\b|^references", re.I)),
]
READING_RE = re.compile(r"^(further |suggested |required )?readings?\b", re.I)


def isolate_display_math(text: str) -> str:
    """Display math must stand as its own block: put blank lines around \\[ ... \\] and
    \\begin{env} ... \\end{env} blocks that start on their own line."""
    out, in_math, fence = [], False, False
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("```"):
            fence = not fence
        if not fence and not in_math and (st.startswith("\\[") or re.match(r"\\begin\{(equation|align|gather|multline)\*?\}", st)):
            if out and out[-1].strip():
                out.append("")
            in_math = not (st.endswith("\\]") and st != "\\[") and not re.search(r"\\end\{\w+\*?\}\s*$", st)
            out.append(line)
            if not in_math:
                out.append("")
            continue
        if in_math:
            out.append(line)
            if st.endswith("\\]") or re.search(r"\\end\{(equation|align|gather|multline)\*?\}\s*$", st):
                in_math = False
                out.append("")
            continue
        out.append(line)
    # a list that starts right under a line of text needs a blank line to be parsed as a list
    fixed = []
    for line in out:
        if (re.match(r"\s{0,3}(\d+\.|[-*+])\s", line) and fixed and fixed[-1].strip()
                and not re.match(r"\s*(\d+\.|[-*+])\s", fixed[-1]) and not fixed[-1].lstrip().startswith(("|", "#"))
                and not fixed[-1].startswith((" ", "\t"))):
            fixed.append("")
        fixed.append(line)
    return "\n".join(fixed)


def md_to_html(text: str) -> str:
    md = markdown.Markdown(extensions=MD_EXT, extension_configs=MD_CFG)
    out = md.convert(isolate_display_math(text))
    # local links: foo.md -> foo.html, dir/ index stays
    out = re.sub(r'href="(?!https?:)([^"#]*?)\.md(#[^"]*)?"', lambda m: f'href="{m.group(1)}.html{m.group(2) or ""}"', out)
    return out


def strip_h1(text: str) -> tuple[str, str]:
    lines = text.lstrip("﻿").splitlines()
    title = ""
    if lines and lines[0].lstrip().startswith("#") and not lines[0].lstrip().startswith("##"):
        title = lines[0].lstrip("# ").strip()
        lines = lines[1:]
    elif lines and lines[0].strip() and not lines[0].startswith("##"):
        # first line is a bare title (no '#')
        title = lines[0].strip()
        lines = lines[1:]
    return title, "\n".join(lines)


def box_sections(body: str) -> str:
    """Wrap each <h2> section whose heading matches a rule in <section class="box ...">."""
    parts = re.split(r"(?=<h2[ >])", body)
    out = []
    for p in parts:
        m = re.match(r"<h2[^>]*>(.*?)</h2>", p, re.S)
        if not m:
            out.append(p)
            continue
        head = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        head = re.sub(r"^\d+(\.\d+)*\.?\s*", "", head)
        cls = next((c for c, rx in BOX_RULES if rx.search(head)), None)
        out.append(f'<section class="box {cls}">{p}</section>' if cls else p)
    return "".join(out)


def h2_section(text: str, pred) -> str:
    """Return the Markdown body of the first '## ' section whose title satisfies pred."""
    blocks = re.split(r"(?m)^(?=## )", text)
    for b in blocks:
        if b.startswith("## "):
            head, _, rest = b.partition("\n")
            if pred(re.sub(r"^\d+\.\s*", "", head[3:].strip())):
                return rest.strip()
    return ""


def lecture_title(raw_title: str) -> tuple[str, str]:
    m = re.match(r"Lecture\s+(\d+)\s*[—–:-]\s*(.+)", raw_title)
    return (str(int(m.group(1))), m.group(2).strip()) if m else ("", raw_title)


def first_table(text: str) -> list[list[str]]:
    rows, started = [], False
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            rows.append(cells)
            started = True
        elif started:
            break
    return rows


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def inline_md(s: str) -> str:
    h = md_to_html(s)
    return re.sub(r"^<p>(.*)</p>$", r"\1", h.strip(), flags=re.S)


def build() -> int:
    cfg = yaml.safe_load((TOOLS / "courses.yml").read_text())
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TOOLS / "templates"), autoescape=False)
    tpl = env.get_template("course.html")
    year = datetime.date.today().year
    written = 0
    og_cards = []

    for key, meta in cfg["courses"].items():
        d = TEACH / key
        if not d.is_dir():
            print(f"skip {key}: no folder", file=sys.stderr)
            continue
        c = dict(meta)
        c["instructor"] = meta.get("instructor", cfg["instructor"])
        units = meta.get("units", [])

        # ---- lectures
        lectures = []
        for f in sorted((d / "lectures").glob("*.md")):
            text = f.read_text()
            raw_title, rest = strip_h1(text)
            if not rest.strip():
                print(f"warning: {f.relative_to(ROOT)} has no content; skipped", file=sys.stderr)
                continue
            num, title = lecture_title(raw_title)
            num = num or str(len(lectures) + 1)
            unit = next((u[2] for u in units if u[0] <= int(num) <= u[1]), "")
            words = len(re.findall(r"\w+", rest))
            obj = h2_section(rest, lambda h: re.match(r"(learning )?objectives", h, re.I))
            lectures.append(dict(num=num, title=title, file=f.name, unit=unit, md=rest,
                                 html=f.stem + ".html", minutes=max(1, round(words / 180)),
                                 summary=re.sub(r"\s+", " ", obj).strip()))
        c["lectures"] = lectures

        sections = []
        for s in cfg["sections"]:
            if s.get("optional") and not (d / s["file"]).exists():
                continue
            sections.append(s)

        def render(out_path: pathlib.Path, *, section, body, page_title, lec=None, prev=None, nxt=None):
            nonlocal written
            depth = len(out_path.relative_to(ROOT).parts) - 1
            root = "../" * depth
            base = "../" * (depth - 2) if depth > 2 else ""
            doc_title = f"Lecture {lec['num']}: {lec['title']}" if lec else page_title
            url = SITE + out_path.relative_to(ROOT).as_posix()
            if url.endswith("/index.html"):
                url = url[:-len("index.html")]
            desc = (f"Lecture {lec['num']} of {c['title']} ({lec['unit']}): full notes, problem set and coding exercises."
                    if lec else c["summary"])
            rel = out_path.relative_to(ROOT / "teaching").with_suffix(".jpg").as_posix()
            og_image = f"assets/og/{rel}"
            if lec:
                card = dict(kicker=f"{c['title']} · Lecture {lec['num']} of {len(c['lectures'])}", title=lec["title"],
                            sub=(lec["unit"] or "").replace("·", "—", 1) if lec["unit"] else c["subtitle"],
                            tags=["Lecture notes", "Problem set", "Coding exercises"])
            elif section == "home":
                card = dict(kicker=f"{c['code']} · {c['level']}", title=c["title"], sub=c["subtitle"], tags=c["topics"][:4])
            else:
                card = dict(kicker=f"{c['title']} · {c['code']}", title=page_title, sub=c["subtitle"],
                            tags=[f"{len(c['lectures'])} lectures", c["level"]])
            og_cards.append(dict(out=og_image, **card))
            jsonld = ""
            if section == "home":
                jsonld = json.dumps({
                    "@context": "https://schema.org", "@type": "Course", "name": c["title"],
                    "description": c["summary"], "url": url, "courseCode": c["code"],
                    "educationalLevel": c["level"], "inLanguage": "en", "isAccessibleForFree": True,
                    "teaches": c["topics"],
                    "provider": {"@type": "Person", "name": "Tuhin G. M. Al Mamun", "url": SITE},
                    "author": {"@type": "Person", "name": "Tuhin G. M. Al Mamun", "url": SITE},
                    "hasPart": [{"@type": "LearningResource", "name": f"Lecture {l['num']}: {l['title']}",
                                 "url": url + "lectures/" + l["html"]} for l in c["lectures"]],
                }, ensure_ascii=False, indent=1)
            page = tpl.render(c=c, sections=sections, section=section, body=body, page_title=esc(page_title),
                              doc_title=esc(doc_title), lec=lec, prev=prev, next=nxt, root=root, base=base,
                              year=year, url=url, desc=esc(desc), jsonld=jsonld, site=SITE, og_image=og_image)
            out_path.write_text(page)
            written += 1

        def units_html() -> str:
            rows = []
            for u in units:
                ls = [l for l in lectures if u[0] <= int(l["num"]) <= u[1]]
                if not ls:
                    continue
                span = f"Lecture {u[0]}" if u[0] == u[1] else f"Lectures {u[0]}–{u[1]}"
                items = "".join(f'<li><a href="lectures/{l["html"]}">{esc(l["title"])}</a></li>' for l in ls)
                rows.append(f'<div class="unit"><h3>{esc(u[2])} <small>{span}</small></h3><ol start="{u[0]}">{items}</ol></div>')
            return '<div class="units">' + "".join(rows) + "</div>"

        # ---- markdown pages
        for s in sections:
            if s.get("generated"):
                continue
            src = d / s["file"]
            if not src.exists():
                print(f"warning: missing {src.relative_to(ROOT)}", file=sys.stderr)
                continue
            title, rest = strip_h1(src.read_text())
            body = md_to_html(rest)
            body = body.replace("<!-- units -->", units_html())
            if s["key"] != "home":
                body = box_sections(body)
            # strip the course name from section titles like "Econometrics — Syllabus"
            ptitle = re.sub(rf"^{re.escape(c['title'])}\s*[—–:-]\s*", "", title) or s["label"]
            if s["key"] in ("assignments", "exams", "syllabus"):
                ptitle = s["label"] if s["key"] != "assignments" else "Assignments: Problem Sets and Solutions"
                if s["key"] == "exams":
                    ptitle = "Exams"
            render(d / s["href"], section=s["key"], body=body, page_title=ptitle)

        # ---- lecture pages
        for i, l in enumerate(lectures):
            body = box_sections(md_to_html(l["md"]))
            prev = lectures[i - 1] if i > 0 else None
            nxt = lectures[i + 1] if i + 1 < len(lectures) else None
            render(d / "lectures" / l["html"], section="lecture-notes", body=body, page_title=l["title"],
                   lec=l, prev=prev, nxt=nxt)

        # ---- lecture notes index
        rows, cur = [], None
        for l in lectures:
            if l["unit"] and l["unit"] != cur:
                cur = l["unit"]
                rows.append(f'<tr class="unit-row"><td colspan="3">{esc(cur)}</td></tr>')
            summ = f'<div class="sum">{inline_md(l["summary"])}</div>' if l["summary"] else ""
            rows.append(f'<tr><td class="num">{l["num"]}</td><td><a href="lectures/{l["html"]}">{esc(l["title"])}</a>{summ}</td>'
                        f'<td class="fmt">{l["minutes"]} min read</td></tr>')
        body = ('<p>Complete written notes for each session. Every set of notes opens with learning objectives and '
                'closes with a problem set, a computational lab and readings. Use <em>Print / save as PDF</em> on any '
                'lecture page for an offline copy.</p>'
                '<table class="sessions"><thead><tr><th>Lec #</th><th>Topic and objectives</th><th>Length</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table>")
        render(d / "lecture-notes.html", section="lecture-notes", body=body, page_title="Lecture Notes")

        # ---- calendar (from the first table in the syllabus)
        syl = (d / "syllabus.md").read_text() if (d / "syllabus.md").exists() else ""
        table = first_table(syl)
        acts = {}
        if table:
            header, *trs = table
            for r in trs:
                if r and re.fullmatch(r"\d+", r[0]):
                    acts[r[0]] = r[-1] if len(r) > 2 else ""
        act_head = table[0][-1] if table and len(table[0]) > 2 else "Key activities"
        rows, cur = [], None
        for l in lectures:
            if l["unit"] and l["unit"] != cur:
                cur = l["unit"]
                rows.append(f'<tr class="unit-row"><td colspan="3">{esc(cur)}</td></tr>')
            rows.append(f'<tr><td class="num">{l["num"]}</td><td><a href="lectures/{l["html"]}">{esc(l["title"])}</a></td>'
                        f'<td>{inline_md(acts.get(l["num"], ""))}</td></tr>')
        body = ('<p>One lecture session per week. Problem sets are due one week after the sessions they cover; see '
                '<a href="problem-sets-solutions.html">Assignments</a> and <a href="examinations.html">Exams</a>.</p>'
                f'<table class="sessions"><thead><tr><th>Ses #</th><th>Topic</th><th>{esc(act_head)}</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table>")
        render(d / "calendar.html", section="calendar", body=body, page_title="Calendar")

        # ---- readings
        parts = ['<p>Readings are listed by session. Core textbooks for the whole course are listed at the end.</p>']
        for l in lectures:
            r = h2_section(l["md"], lambda h: READING_RE.match(h))
            if r:
                parts.append(f'<h2 id="lec-{l["num"]}">Lecture {l["num"]}: <a href="lectures/{l["html"]}">{esc(l["title"])}</a></h2>')
                parts.append(md_to_html(r))
        core = h2_section(syl, lambda h: re.match(r"core|references|textbooks", h, re.I))
        papers = h2_section(syl, lambda h: re.match(r"recommended research papers", h, re.I))
        if core:
            parts.append('<h2 id="core">Core textbooks</h2>' + md_to_html(core))
        if papers:
            parts.append('<h2 id="papers">Recommended research papers</h2>' + md_to_html(papers))
        render(d / "readings.html", section="readings", body="".join(parts), page_title="Readings")
        print(f"{key}: {len(lectures)} lectures")

    (TOOLS / "og_cards.json").write_text(json.dumps(og_cards, ensure_ascii=False, indent=0))
    print(f"wrote {written} pages")
    return 0


if __name__ == "__main__":
    rc = build()
    import subprocess
    subprocess.run([sys.executable, str(TOOLS / "og_tags.py")], check=True)  # complete link-preview tags
    sys.exit(rc)
