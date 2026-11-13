#!/usr/bin/env python3
"""Site steward: audits the site, learns its conventions, remembers what it found, and fixes the safe issues.

No external AI and no network access are used. It is a set of deterministic checks plus a small memory:

  * hard checks      broken links and anchors, missing images, titles and descriptions, alt text, heading order,
                     sitemap and Open Graph files, large images, model counts that disagree with the OES catalog;
  * learned norms    for each folder it measures which features most pages share (author note, canonical link,
                     Open Graph image, share bar, ...) and flags the pages that lack a feature that 90% or more of
                     their siblings have. New page types therefore get a standard without anyone writing a rule;
  * memory           memory.json keeps every finding with the run it first appeared in, so the report separates
                     new problems, persistent ones and resolved ones, and ranks them by severity times persistence;
  * safe fixes       with --fix it reruns the site's own generators (author note, sitemap) and repairs Open Graph
                     paths. Everything else is only reported.

    python3 tools/steward/steward.py            audit and write tools/steward/REPORT.md
    python3 tools/steward/steward.py --fix      also apply the safe fixes, then audit again
"""
import argparse, collections, datetime, json, math, pathlib, re, subprocess, sys, urllib.parse

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.exit("steward needs beautifulsoup4 and lxml: pip install beautifulsoup4 lxml")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = "https://tuhingm.github.io/almamun/"
SKIP_DIRS = {"tools", ".git", ".github", "node_modules", "__pycache__", ".pytest_cache"}
SKIP_PREFIX = ("oes/engine/",)
SEVERITY = {"error": 3, "warning": 2, "note": 1}
NORM_FEATURES = {
    "canonical link": lambda s: bool(s.find("link", rel="canonical")),
    "Open Graph image": lambda s: bool(s.find("meta", property="og:image")),
    "meta description": lambda s: bool(s.find("meta", attrs={"name": "description"})),
    "author's note": lambda s: bool(s.find(class_="author-note")),
    "structured data": lambda s: bool(s.find("script", type="application/ld+json")),
    "share bar": lambda s: bool(s.find(class_=re.compile(r"share"))),
}


def pages():
    out = []
    for p in sorted(ROOT.rglob("*.html")):
        rel = p.relative_to(ROOT).as_posix()
        if p.parts[len(ROOT.parts)] in SKIP_DIRS or rel.startswith(SKIP_PREFIX) or "/examples/" in rel or rel.startswith("google"):
            continue
        out.append(p)
    return out


class Site:
    def __init__(self):
        self.files = pages()
        self.soup, self.ids = {}, {}
        for p in self.files:
            s = BeautifulSoup(p.read_text(encoding="utf-8", errors="replace"), "lxml")
            self.soup[p] = s
            self.ids[p] = {t.get("id") for t in s.find_all(id=True)} | {t.get("name") for t in s.find_all("a", attrs={"name": True})}

    def model_ids(self):
        """The OES page opens a model from the URL hash, so #<model id> is a valid target."""
        if not hasattr(self, "_mids"):
            c = ROOT / "oes" / "engine" / "catalog.json"
            try:
                d = json.loads(c.read_text())
                self._mids = {m["id"] for m in (d["models"] if isinstance(d, dict) and "models" in d else d)}
            except Exception:
                self._mids = set()
        return self._mids

    def rel(self, p):
        return p.relative_to(ROOT).as_posix()

    def is_redirect(self, s):
        return bool(s.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}))

    def resolve(self, page, href):
        """Local path for an href, or None when the link is external or not a file."""
        href = href.strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "data:", "#", "sms:")):
            return None
        if href.startswith(("http://", "https://")):
            if not href.startswith(SITE):
                return None
            href = "/" + href[len(SITE):]
        u = urllib.parse.urlsplit(href)
        path = urllib.parse.unquote(u.path)
        if path.startswith("/"):
            base = ROOT / path.lstrip("/")
        else:
            base = (page.parent / path).resolve() if path else page
        return base, u.fragment


def audit(site):
    F = []  # findings: (check, severity, page, message)

    def add(check, sev, page, msg):
        F.append((check, sev, page, msg))

    titles = collections.defaultdict(list)
    jumps = collections.Counter()
    for p, s in site.soup.items():
        rel = site.rel(p)
        red = site.is_redirect(s)
        # links, anchors, assets
        for tag, attr in (("a", "href"), ("link", "href"), ("script", "src"), ("img", "src"), ("source", "src")):
            for t in s.find_all(tag):
                v = t.get(attr)
                if v is None:
                    continue
                r = site.resolve(p, v)
                if not r:
                    continue
                target, frag = r
                if target.is_dir():
                    target = target / "index.html"
                if not target.exists():
                    add("broken link" if tag == "a" else "missing file", "error", rel, f"{tag} {attr}={v}")
                    continue
                if frag and tag == "a" and target == ROOT / "oes" / "index.html" and frag in site.model_ids():
                    continue
                if frag and tag == "a" and target.suffix == ".html" and target in site.ids and frag not in site.ids[target]:
                    add("broken anchor", "warning", rel, f"{v}")
        if red:
            continue
        # head
        title = (s.title.string or "").strip() if s.title and s.title.string else ""
        if not title:
            add("title", "error", rel, "missing <title>")
        else:
            titles[title].append(rel)
            if len(title) > 75:
                add("title", "note", rel, f"title is {len(title)} characters; search results cut it near 60")
        d = s.find("meta", attrs={"name": "description"})
        if d and d.get("content"):
            n = len(d["content"].strip())
            if n < 50 or n > 170:
                add("description", "note", rel, f"meta description is {n} characters (50 to 170 reads best)")
        if not (s.html and s.html.get("lang")):
            add("accessibility", "warning", rel, "<html> has no lang attribute")
        h1 = s.find_all("h1")
        if len(h1) != 1:
            add("headings", "warning", rel, f"{len(h1)} <h1> elements (one is expected)")
        last = 0
        for h in s.find_all(re.compile(r"^h[1-6]$")):
            lv = int(h.name[1])
            if last and lv > last + 1:
                jumps[rel.split("/")[0] if "/" in rel else "."] += 1
                break
            last = lv
        for im in s.find_all("img"):
            if im.get("alt") is None:
                add("accessibility", "warning", rel, f"image without alt text: {(im.get('src') or '')[:60]}")
        for a in s.find_all("a"):
            if not a.get_text(strip=True) and not a.get("aria-label") and not a.find("img") and not a.find("svg") and a.get("href"):
                add("accessibility", "note", rel, f"link with no text: {a.get('href')[:60]}")
                break
        og = s.find("meta", property="og:image")
        if og and og.get("content", "").startswith(SITE):
            f = ROOT / og["content"][len(SITE):].split("?")[0]
            if not f.exists():
                add("og image", "error", rel, f"og:image file missing: {og['content'][len(SITE):]}")
    for g, n in jumps.items():
        add("headings", "note", f"{g}/", f"{n} pages skip a heading level (for example h1 to h3); fine if it is the house style, but screen readers prefer an unbroken outline")
    for t, ps in titles.items():
        if len(ps) > 1:
            add("duplicate title", "note", ", ".join(ps[:3]), f"{len(ps)} pages share the title '{t[:50]}'")
    # sitemap
    sm = ROOT / "sitemap.xml"
    if sm.exists():
        locs = set(re.findall(r"<loc>([^<]+)</loc>", sm.read_text()))
        listed = {urllib.parse.unquote(u[len(SITE):]) for u in locs if u.startswith(SITE)}
        for u in sorted(listed):
            tp = ROOT / (u or "index.html")
            if u.endswith("/") or not u:
                tp = ROOT / u / "index.html"
            if not tp.exists():
                add("sitemap", "error", "sitemap.xml", f"lists a page that does not exist: {u}")
        for p in site.files:
            rel = site.rel(p)
            url = rel[:-len("index.html")] if rel.endswith("index.html") else rel
            if url not in listed and not site.is_redirect(site.soup[p]):
                add("sitemap", "warning", rel, "page is missing from sitemap.xml")
    # large images
    for im in ROOT.rglob("*"):
        if im.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif") and not any(x in SKIP_DIRS for x in im.parts) and im.stat().st_size > 600_000 and "/og/" not in im.as_posix():
            add("page weight", "note", im.relative_to(ROOT).as_posix(), f"image is {im.stat().st_size // 1000} KB")
    # OES catalog against the pages that describe it
    cat = ROOT / "oes" / "engine" / "catalog.json"
    if cat.exists():
        c = json.loads(cat.read_text())
        models = c["models"] if isinstance(c, dict) and "models" in c else c
        n = len(models)
        for path in ("oes/index.html",):
            txt = (ROOT / path).read_text(errors="replace")
            for m in re.finditer(r"\b(\d{2,3}) (?:econometric )?models\b", txt):
                if int(m.group(1)) != n:
                    add("count", "error", path, f"says {m.group(1)} models, the catalog has {n}")
                    break
        for mod in models:
            g = mod.get("guide")
            if g and not (ROOT / g).exists():
                add("guide link", "error", "oes/engine/catalog.json", f"model {mod.get('id')} points to a missing guide: {g}")
    # learned norms: features nearly every sibling page has
    groups = collections.defaultdict(list)
    for p in site.files:
        if site.is_redirect(site.soup[p]):
            continue
        rel = site.rel(p)
        groups[rel.split("/")[0] if "/" in rel else "."].append(p)
    for g, ps in groups.items():
        if len(ps) < 5:
            continue
        for name, test in NORM_FEATURES.items():
            have = [p for p in ps if test(site.soup[p])]
            share = len(have) / len(ps)
            if 0.9 <= share < 1:
                for p in ps:
                    if p not in have:
                        add("convention", "warning", site.rel(p), f"lacks the {name} that {int(share * 100)}% of the pages in {g}/ have")
    return F


def fingerprint(f):
    return f"{f[0]}|{f[2]}|{f[3]}"


def learn(findings, today):
    mem_path = HERE / "memory.json"
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else {"runs": [], "issues": {}}
    run_no = len(mem["runs"]) + 1
    current = {fingerprint(f): f for f in findings}
    new, persistent = [], []
    for fp, f in current.items():
        rec = mem["issues"].get(fp)
        if rec and rec.get("open", True):
            rec["last_run"], rec["runs_seen"] = run_no, rec.get("runs_seen", 1) + 1
            persistent.append((f, rec))
        else:
            mem["issues"][fp] = dict(check=f[0], severity=f[1], page=f[2], message=f[3], first_run=run_no, first_date=today, last_run=run_no, runs_seen=1, open=True)
            new.append((f, mem["issues"][fp]))
    resolved = []
    for fp, rec in mem["issues"].items():
        if rec.get("open", True) and fp not in current:
            rec["open"], rec["resolved_run"], rec["resolved_date"] = False, run_no, today
            resolved.append(rec)
    counts = collections.Counter(f[1] for f in findings)
    mem["runs"].append(dict(date=today, errors=counts["error"], warnings=counts["warning"], notes=counts["note"], pages=len(pages())))
    mem["runs"] = mem["runs"][-60:]
    # keep resolved records for a while only
    mem["issues"] = {k: v for k, v in mem["issues"].items() if v.get("open", True) or v.get("resolved_run", 0) > run_no - 20}
    mem_path.write_text(json.dumps(mem, indent=1, sort_keys=True))
    return mem, new, persistent, resolved, run_no


def priority(sev, runs_seen):
    return SEVERITY[sev] * (1 + math.log(runs_seen))


def report(findings, mem, new, persistent, resolved, run_no, today, fixed):
    lines = [f"# Site steward report, {today} (run {run_no})", ""]
    c = collections.Counter(f[1] for f in findings)
    lines += [f"**{c['error']} errors, {c['warning']} warnings, {c['note']} notes** across {len(pages())} pages. New: {len(new)}. Persisting: {len(persistent)}. Resolved since the last run: {len(resolved)}.", ""]
    if len(mem["runs"]) > 1:
        prev = mem["runs"][-2]
        lines.append(f"Previous run: {prev['errors']} errors, {prev['warnings']} warnings, {prev['notes']} notes.")
        lines.append("")
    if fixed:
        lines += ["## Fixed automatically", ""] + [f"- {x}" for x in fixed] + [""]
    ranked = sorted([(priority(f[1], r["runs_seen"]), f, r) for f, r in new + persistent], key=lambda t: -t[0])
    lines += ["## Fix first (severity times persistence)", ""]
    for sc, f, r in ranked[:15]:
        age = "new" if r["runs_seen"] == 1 else f"seen in {r['runs_seen']} runs"
        lines.append(f"- **{f[1]}** · {f[0]} · `{f[2]}` · {f[3]} ({age})")
    if not ranked:
        lines.append("Nothing to fix.")
    by = collections.defaultdict(list)
    for f in findings:
        by[f[0]].append(f)
    lines += ["", "## All findings by check", ""]
    for chk, fs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"### {chk} ({len(fs)})")
        for f in fs[:25]:
            lines.append(f"- {f[1]} · `{f[2]}` · {f[3]}")
        if len(fs) > 25:
            lines.append(f"- … and {len(fs) - 25} more")
        lines.append("")
    if resolved:
        lines += ["## Resolved", ""] + [f"- {r['check']} · `{r['page']}` · {r['message']}" for r in resolved[:20]] + [""]
    lines += ["## How the steward decides", "",
              "Hard checks apply everywhere. Convention findings come from the site itself: when 90% or more of the pages in a folder have a feature, the rest are flagged. "
              "Priority is the severity weight (error 3, warning 2, note 1) times 1 plus the log of the number of runs the issue has survived. "
              "The memory file `tools/steward/memory.json` holds the history."]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


def fix_og(site):
    """Repair og:image URLs whose file is missing when the other naming form exists (flat or folder)."""
    fixed = []
    for p, s in site.soup.items():
        og = s.find("meta", property="og:image")
        if not og or not og.get("content", "").startswith(SITE):
            continue
        rel = og["content"][len(SITE):].split("?")[0]
        if (ROOT / rel).exists():
            continue
        m = re.match(r"assets/og/([^/]+)/([^/]+)\.jpg$", rel)
        alts = []
        if m:
            alts.append(f"assets/og/{m.group(1)}-{m.group(2)}.jpg")
        m2 = re.match(r"assets/og/([^/]+)-([^/]+)\.jpg$", rel)
        if m2:
            alts.append(f"assets/og/{m2.group(1)}/{m2.group(2)}.jpg")
        for a in alts:
            if (ROOT / a).exists():
                txt = p.read_text(encoding="utf-8")
                p.write_text(txt.replace(rel, a), encoding="utf-8")
                fixed.append(f"{site.rel(p)}: og:image {rel} -> {a}")
                break
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="apply the safe fixes")
    ap.add_argument("--strict", action="store_true", help="exit 1 when there are errors")
    a = ap.parse_args()
    today = datetime.date.today().isoformat()
    fixed = []
    if a.fix:
        site = Site()
        fixed += fix_og(site)
        for name, cmd in (("author's note", ["python3", "tools/author_note.py"]), ("sitemap", ["python3", "tools/make_sitemap.py"]),
                          ("ask-box script", ["python3", "tools/site_ai/inject.py"]), ("ask-box index", ["python3", "tools/site_ai/build_index.py"])):
            if (ROOT / cmd[1]).exists():
                before = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
                ok, last = run(cmd)
                after = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
                if ok and before != after:
                    fixed.append(f"reran the {name} generator: {last[0]}")
    site = Site()
    findings = audit(site)
    mem, new, persistent, resolved, run_no = learn(findings, today)
    report(findings, mem, new, persistent, resolved, run_no, today, fixed)
    c = collections.Counter(f[1] for f in findings)
    print(f"steward run {run_no}: {c['error']} errors, {c['warning']} warnings, {c['note']} notes; {len(new)} new, {len(resolved)} resolved, {len(fixed)} fixed")
    sys.exit(1 if a.strict and c["error"] else 0)


if __name__ == "__main__":
    main()
