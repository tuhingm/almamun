# Site steward

A self-contained maintenance agent for this site. It uses no external AI and no network access, only deterministic checks and a small memory.

**What it does.** It checks links, anchors and assets, titles, descriptions, headings, alt text, the sitemap, Open Graph images and the OES model count; it learns which features most pages in a folder share (author's note, canonical link, share bar, and so on) and flags the pages that lack one; it remembers every finding in `memory.json`, so the report separates new, persistent and resolved problems and ranks them by severity times persistence.

**What it fixes.** With `--fix` it reruns the site's own generators (author's note, sitemap) and repairs Open Graph paths. Everything else is reported, never changed.

**How it runs.** `python3 tools/steward/steward.py [--fix]` writes `REPORT.md`. The workflow `.github/workflows/steward.yml` runs it every Sunday and opens a pull request with the fixes and the report, so nothing goes live until you merge it. Enable "Allow GitHub Actions to create and approve pull requests" in the repository settings once.

**What it cannot do.** It cannot write new pages, models or text. New content still needs you or an assistant like Claude. It can only audit, measure, remember and repair.
