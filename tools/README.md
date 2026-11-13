# Course site builder

The course pages under `teaching/` are written in Markdown and rendered to HTML by
`build_courses.py`. The site is served as plain static files (`.nojekyll` at the repo root),
so **after editing any `.md` file in `teaching/`, rebuild and commit the generated `.html` files**:

```bash
pip install markdown pymdown-extensions jinja2 pyyaml
python3 tools/build_courses.py
```

## What lives where
| Path | Purpose |
|---|---|
| `tools/courses.yml` | course metadata (title, level, prerequisites, topics, lecture parts) and the section menu |
| `tools/templates/course.html` | page layout shared by every course page |
| `assets/css/course.css`, `assets/js/course.js` | styles; mobile menu and "On this page" contents |
| `teaching/<course>/index.md` | course home (`<!-- units -->` is replaced by the list of parts and lectures) |
| `teaching/<course>/syllabus.md`, `problem-sets-solutions.md`, `examinations.md`, `outline.md` | section pages (`outline.md` is optional) |
| `teaching/<course>/lectures/NN-*.md` | lecture notes; the first line must be `# Lecture NN — Title` |

`calendar.html`, `lecture-notes.html` and `readings.html` are generated: the calendar from the first
table in `syllabus.md`, the readings from each lecture's `## Reading` section.

## Writing lecture notes
- Math: inline `\( ... \)`, display `\[ ... \]` on their own lines, or `\begin{align} ... \end{align}`.
  Inside a table cell, write `\mid` instead of a bare `|`.
- Section headings named *Learning objectives*, *Worked example*, *Problem set*, *Computational lab*
  and *Reading* are styled as boxes automatically.
- Theorem-style blocks: `<div class="theorem" markdown="1"> ... </div>` (also `definition`, `proof`, `remark`).
- Collapsible solutions: `<details class="solution" markdown="1"><summary>Solution</summary> ... </details>`.

## Site-wide finishing pass (run after any builder)
Every page shares one top menu, link-preview tags and search metadata. After running any page builder, run:

```bash
python3 tools/author_note.py     # author's note on every guide, lecture, course and OES page
python3 tools/site_polish.py      # same menu on every page, missing preview tags, author/robots, JSON-LD + breadcrumbs
python3 tools/make_og_images.py   # preview images (needs Playwright; OG_ONLY=assets/og/dsge-guide renders a subset)
python3 tools/og_tags.py          # completes og:/twitter: tags; bump VERSION to refresh cached previews
python3 tools/make_sitemap.py
```

| Path | Purpose |
|---|---|
| `assets/css/site-header.css` | the one menu style for every page (desktop row; dropdown below 1040px) |
| `tools/site_polish.py` | menu markup, name variants for search (`ALT_NAMES`), structured data |
| `tools/og_cards_site.json` | preview cards for the panel, causal-inference and DSGE guide pages (written by `site_polish.py`) |

## Machine Learning Guide (`ml-guide/`)
```bash
pip install numpy pandas scipy statsmodels scikit-learn xgboost doubleml econml linearmodels wooldridge pyreadr
python3 tools/ml_selector/build_ml.py            # tool page + example data (add --rerun to recompute the cached checks)
python3 tools/ml_selector/build_ml_guides.py     # the nine method pages (or name some: lasso rf dml-plr ...)
```
| Path | Purpose |
|---|---|
| `tools/ml_selector/ml_analysis.py` | decision engine (rules A1–A5, B1–B5 and their cut-offs); also runs in the browser via Pyodide |
| `tools/ml_selector/template.html`, `app.js` | the interactive tool page |
| `tools/ml_selector/guides/*.md` | method pages; every python block is executed and its output inserted |
| `tools/ml_selector/data/` | downloaded source data and cached check results (not committed) |
