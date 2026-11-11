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
