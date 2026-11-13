"""Insert the author's note into the hero of every guide, lecture, course and OES page.

Idempotent: pages that already carry the note are skipped. Run it after any page builder
(build_courses.py, the guide builders) as part of the site-wide finishing pass.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SIGN = '<span class="an-sign">— Tuhin G. M. Al Mamun, Ph.D.</span>'
TEXT = {
    "empirical": ("The decision path, the choice of tests and diagnostics, and the inference design in this guide are my own. "
                  "I built them from my training and research in econometrics, and I implemented and checked every step myself in Python. "
                  "The estimators are standard methods from the literature, cited where they are used."),
    "dsge": ("The model setup, the solution steps and the estimation design in this guide are my own. "
             "I built them from my training and research in macroeconomics, and I implemented and checked every step myself. "
             "The models are standard ones from the literature, cited where they are used."),
    "oes": ("The model menu, the default specifications, and the tests and inference settings in Open Econometric Systems are my own design. "
            "I built them from my training and research in econometrics, and I implemented and tested every estimator myself. "
            "The estimators are standard methods from the literature, documented in each model's guide."),
    "lecture": ("The structure of this lecture, the decision path and the worked examples are my own. "
                "I prepared them from my training and research in econometrics, and I implemented and checked every example myself. "
                "The methods are standard ones from the literature, cited where they are used."),
    "course": ("The design of this course, the lecture notes, the worked examples and the problem sets are my own. "
               "I prepared them from my training, research and teaching in economics. "
               "The theory and methods are standard, and the sources are listed in each lecture's readings."),
}
GUIDES = ["time-series-guide", "panel-data-guide", "causal-inference-guide", "ml-guide",
          "bayesian-guide", "markov-guide", "dsge-guide"]


def kind(rel: str) -> str:
    if rel.startswith("oes/"):
        return "oes"
    if rel.startswith("teaching/"):
        return "course"
    if "/lectures/" in rel:
        return "lecture"
    return "dsge" if rel.startswith("dsge-guide/") else "empirical"


def targets():
    for g in GUIDES:
        yield from sorted((ROOT / g).rglob("*.html"))
    yield ROOT / "oes" / "index.html"
    yield from sorted((ROOT / "teaching").rglob("*.html"))


def main():
    done = skipped = 0
    for p in targets():
        rel = p.relative_to(ROOT).as_posix()
        if "/examples/" in rel or "/img/" in rel:
            continue
        s = p.read_text(encoding="utf-8")
        if 'class="author-note"' in s:
            skipped += 1
            continue
        h = s.find('<section class="hero')
        end = s.find('</div></section>', h)
        if h < 0 or end < 0:
            print("no hero, skipped:", rel)
            continue
        note = (f'<aside class="author-note" aria-label="Author\'s note"><span class="an-label">Author\'s note</span>'
                f'<p>{TEXT[kind(rel)]}</p>{SIGN}</aside>')
        s = s[:end] + note + s[end:]
        m = re.search(r'<link rel="stylesheet" href="((?:\.\./)*)assets/css/[^"]+\.css">', s)
        prefix = m.group(1) if m else "../" * rel.count("/")
        s = s.replace("</head>", f'<link rel="stylesheet" href="{prefix}assets/css/author-note.css">\n</head>', 1)
        p.write_text(s, encoding="utf-8")
        done += 1
    print(f"{done} pages updated, {skipped} already had the note")


if __name__ == "__main__":
    main()
