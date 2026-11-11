#!/usr/bin/env python3
"""Run the coding exercises for a course and insert them, with their real output, into the lecture notes.

usage: python3 tools/exercises/run_exercises.py <course> [--check]

tools/exercises/<course>.py defines INTRO (markdown) and EXERCISES = {lecture_file: [exercise, ...]},
each exercise a dict with: title, data, task (markdown), starter (code), solution (code), notes (markdown).
Every solution is executed; its printed output is pasted under it. A '## Coding exercises' section is
(re)written in teaching/<course>/lectures/<lecture_file>, just before the '## Reading' section.
--check runs everything and prints outputs without writing files.
"""
import contextlib, importlib.util, io, pathlib, re, sys, textwrap, time, warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
course = sys.argv[1]
CHECK = "--check" in sys.argv
only = [a for a in sys.argv[2:] if not a.startswith("--")]
spec = importlib.util.spec_from_file_location("ex", ROOT / "tools" / "exercises" / f"{course}.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def run(code):
    import numpy as np, pandas as pd
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 100); pd.set_option("display.precision", 4); pd.set_option("display.max_columns", 12)
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        exec(compile(code, "<exercise>", "exec"), {"__name__": "__main__"})
    import matplotlib.pyplot as plt; plt.close("all")
    return buf.getvalue().rstrip()


def block(lang, code):
    return f"```{lang}\n{textwrap.dedent(code).strip()}\n```"


SECTION_RE = re.compile(r"(?ms)^## Coding exercises\n.*?(?=^## |\Z)")
READING_RE = re.compile(r"(?m)^## (Reading|Further reading|Suggested reading|Readings)\b")

for lec, exs in M.EXERCISES.items():
    if only and not any(lec.startswith(o) for o in only):
        continue
    path = ROOT / "teaching" / course / "lectures" / lec
    parts = ["## Coding exercises", "", M.INTRO.strip(), ""]
    for k, ex in enumerate(exs, 1):
        t0 = time.time()
        out = run(ex["solution"])
        num = re.match(r"(\d+)", lec).group(1).lstrip("0") or "0"
        print(f"--- {lec} ex {k} ({time.time()-t0:.1f}s)\n{out}\n")
        parts += [f"### Exercise {num}.{k} — {ex['title']}", "",
                  f"*Data:* {ex['data']}", "", textwrap.dedent(ex["task"]).strip(), "",
                  block("python", ex["starter"]), "",
                  '<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>', "",
                  block("python", ex["solution"]), "", "Output:", "", block("text", out), "",
                  textwrap.dedent(ex.get("notes", "")).strip(), "", "</details>", ""]
    section = "\n".join(parts).rstrip() + "\n\n"
    if CHECK:
        continue
    text = path.read_text()
    text = SECTION_RE.sub("", text)
    m = READING_RE.search(text)
    text = text[:m.start()] + section + text[m.start():] if m else text.rstrip() + "\n\n" + section
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)}")
