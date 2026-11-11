#!/usr/bin/env python3
"""Run the MATLAB / Dynare coding exercises for a course and insert them, with their real output, into the lecture notes.

usage: python3 tools/exercises/run_matlab_exercises.py <course> [lecture-prefix ...] [--check]

tools/exercises/<course>.py defines INTRO (markdown) and EXERCISES = {lecture_file: [exercise, ...]}. Each exercise is a
dict with: title, data, task (markdown), starter and solution (each a list of (filename, code) pairs; the last file is the
script that is run), and notes (markdown).

Every solution script is executed in GNU Octave in a fresh folder that holds the course data files, the course helper
functions (teaching/<course>/code) and the exercise's own files. Dynare models (.mod files) are processed by the official
Dynare preprocessor; the first-order solution, impulse responses and theoretical moments are then computed by a small
test harness that mirrors Dynare's stoch_simul (set DYNLITE to its folder). Printed output is pasted under the solution.
"""
import os, pathlib, re, shutil, subprocess, sys, tempfile, textwrap, time
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[2]
course = sys.argv[1]
CHECK = "--check" in sys.argv
only = [a for a in sys.argv[2:] if not a.startswith("--")]
spec = importlib.util.spec_from_file_location("ex", ROOT / "tools" / "exercises" / f"{course}.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

OCTAVE = os.environ.get("OCTAVE", "/opt/oct/bin/octave-cli")
DYNLITE = os.environ.get("DYNLITE", "/home/claude/dynlite")
DATA = ROOT / "teaching" / course / "data"
CODE = ROOT / "teaching" / course / "code"


def run(files):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for src in list(DATA.glob("*.csv")) + list(CODE.glob("*.m")):
            shutil.copy(src, tmp / src.name)
        for name, code in files:
            (tmp / name).write_text(textwrap.dedent(code).strip() + "\n")
        script = files[-1][0]
        wrapper = (f"addpath('{DYNLITE}'); more off; format short; warning('off','all');\n"
                   f"try\n  run('{script}');\ncatch err\n  disp(['ERROR: ' err.message]); exit(1);\nend\n")
        (tmp / "_wrap.m").write_text(wrapper)
        env = dict(os.environ, OCTAVE_HOME=str(pathlib.Path(OCTAVE).parents[1]))
        p = subprocess.run([OCTAVE, "--no-gui", "--quiet", "--no-window-system", "_wrap.m"], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=1800)
        out = p.stdout.rstrip()
        out = "\n".join(l.rstrip() for l in out.splitlines()
                        if not l.startswith(("warning:", "Total computing time", "Note: ")))
        if p.returncode != 0 or "ERROR:" in out:
            raise RuntimeError(f"{script} failed:\n{out}\n{p.stderr[-3000:]}")
        return out


def block(lang, code):
    return f"```{lang}\n{textwrap.dedent(code).strip()}\n```"


def files_md(files):
    parts = []
    for name, code in files:
        parts += [f"*File `{name}`*", "", block("matlab", code), ""]
    return parts


SECTION_RE = re.compile(r"(?ms)^## Coding exercises\n.*?(?=^## |\Z)")
READING_RE = re.compile(r"(?m)^## (Reading|Further reading|Suggested reading|Readings)\b")

failed = 0
for lec, exs in M.EXERCISES.items():
    if only and not any(lec.startswith(o) for o in only):
        continue
    path = ROOT / "teaching" / course / "lectures" / lec
    parts = ["## Coding exercises", "", M.INTRO.strip(), ""]
    for k, ex in enumerate(exs, 1):
        t0 = time.time()
        try:
            out = run(ex["solution"])
        except Exception as e:  # report and continue
            print(f"!!! {lec} ex {k}: {e}\n"); failed += 1; continue
        num = re.match(r"(\d+)", lec).group(1).lstrip("0") or "0"
        print(f"--- {lec} ex {k} ({time.time()-t0:.1f}s)\n{out}\n")
        parts += [f"### Exercise {num}.{k} — {ex['title']}", "", f"*Data:* {ex['data']}", "",
                  textwrap.dedent(ex["task"]).strip(), ""]
        parts += files_md(ex["starter"])
        parts += ['<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>', ""]
        parts += files_md(ex["solution"])
        parts += ["Output:", "", block("text", out), "", textwrap.dedent(ex.get("notes", "")).strip(), "", "</details>", ""]
    section = "\n".join(parts).rstrip() + "\n\n"
    if CHECK or failed:
        continue
    text = path.read_text()
    if SECTION_RE.search(text):
        text = SECTION_RE.sub(lambda m: section, text)
    else:
        m = READING_RE.search(text)
        text = text[:m.start()] + section + text[m.start():] if m else text.rstrip() + "\n\n" + section
    path.write_text(text)
sys.exit(1 if failed else 0)
