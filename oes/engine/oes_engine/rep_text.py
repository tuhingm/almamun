"""Results-text engine: sentences filled from the numbers of the run, never typed by hand.

Every paragraph keeps the facts it used (Paragraph.facts) so an automated test can check that each number printed in the
text appears among those facts (check_numbers). Wording is deliberately cautious: associations, not causes, and every failed
assumption test is stated plainly (see the `problem` flag of rep_model.TestResult).
"""
import re

from .rep_model import Paragraph
from .rep_tables import MINUS, fp_text

ALLOWED = {"0.01", "0.05", "0.10", "5", "10", "95", "1", "2", "3", "4", "100", "0.1", "0.5"}


def _fix_minus(s):
    return re.sub(r"(?<![\w.])-(?=\d)", MINUS, s)


class Fmt(str):
    """A formatted number that remembers the value it was formatted from, so the facts of a paragraph stay numeric."""
    value = None


def pv(p):
    s = Fmt(fp_text(p))
    s.value = None if p is None else float(p)
    return s


def P(kind, template, **facts):
    """Paragraph from a template whose slots are filled from `facts` (Python format specs, e.g. {b:.3f}).

    The facts kept with the paragraph are the numbers and labels that went into it; a formatted p-value (pv) contributes its number.
    """
    keep = {}
    for k, v in facts.items():
        if isinstance(v, Fmt):
            if v.value is not None:
                keep[k] = v.value
        elif isinstance(v, (int, float, str)) and not isinstance(v, bool):
            keep[k] = v
    return Paragraph(kind, _fix_minus(template.format(**facts)), keep)


def join(items, conj="and"):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def direction(b):
    return "positive" if b > 0 else "negative"


def sig_phrase(p, level=0.05):
    if p is None:
        return "could not be tested"
    return "is statistically significant" if p < level else "is not statistically significant"


def number_tokens(text):
    return re.findall(r"\d+\.\d+|\d+", text.replace(MINUS, "-"))


def check_numbers(par, tol_decimals=(0, 1, 2, 3, 4, 5, 6)):
    """Return the numbers in a paragraph that match no recorded fact at the printed precision."""
    pool, labels = [], set()
    for v in par.facts.values():
        if isinstance(v, (int, float)):
            pool.append(abs(float(v)))
        elif isinstance(v, str):
            labels.update(number_tokens(v))      # numbers inside labels that came from the data (time periods, group names)
    missing = []
    for tok in number_tokens(par.text):
        if tok in ALLOWED or tok in labels:
            continue
        d = len(tok.split(".")[1]) if "." in tok else 0
        x = float(tok)
        ok = any(round(v, d) == x or (tok.startswith("0.000") and v < 0.001) for v in pool)
        # p < 0.001 is printed as a bound
        if not ok and tok in ("0.001", "0.0001") and any(v < float(tok) for v in pool):
            ok = True
        if not ok:
            missing.append(tok)
    return missing
