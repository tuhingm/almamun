#!/usr/bin/env python3
"""Add the "Ask this site" script to every page (idempotent). Run after the page builders, then build_index.py."""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_index import pages  # noqa: E402

MARK = 'data-site-ask'


def main():
    n = 0
    for p, rel in pages():
        s = p.read_text(encoding="utf-8")
        if MARK in s or "http-equiv" in s.lower() and "refresh" in s.lower() or "</body>" not in s:
            continue
        up = "../" * rel.count("/")
        tag = f'<script defer src="{up}assets/ai/ask.js" {MARK}></script>\n'
        p.write_text(s.replace("</body>", tag + "</body>", 1), encoding="utf-8")
        n += 1
    print(f"ask box added to {n} pages")


if __name__ == "__main__":
    main()
