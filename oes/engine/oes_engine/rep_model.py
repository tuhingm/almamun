"""Report data model of the OES reporting layer.

    Report  ->  Step  ->  Table | Figure | Paragraph | TestResult

A Report is what a researcher needs to write up one analysis: ordered steps (data, tests, estimation, diagnostics,
effects ...), each with publication-style tables, AER-style figures, tests and a short interpretation written from the
numbers of the run. Nothing in a report is typed by hand: every number comes from the estimation and the paragraph
generators record the facts they used (Paragraph.facts) so a test can check that the text matches the tables.

The reporting layer only reads results; it never changes an estimate (the estimators in the other modules are untouched).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Table:
    key: str
    title: str
    header: List[List[str]]                    # one or more header rows (already formatted)
    rows: List[Dict[str, Any]]                 # {"cells": [str], "kind": body|section|stat|rule}
    notes: List[str] = field(default_factory=list)
    spans: Optional[List[Any]] = None          # column groups for the first header row: [(start, end, label)]
    align: Optional[List[str]] = None          # "l" | "r" | "c" per column
    number: int = 0                            # assigned by Report.finalize
    raw: Optional[Dict[str, Any]] = None       # unformatted numbers for CSV/JSON users

    def caption(self):
        return f"Table {self.number}. {self.title}"


@dataclass
class Figure:
    key: str
    title: str
    spec: Dict[str, Any]                       # declarative description, drawn by rep_fig
    note: str = ""
    source: str = ""
    width_in: float = 6.5
    height_in: float = 3.6
    number: int = 0
    svg: str = ""

    def caption(self):
        return f"Figure {self.number}. {self.title}"


@dataclass
class Paragraph:
    kind: str                                  # "result" | "warning" | "method" | "note"
    text: str
    facts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    name: str
    statistic: Optional[float]
    p_value: Optional[float]
    null: str
    df: Optional[str] = None
    reject: Optional[bool] = None              # at the 5 per cent level; None when no p-value exists
    problem: bool = False                      # True when rejection signals a violated assumption
    reference: str = ""


@dataclass
class Step:
    key: str
    title: str
    purpose: str = ""
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tests: List[TestResult] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    validated: str = "not validated"           # reference used in the automated test, or "not validated"
    refs: List[str] = field(default_factory=list)   # keys into Report.references


@dataclass
class Report:
    model_id: str
    title: str
    steps: List[Step] = field(default_factory=list)
    methods: List[Paragraph] = field(default_factory=list)
    limitations: List[Paragraph] = field(default_factory=list)
    references: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Number tables and figures in the order of the steps and draw the figures."""
        from . import rep_fig
        nt = nf = 0
        for s in self.steps:
            for t in s.tables:
                nt += 1
                t.number = nt
            for fg in s.figures:
                nf += 1
                fg.number = nf
                fg.svg = rep_fig.render(fg.spec, fg.width_in, fg.height_in).svg()
        return self

    def used_references(self):
        keys = []
        for s in self.steps:
            for k in s.refs:
                if k not in keys:
                    keys.append(k)
        for k in self.meta.get("extra_refs", []):
            if k not in keys:
                keys.append(k)
        keys = [k for k in keys if k in self.references]
        return [self.references[k] for k in sorted(keys, key=lambda k: self.references[k])]

    def unvalidated_steps(self):
        return [s.title for s in self.steps if s.validated == "not validated"]
