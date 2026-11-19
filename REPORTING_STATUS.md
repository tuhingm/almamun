# Reporting layer: status

Branch `reporting-layer`. Phase 1 is done; phases 2 to 5 have not been started.

## Architecture (in `oes/engine/oes_engine/`)
| Module | Role |
|---|---|
| `rep_model.py` | Report > Step > Table / Figure / Paragraph / TestResult (dataclasses). `Report.finalize()` numbers items and draws figures. |
| `rep_tables.py` | Table builders (regression, descriptives, correlation, tests) and renderers: plain text, HTML for Word (inline styles), LaTeX booktabs, CSV. |
| `rep_fig.py` | AER-style figures from a declarative spec; pure Python (no matplotlib in Pyodide); one scene gives SVG and vector PDF. Okabe-Ito palette plus dash patterns (readable in grayscale). |
| `rep_text.py` | Text engine. Every paragraph records the facts it used; `check_numbers` proves each printed number comes from them. |
| `rep_export.py` | JSON for the page, full HTML report, LaTeX document, file bundle. |
| `rep_ols.py`, `rep_anova1.py` | The two Phase 1 reporters (registry: `REPORTERS` in `__init__.py`). |
| `oes/report-ui.js` | Page component: TOC, toggleable steps, copy buttons, PNG 300 dpi, SVG, PDF, ZIP, Word `.doc`. |

`run()` adds `out["report"]` after the normal result. The old output is unchanged (`test_matches_engine`); `spec["report"]=False` switches it off; a reporter failure returns `report_error` and leaves the result intact.

## Done (Phase 1)
- OLS: 9 steps, 13 figures, 8 tables (data, correlations, estimation with 4 covariance estimators and nested specifications, effects, VIF, 9 diagnostics, CUSUM, influence, added-variable plots), methods, limitations, references.
- One-way ANOVA (A1): 4 steps, 8 figures, 7 tables.
- Tests: 28 in `stat-validation/tests/test_reporting.py`; full suite 205 passed. Headless Chrome ran `report-ui.js` on a real OLS report (render, CRC32 check value, PNG with pHYs chunk, ZIP).

## Dedicated reports so far
OLS, one-way ANOVA, VAR (9 steps, 10 figures). All other models: generic report (summary, tables, tests, charts, coefficient plots).

## Not done
- Phases 2 to 5 (the other 98 models): not started. They still show only the old output.
- Progressive display of steps for slow models: not implemented.
- Dataset provenance and licence per example dataset: not yet recorded.
- `xgb` cannot run in the local environment (no xgboost), so its example was not re-run here.

## Not tested in a real browser with Pyodide
Pyodide could not be loaded here. Tested only with headless Chrome and pre-generated JSON: not tested are Pyodide loading the new modules (worker.js file list is updated), pasting into real Word/Excel, opening the `.doc` in Word, PNG export in Safari/Firefox.

## Proposed steps for the next models
- VAR: data and stationarity, lag selection (IC table + figure), estimation, stability (roots), residual tests (autocorrelation, normality, ARCH), Granger causality, IRF with bootstrap bands, FEVD, historical decomposition (marked not validated if no reference).
- Panel FE: data and balance, pooled/FE/RE side by side, Hausman, F test for effects, clustered SE, serial correlation (Wooldridge), cross-sectional dependence (Pesaran), effect sizes, residual plots.
- Cox: data and censoring, Kaplan-Meier by group, log-rank, Cox table (hazard ratios), proportional-hazards test (Schoenfeld), martingale residuals, baseline survival, concordance.
- PCA: data and KMO/Bartlett, scree and parallel analysis, loadings, scores, biplot, explained variance table, reconstruction error.

## Flag
The page footer and the Python snippet still link to a GitHub repository. This predates my work and I have not changed it; your rule says not to link to GitHub from the website.

## Update
- Dedicated reports: OLS, ANOVA (one-way), VAR, PCA. All other models use the generic report (engine's own numbers, no new statistic).
- Fixes since PR #62: single table/figure button handler; in-page test-suite report check (PR #63).
- Still to do: dedicated reports for the other 96 models; dataset provenance/licence record; decision on Methods/limitations text; GitHub links in footer and Python snippet (pre-existing, flagged).
