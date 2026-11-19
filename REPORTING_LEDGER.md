# Reporting validation ledger

Each reported step names the automated test that checks it against an independent computation. A step with no such test is marked "not validated" in the report and here.

| Model | Step | Reference | Test | Status |
|---|---|---|---|---|
| ols | Descriptives | pandas/scipy recomputation | test_descriptives | validated |
| ols | Correlations | scipy.stats.pearsonr | test_correlations | validated |
| ols | Estimation (6 covariance types) | closed-form OLS, sandwich formulas in numpy | test_estimation, test_matches_engine | validated |
| ols | Effects | numpy closed forms | test_effects | validated |
| ols | VIF, condition number | statsmodels VIF, SVD | test_vif | validated |
| ols | Diagnostics (BG, BP, White, ARCH, JB, DW, LB, SW, RESET) | definitions in numpy/scipy | test_diagnostics | validated |
| ols | Stability | statsmodels recursive_olsresiduals; Brown-Durbin-Evans identity | test_stability | validated. CUSUM bounds use the standard 0.948 constant |
| ols | Influence | statsmodels OLSInfluence | test_influence | validated |
| ols | Added-variable plots | Frisch-Waugh-Lovell | test_avplots | validated |
| ols | Generated text | every number matches a recorded fact | test_ols_text_numbers | validated |
| stat_anova1 | all four steps | scipy, statsmodels, closed forms | test_anova_* | validated; Games-Howell and Hedges' g are checked against closed forms only (no third-party reference installed) |
| all | Figures, table exports, LaTeX compile | structure, pdflatex | test_exports, test_table_formats, test_figure_style | validated (format checks) |
| var | data, ADF, lag selection, estimation, stability, diagnostics, Granger, IRF point estimates, FEVD | direct OLS, companion matrix, definitions | test_reporting_var.py (10 tests) | validated |
| var | ADF p-value | statsmodels MacKinnon approximation | none independent | NOT VALIDATED (statistic is validated) |
| var | IRF 95% bands | statsmodels asymptotic standard errors | none independent | NOT VALIDATED, stated in the report |
| other 97 models | summary, result tables, charts, coefficient plots | engine output reproduced unchanged | test_generic_reports | formatting validated; no new statistic; no dedicated steps yet |

## PCA (stat_pca) dedicated report
5 steps (data, suitability, eigenvalues, loadings, scores), 10 figures. Tests: `stat-validation/tests/test_reporting_pca.py` (8).
- Eigenvalues, variance ratios: numpy `eigvalsh` and sklearn `PCA` — validated.
- Loadings, weights, scores: sklearn (up to sign) — validated.
- Bartlett statistic and p-value, KMO: from definitions with numpy/scipy — validated.
- Broken-stick: from its definition — validated.
- Parallel analysis (Monte-Carlo, seeded): **NOT VALIDATED** (no reference implementation); marked in the report.

## Static panel (pols, fe, re, fd, fedk) dedicated report
5 steps (data, estimation, tests, residuals, effects), 11 figures. Tests: `test_reporting_panel.py` (9).
- Pooled OLS: statsmodels OLS with cluster SEs. FE: LSDV (statsmodels dummies). Two-way FE: LSDV with unit and period dummies.
- FE/RE vs linearmodels 7.1: coefficients agree to 1e-5; clustered SEs agree to 5e-4 relative (linearmodels 7.x differs from the engine's small-sample factor in the 4th digit; the engine follows the guide's linearmodels version).
- F test of unit effects (statsmodels compare_f_test), Breusch-Pagan LM (definition), Jarque-Bera (scipy), unit effects (LSDV): validated.
- Hausman, Mundlak, Pesaran CD: computed by the engine's own helper functions; the Hausman and Mundlak formulas are NOT independently validated here. CD: NOT validated.
- fd column and first-differences figure inputs: from the engine's formulas, NOT independently validated.
