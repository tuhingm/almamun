# Implementation ledger

Status of the 31 Statistics Division methods. 'MC reps' is the total number of Monte Carlo replications actually run for the method; 'capped cells' counts (scenario, n) cells that stopped before the 2,000 requested replications because of the per-cell time budget.

| Code | Method | Implemented | Unit/integration tests | Benchmarked | Monte Carlo | MC reps | capped cells | Status |
|---|---|---|---|---|---|---|---|---|
| A1 | One-way ANOVA | yes | yes | yes | yes | 40000 | 0 | Validated |
| A2 | Two-way ANOVA | yes | yes | yes | yes | 39036 | 9 | Validated |
| A3 | Repeated-measures ANOVA | yes | yes | yes | NO | – | – | Validated |
| A4 | Analysis of covariance (ANCOVA) | yes | yes | yes | NO | – | – | Validated |
| A5 | Multivariate ANOVA (MANOVA) | yes | yes | yes | NO | – | – | Validated |
| A6 | Multivariate ANCOVA (MANCOVA) | yes | yes | yes | NO | – | – | Validated |
| B1 | Mann-Whitney U test | yes | yes | yes | yes | 40000 | 0 | Validated |
| B2 | Wilcoxon signed-rank test | yes | yes | yes | yes | 27792 | 3 | Validated |
| B3 | Kruskal-Wallis test | yes | yes | yes | NO | – | – | Validated |
| B4 | Friedman test | yes | yes | yes | NO | – | – | Validated |
| B5 | Kendall's tau correlation | yes | yes | yes | yes | 25951 | 4 | Validated |
| C1 | Principal component analysis (PCA) | yes | yes | yes | NO | – | – | Validated |
| C2 | Exploratory factor analysis (EFA) | yes | yes | yes | NO | – | – | Validated |
| C3 | Confirmatory factor analysis (CFA) | yes | yes | yes | yes | 16000 | 0 | Validated |
| C4 | Linear discriminant analysis (LDA) | yes | yes | yes | NO | – | – | Validated |
| C5 | Canonical correlation analysis (CCA) | yes | yes | yes | NO | – | – | Validated |
| C6 | Correspondence analysis (CA) | yes | yes | yes | NO | – | – | Validated |
| D1 | Kaplan-Meier estimator and log-rank test | yes | yes | yes | NO | – | – | Validated |
| D2 | Cox proportional hazards model | yes | yes | yes | NO | – | – | Validated |
| D3 | Accelerated failure time model | yes | yes | yes | NO | – | – | Validated |
| D4 | Fine-Gray competing-risks regression | yes | yes | yes | NO | – | – | Partially validated |
| D5 | Parametric survival (Weibull, exponential, log-normal) | yes | yes | yes | NO | – | – | Validated |
| E1 | Linear mixed-effects model (LMM) | yes | yes | yes | yes | 21551 | 18 | Validated |
| E2 | Generalised linear mixed model (GLMM) | yes | yes | yes | yes | 1071 | 8 | Partially validated |
| E3 | Generalised estimating equations (GEE) | yes | yes | yes | NO | – | – | Validated with a documented deviation |
| E4 | Intraclass correlation (ICC) and reliability | yes | yes | yes | NO | – | – | Validated |
| F1 | Bootstrap confidence intervals | yes | yes | yes | NO | – | – | Validated |
| F2 | Permutation tests | yes | yes | yes | NO | – | – | Validated |
| F3 | Power and sample-size analysis | yes | yes | yes | NO | – | – | Validated |
| G1 | Fixed-effect meta-analysis | yes | yes | yes | NO | – | – | Validated, small-study tests unchecked |
| G2 | Random-effects meta-analysis | yes | yes | yes | NO | – | – | Validated, small-study tests unchecked |

## Reasons behind non-plain statuses

* **D4 Fine-Gray competing-risks regression** (Partially validated): Point estimates validated (equals Cox when no competing events; simulation unbiased). Standard errors omit the censoring-weight estimation variance and no R cmprsk comparison was possible, so inference rests on simulation only.
* **E2 Generalised linear mixed model (GLMM)** (Partially validated): Matches an independent quadrature MLE and published lme4 values; no direct GLMM reference implementation was available in this environment.
* **E3 Generalised estimating equations (GEE)** (Validated with a documented deviation): Independence/exchangeable match statsmodels; AR(1) with estimated alpha uses a different moment estimator (see limitations); QIC follows Pan (2001) and differs from statsmodels' QIC.
* **G1 Fixed-effect meta-analysis** (Validated, small-study tests unchecked): Pooling and heterogeneity match statsmodels and metafor; Egger and Begg tests were not cross-checked.
* **G2 Random-effects meta-analysis** (Validated, small-study tests unchecked): Pooled estimate, tau2 and HKSJ interval match metafor published values and statsmodels; Egger and Begg tests were not cross-checked.

## Known gaps (applies to the whole division)

* All computation and simulation was run in CPython with the Pyodide-pinned package versions. The pages were exercised in a simulated browser (jsdom); they have not been run in a real browser with real Pyodide, so browser speed and memory are untested.
* The OES single-column CSV reader requires at least two columns; a one-column file is rejected (inherited behaviour of the existing OES core).
* Some reference comparisons were possible only against Python packages; R packages (lme4, geepack, cmprsk) were not available, so lme4 comparisons use published values.
* The BCG vaccine table was entered by hand from the published table and checked by reproducing metafor's published REML results.
* Several example datasets are distributed by R packages under GPL; check the licence before redistributing them.
