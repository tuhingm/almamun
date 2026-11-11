# Lecture 14 — Statistical Modeling Workflow, Simulation, and Reproducibility

## Learning objectives
Specify estimands and statistical models; design simulation studies; evaluate estimator performance; document computational decisions; report uncertainty and create reproducible archives.

## 1. Analysis plan
State the observational unit, target population, outcome, predictors, sampling process, estimand, model, identifying assumptions, missing-data strategy, and inferential target. Distinguish descriptive, predictive, and causal objectives. Select models for the target question rather than the largest in-sample fit statistic.

## 2. Data provenance
Preserve immutable raw data; document transformations, units, coding, exclusions, merges, duplicates, and missingness. Separate data cleaning from analysis. Record software versions, dependencies, random seeds, and computational environment.

## 3. Simulation design
Specify a data-generating process, parameter values, sample sizes, dependence, estimator, and performance measures. Across \(R\) replications,
\[
Bias=R^{-1}\sum_r(\widehat\theta_r-\theta_0),\quad
RMSE=\sqrt{R^{-1}\sum_r(\widehat\theta_r-\theta_0)^2}.
\]
Also evaluate empirical standard deviation, average estimated standard error, interval coverage, rejection rates, convergence, and failure rates.

## 4. Monte Carlo uncertainty
For estimated rejection frequency \(\widehat p\),
\[
MCSE(\widehat p)=\sqrt{\widehat p(1-\widehat p)/R}.
\]
Simulation estimates are themselves random. Choose \(R\) so Monte Carlo error is small relative to the differences under study.

## 5. Reproducible workflow
Use scripts or notebooks that execute from raw inputs to final tables and figures. Use version control, relative paths, structured project folders, automated checks, and execution logs. A random seed does not replace documentation of the DGP, software, and transformations.

## 6. Reporting and robustness
Report estimand, estimator, sample definition, assumptions, standard errors, confidence intervals, and diagnostics. Separate pre-specified from exploratory analysis. Robustness checks should vary assumptions or design choices that materially affect the target estimate, not search mechanically for significance.

## Worked example
Compare OLS and a robust estimator under heteroskedastic errors. Specify coefficients, sample sizes, and the error variance function; repeat estimation; calculate bias, RMSE, coverage, and rejection rates. A robust covariance estimator may change inference without changing the OLS point estimate.

## Problem set
1. Define a DGP for a heteroskedastic linear model.
2. Derive the Monte Carlo standard error of a rejection rate.
3. Distinguish estimator bias from Monte Carlo error.
4. Design a simulation comparing estimators under dependence.
5. List files and metadata required for independent replication.

## Computational lab
Build a configurable simulation script. Save replication-level estimates, summarize bias/RMSE/coverage/size, and create an archive with README and dependency record.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 14.1 — A reproducible analysis function with data validation

*Data:* `wage1` and `wage2`.

Good statistical code fails loudly on bad data and gives identical results on every run.

1. Write a function `analyze(df)` that (a) validates the input — required columns present, no missing values, wages positive, education between 0 and 20 — and raises an informative error otherwise; (b) fits `log(wage) ~ educ + exper + expersq` with HC1 standard errors; and (c) returns a tidy table.
2. Run it on `wage1`. Then run it on `wage2` (after constructing `expersq`) and compare.
3. Deliberately corrupt one value and show that validation catches it.

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
REQUIRED = ["wage", "educ", "exper"]
def analyze(df):
    # TODO: assertions, then fit and return a DataFrame of estimates
    ...
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
REQUIRED = ["wage", "educ", "exper"]
def validate(df):
    missing = [c for c in REQUIRED if c not in df]
    if missing: raise ValueError(f"missing columns: {missing}")
    if df[REQUIRED].isna().any().any(): raise ValueError("missing values in required columns")
    if (df["wage"] <= 0).any(): raise ValueError(f"{(df['wage'] <= 0).sum()} non-positive wages")
    if not df["educ"].between(0, 20).all(): raise ValueError("education outside [0, 20]")
def analyze(df, name):
    validate(df)
    d = df.assign(expersq=df["exper"]**2)
    r = smf.ols("np.log(wage) ~ educ + exper + expersq", data=d).fit(cov_type="HC1")
    return pd.DataFrame({"data": name, "n": int(r.nobs), "coef": r.params, "robust se": r.bse}).drop("Intercept")
print(pd.concat([analyze(wd.data("wage1"), "wage1"), analyze(wd.data("wage2"), "wage2")]).round(4).to_string())
bad = wd.data("wage1").copy(); bad.loc[10, "wage"] = -3.0
try:
    analyze(bad, "corrupted")
except ValueError as e:
    print("\nvalidation caught a problem:", e)
```

Output:

```text
data    n    coef  robust se
educ     wage1  526  0.0904     0.0078
exper    wage1  526  0.0410     0.0050
expersq  wage1  526 -0.0007     0.0001
educ     wage2  935  0.0780     0.0068
exper    wage2  935  0.0163     0.0140
expersq  wage2  935  0.0002     0.0006

validation caught a problem: 1 non-positive wages
```

The function produces the same tidy table on both datasets, and it stops with an informative message when a single wage is
corrupted, instead of silently producing a missing log or a wrong estimate. The return to education is 9.0% in the 1976
CPS and 7.8% in the 1980 NLS men; the experience profile is much flatter in `wage2`, which covers men of a narrow age range.
Validation checks are cheap, and they turn data errors into loud failures rather than wrong results.

</details>

### Exercise 14.2 — A simulation study with a fixed seed: which interval covers the mean of skewed data?

*Data:* `ceosal1` CEO salaries, treated as a population.

Design a small, reproducible simulation study.

1. Using `numpy.random.SeedSequence`, create independent random streams for each sample size \(n\in\{10,30,100\}\).
2. For each \(n\), draw 2,000 samples and compute three 95% intervals for the population mean: the \(t\)-interval, the bootstrap percentile interval and the bootstrap-\(t\) interval (199 resamples each).
3. Report coverage with Monte Carlo standard errors, and the average interval width. Run the script twice to confirm identical output.

```python
import numpy as np, wooldridge as wd
pop = wd.data("ceosal1")["salary"].to_numpy(float); mu = pop.mean()
streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2026).spawn(3)]
# TODO: loop over n and stream; compute the three intervals; tabulate coverage and width
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
pop = wd.data("ceosal1")["salary"].to_numpy(float); mu = pop.mean()
streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2026).spawn(3)]
rows = []
for n, rng in zip((10, 30, 100), streams):
    hit = np.zeros(3); width = np.zeros(3); R, B = 2000, 199
    for _ in range(R):
        x = rng.choice(pop, n); m, s = x.mean(), x.std(ddof=1); se = s / np.sqrt(n)
        q = stats.t.ppf(0.975, n - 1); ints = [(m - q * se, m + q * se)]
        bs = x[rng.integers(0, n, (B, n))]; bm = bs.mean(1); bse = bs.std(1, ddof=1) / np.sqrt(n)
        ints.append(tuple(np.percentile(bm, [2.5, 97.5])))
        tstar = (bm - m) / bse; lo, hi = np.percentile(tstar, [97.5, 2.5])
        ints.append((m - lo * se, m - hi * se))
        for j, (a, b) in enumerate(ints):
            hit[j] += a <= mu <= b; width[j] += b - a
    for j, lab in enumerate(("t-interval", "percentile", "bootstrap-t")):
        p = hit[j] / R
        rows.append((n, lab, round(p, 3), round(np.sqrt(p * (1 - p) / R), 3), round(width[j] / R, 1)))
print(pd.DataFrame(rows, columns=["n", "interval", "coverage", "MC se", "avg width"]).to_string(index=False))
```

Output:

```text
n    interval  coverage  MC se  avg width
 10  t-interval     0.821  0.009     1322.2
 10  percentile     0.748  0.010     1007.1
 10 bootstrap-t     0.856  0.008     3161.2
 30  t-interval     0.781  0.009      826.4
 30  percentile     0.769  0.009      737.4
 30 bootstrap-t     0.811  0.009     1451.3
100  t-interval     0.839  0.008      503.7
100  percentile     0.836  0.008      477.5
100 bootstrap-t     0.855  0.008      707.2
```

Because each sample size has its own seeded stream, the table is reproduced exactly on every run, and adding a new design
would not change the existing results. With such a skewed population, **all** three intervals undercover. The percentile
bootstrap is worst at small \(n\), and the bootstrap-\(t\) is best, at the cost of much wider intervals. Even at
\(n=100\), none reaches 90%. The Monte Carlo standard errors (about 0.01) show that differences of 3–4 points are real,
not simulation noise.

</details>

## Reading
Gentle, *Elements of Computational Statistics*; Boos and Stefanski, *Essential Statistical Inference*; Peng, “Reproducible Research in Computational Science”; National Academies, *Reproducibility and Replicability in Science*.