# Lecture 14 — Research Design, Monte Carlo, Replication, and Reporting

## Learning objectives
By the end of this lecture you should be able to:

1. move from a research question to a precisely stated estimand, an identification argument and an estimator;
2. design, run and report a Monte Carlo study, including Monte Carlo standard errors;
3. use the bootstrap correctly (pairs, wild, block, cluster), and know when it fails;
4. control for multiple testing (Holm, Benjamini–Hochberg, Romano–Wolf) and report specification robustness transparently;
5. build a replication package that meets current journal standards.

## Motivation
The previous thirteen lectures supplied estimators and the assumptions under which they work. This lecture is about using them in research that others can trust and reproduce. Two facts motivate it. First, the finite-sample behavior of an estimator in *your* setting is often unknown, and simulation is how you find out. Second, the credibility of empirical economics has been damaged less by bad estimators than by undisclosed researcher choices, unreported specifications and code that cannot be re-run. Journals now enforce reproducibility (the AEA Data Editor has done so since 2019), and evidence of selective reporting is widespread (Brodeur, Cook and Heyes, 2020).

## 1. From question to estimand
A good empirical paper answers four questions, in order.

1. **Estimand.** What population quantity do you want? "The effect of class size on test scores" is not yet an estimand. "The average effect, on third-grade reading scores, of reducing class size from 22 to 15 for students in Tennessee public schools in 1985–89" is one.
2. **Identification.** Under which assumptions does that quantity equal a feature of the observable distribution (a regression coefficient, a Wald ratio, a discontinuity)? State them in words and in notation, and say which can be partially tested.
3. **Estimation and inference.** Which estimator, which variance estimator (matching the dependence structure, Lecture 4), and which finite-sample corrections?
4. **Validation.** Placebo outcomes, placebo treatments, pre-trends, balance, density tests, sensitivity analysis.

**Pre-analysis plans.** For experiments, and increasingly for observational work using new data, register the primary outcomes, specifications and hypotheses before seeing the outcome data (the AEA RCT Registry, OSF). A plan does not forbid exploration. It makes the distinction between confirmatory and exploratory results visible (Olken, 2015).

## 2. Monte Carlo experiments
**Purpose.** Simulation answers questions that asymptotic theory leaves open: Is the bias of my estimator material at \(n=200\)? Do my 95% intervals cover? How many clusters do I need?

**Design checklist.**

- **Data-generating process (DGP).** Make it realistic: calibrate it to your data (sample size, number of clusters, regressor distributions, \(R^2\), error dependence). Vary the features that theory says matter (instrument strength, persistence, heteroskedasticity).
- **Estimators and procedures** compared on the same simulated datasets (common random numbers reduce the noise in comparisons).
- **Performance measures** for an estimator \(\widehat\theta_r\), \(r=1,\dots,R\):
\[
\text{bias}=\bar\theta-\theta,\quad \text{RMSE}=\sqrt{R^{-1}\textstyle\sum_r(\widehat\theta_r-\theta)^2},\quad \text{median bias},\quad \text{coverage}=R^{-1}\textstyle\sum_r1\{\theta\in CI_r\},\quad \text{size, power}.
\]
Report median bias and interquartile ranges when the estimator may lack moments (just-identified IV has no finite mean).
- **Monte Carlo standard errors.** Simulation results are estimates. For a rejection rate \(\widehat p\), the MC standard error is \(\sqrt{\widehat p(1-\widehat p)/R}\). With \(R=1{,}000\) and true size 5%, it is 0.0069, so an observed 6.2% is **not** evidence of size distortion. To detect a 1-point distortion reliably, you need \(R\approx10{,}000\).
- **Reproducibility.** Fix and report seeds, save simulated results, and parallelize with independent random streams (for example, `numpy.random.SeedSequence.spawn`).

## 3. The bootstrap
The bootstrap approximates the sampling distribution of a statistic by resampling from the data (Efron, 1979). **The resampling scheme must mimic the dependence structure of the data.**

| Data structure | Bootstrap |
|---|---|
| i.i.d. observations | pairs (resample rows \((Y_i,X_i)\)) |
| heteroskedastic regression | **wild**: \(Y_i^*=X_i'\widehat\beta+\widehat u_iv_i\), with \(v_i\) Rademacher (±1) |
| clustered | resample whole clusters, or the **wild cluster** bootstrap (one \(v_g\) per cluster), with the null imposed |
| time series | **block** (moving or stationary) bootstrap, or sieve (residual-based AR) |

**Asymptotic refinement.** Bootstrapping an **asymptotically pivotal** statistic, such as a \(t\)-statistic, gives more accurate tests and intervals than normal approximations (the percentile-\(t\) method). Bootstrapping the estimator itself (the percentile method) does not provide this refinement.

**When the bootstrap fails.** Examples: parameters on the boundary of the parameter space, the maximum of a sample, non-differentiable functionals, nearest-neighbor matching estimators (Abadie and Imbens, 2008), weak instruments, and unit-root autoregressions (\(\phi=1\)). Subsampling or the \(m\)-out-of-\(n\) bootstrap can help in some of these cases.

## 4. Multiple testing and specification robustness
**Multiple hypotheses.** Testing \(m\) hypotheses at 5% each gives a probability of at least one false rejection of up to \(1-0.95^m\): 40% for \(m=10\).

- **Family-wise error rate (FWER).** Bonferroni tests each hypothesis at \(\alpha/m\). **Holm's** step-down procedure is uniformly more powerful: sort p-values \(p_{(1)}\le\dots\le p_{(m)}\) and reject \(H_{(j)}\) while \(p_{(j)}\le\alpha/(m-j+1)\). **Romano–Wolf** (2005) uses the bootstrap to exploit the correlation between test statistics, which gives more power when outcomes are correlated.
- **False discovery rate (FDR).** **Benjamini–Hochberg** controls the *expected share* of false rejections among rejections: reject \(H_{(1)},\dots,H_{(k)}\) for the largest \(k\) with \(p_{(k)}\le k\alpha/m\). It is appropriate for exploratory analysis with many outcomes. Anderson (2008) applies these methods to early-childhood interventions.
- **Index outcomes.** Combine related outcomes into a standardized summary index (Kling, Liebman and Katz, 2007), which reduces the number of tests and increases power.

**Specification robustness.** Report how the main estimate changes across defensible choices of controls, samples, functional forms and estimators: a "specification curve" or multiverse analysis (Simonsohn, Simmons and Nelson, 2020). Robustness tables that only show specifications that work are not robustness checks.

## 5. Reporting standards
- Lead with **effect sizes and confidence intervals** in meaningful units (percent, standard deviations, dollars per person). Interpret magnitude against a benchmark.
- Avoid significance stars as the main message. "Not significant" is not "no effect": report the interval and what it rules out.
- Report the first-stage \(F\) (effective \(F\)) for IV, the number of clusters, the variance estimator, and the number of treated units.
- Show the raw data where possible: binned scatter plots, event-study plots with pre-periods, RD plots with binned means.
- Be explicit about the estimand. LATE, ATT and ATE differ, and so do weighted and unweighted estimates.

## 6. Replication packages
Modern journals require a replication package that reproduces every table and figure from raw data. The AEA Data and Code Availability Policy and the **Social Science Data Editors' template README** set the standard. A package should contain:

1. a **README** with data sources, access conditions (including for confidential data), the software and versions used, expected run time, and a table mapping every table and figure to the program that produces it;
2. **raw data** (or instructions to obtain them) kept separate from **derived data**, with the cleaning code;
3. a **master script** that runs everything in order with a single command;
4. a **fixed environment**: `requirements.txt` or `environment.yml` (Python), `renv.lock` (R), the Stata version plus ado-files;
5. **fixed seeds** for anything random (bootstrap, simulation, sample splits);
6. **relative paths only**, and no manual steps.

**Test it** by running the package from a clean machine or container before submission. Version control with Git, with tagged releases for submitted versions, makes the history auditable.

## Worked example
**How many replications?** You want to compare the size of a cluster-robust \(t\)-test and a wild cluster bootstrap with \(G=12\) clusters, where the true size of the first test might be about 9% and of the second about 5.5%. With \(R\) replications per design, the MC standard errors are \(\sqrt{0.09\times0.91/R}\) and \(\sqrt{0.055\times0.945/R}\). With \(R=2{,}000\), these are 0.0064 and 0.0051. Even treating the two estimates as independent, the MC standard error of their difference is \(\sqrt{0.0064^2+0.0051^2}=0.0082\), so a 3.5-point gap is about four standard errors from zero. Because both tests are run on the same simulated datasets, the true standard error of the difference is smaller still. With \(R=200\), the SEs are 0.020 and 0.016, and the comparison would be uninformative. With 399 bootstrap draws per replication, the total cost is \(2{,}000\times399\approx800{,}000\) model fits: plan computing time accordingly, and use a parallel, seeded design.

## Problem set
1. **MC error.** In 500 replications, a test rejects 33 times at nominal 5%. Is there evidence of over-rejection? Compute an approximate 95% interval for the true rejection rate.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\widehat p=0.066\), with MC SE \(\sqrt{0.066\times0.934/500}=0.0111\). The 95% interval is \(0.066\pm0.022=[0.044,\,0.088]\), which contains 0.05. That is not strong evidence; increase \(R\).
    </details>

2. **Holm versus BH.** With p-values \((0.001,\,0.008,\,0.012,\,0.030,\,0.040,\,0.300)\) and \(\alpha=0.05\), which hypotheses does Holm reject? Which does Benjamini–Hochberg reject?

    <details class="solution" markdown="1"><summary>Solution</summary>

    Holm thresholds are \(0.05/6,0.05/5,\dots\) \(=0.0083,0.010,0.0125,0.0167,0.025,0.05\). Compare: \(0.001\le0.0083\) ✓, \(0.008\le0.010\) ✓, \(0.012\le0.0125\) ✓, \(0.030>0.0167\) ✗ — stop. Holm rejects 3. BH thresholds \(k\alpha/6\): \(0.0083,0.0167,0.025,0.0333,0.0417,0.05\). The largest \(k\) with \(p_{(k)}\le k\alpha/m\) is \(k=5\) (\(0.040\le0.0417\)), so BH rejects 5.
    </details>

3. **Wrong bootstrap.** A researcher bootstraps standard errors in a state-year panel by resampling individual state-year observations. Explain the problem and the fix.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Resampling observations independently destroys within-state serial correlation, so the bootstrap reproduces the downward bias of non-clustered standard errors. Resample states (the cluster bootstrap), or use a wild cluster bootstrap with the null imposed when the number of states is small.
    </details>

4. **Estimand clarity.** Rewrite "We study the effect of microfinance on poverty" as a precise estimand for a study that randomizes branch openings across villages.

    <details class="solution" markdown="1"><summary>Solution</summary>

    "The intention-to-treat effect of opening a microfinance branch in a village, on the share of households below the national poverty line 24 months later, for villages in the study districts eligible for expansion." If take-up is partial, the IV estimand is the LATE for households that borrow because a branch opened.
    </details>

5. **README audit.** List five things that commonly prevent a replication package from running, and how to prevent each.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Absolute paths (use relative paths and a single root variable); missing packages or version drift (lock files, `ssc install` lines, `renv`); undocumented manual steps (script everything from a master file); unseeded randomness (set and report seeds); missing restricted data (document access precisely and provide simulated or synthetic data so the code runs end to end).
    </details>

## Computational lab
**Goal:** run a properly designed Monte Carlo with reported MC standard errors, and apply multiple-testing corrections.

```python
import numpy as np
from statsmodels.stats.multitest import multipletests

ss = np.random.SeedSequence(2024)
rngs = [np.random.default_rng(s) for s in ss.spawn(4)]   # independent streams, e.g. one per design

def coverage(rng, n, R=4000):
    hits = 0
    for _ in range(R):
        x = rng.lognormal(size=n)                        # skewed data: is the t-interval OK?
        m, s = x.mean(), x.std(ddof=1)
        hits += abs(m - np.exp(0.5)) <= 1.96 * s / np.sqrt(n)
    p = hits / R
    return p, np.sqrt(p * (1 - p) / R)

for rng, n in zip(rngs, (10, 30, 100, 1000)):
    p, se = coverage(rng, n)
    print(f"n={n:5d}: coverage {p:.3f} (MC s.e. {se:.3f})")

pvals = np.array([0.001, 0.008, 0.012, 0.030, 0.040, 0.300])
for method in ("bonferroni", "holm", "fdr_bh"):
    print(method, multipletests(pvals, alpha=0.05, method=method)[0])
```

Tasks: (i) interpret the coverage results for the mean of a lognormal and explain the role of skewness; (ii) add a bootstrap percentile-\(t\) interval and compare coverage at \(n=30\); (iii) write a template README and master script for your own term project, following the Social Science Data Editors' template.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 14.1 — Wild cluster bootstrap with few clusters

*Data:* `wagepan` — 545 workers, 1980–1987; the dummies `occ1`–`occ9` give each person-year's occupation (9 groups).

Many applied papers have few clusters. Build an `occupation` variable from the dummies `occ1`–`occ9` and use it (9 groups) as the cluster:

1. Regress `lwage` on `union`, `educ`, `exper`, `expersq` and year dummies; report the cluster-robust \(t\)-statistic on `union` with 9 occupation clusters.
2. Implement a **wild cluster bootstrap-t** with Rademacher weights and the null \(\beta_{union}=0\) imposed (999 draws, seed 1). Compare p-values.
3. Why can the conventional cluster-robust p-value be misleading here?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan")
df["occupation"] = df[[f"occ{k}" for k in range(1, 10)]].to_numpy().argmax(1) + 1
f = "lwage ~ union + educ + exper + expersq + C(year)"
# TODO: restricted model without union; y* = fitted_r + v_g * resid_r; refit unrestricted; cluster t each draw
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wagepan").reset_index(drop=True)
df["occupation"] = df[[f"occ{k}" for k in range(1, 10)]].to_numpy().argmax(1) + 1
f = "lwage ~ union + educ + exper + expersq + C(year)"
g = df["occupation"].to_numpy(); G = len(np.unique(g))
fit = lambda d: smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": True})
base = fit(df); t0 = base.tvalues["union"]
from scipy import stats
print(f"{G} clusters: union = {base.params['union']:.4f}, cluster t = {t0:.2f}, p (normal) = {2*stats.norm.sf(abs(t0)):.4f}, p (t_{G-1}) = {2*stats.t.sf(abs(t0), G-1):.4f}")

r = smf.ols(f.replace("union + ", ""), data=df).fit()
rng = np.random.default_rng(1); ts = []
groups = np.unique(g)
for _ in range(999):
    v = dict(zip(groups, rng.choice([-1.0, 1.0], size=G)))
    d = df.copy(); d["lwage"] = r.fittedvalues + r.resid * np.vectorize(v.get)(g)
    ts.append(fit(d).tvalues["union"])
p_wcb = np.mean(np.abs(ts) >= abs(t0))
print(f"wild cluster bootstrap-t p-value: {p_wcb:.3f}   (with G = {G}, Rademacher weights give only 2^{G} = {2**G} distinct draws)")
```

Output:

```text
9 clusters: union = 0.1768, cluster t = 5.37, p (normal) = 0.0000, p (t_8) = 0.0007
wild cluster bootstrap-t p-value: 0.012   (with G = 9, Rademacher weights give only 2^9 = 512 distinct draws)
```

With only 9 clusters, the cluster-robust variance estimator is noisy and biased downward, so normal (or even \(t_{G-1}\))
critical values overstate significance. The wild cluster bootstrap with the null imposed gives a more honest p-value —
here noticeably larger than the conventional one. With very few clusters, Rademacher weights allow only \(2^G\) distinct
bootstrap samples; Webb's six-point weights are recommended for \(G<12\). The deeper question, as in Lecture 4, is
whether occupation is the right clustering level for this design at all (Abadie et al., 2023).

</details>

### Exercise 14.2 — Multiple outcomes and multiple testing in the NSW experiment

*Data:* `jtrain2` — NSW randomized experiment.

1. Estimate the effect of training on six outcomes: `re78`, `unem78`, an indicator of positive 1978 earnings, earnings above \$5,000, above \$10,000, and the change `re78 − re75`.
2. Report unadjusted p-values and Holm, Bonferroni and Benjamini–Hochberg adjustments.
3. Build a standardized summary index (Kling, Liebman and Katz, 2007): standardize each outcome by the control-group mean and SD (signing `unem78` so higher is better), average them, and test the treatment effect on the index.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
d = wd.data("jtrain2")
d["pos78"] = (d.re78 > 0).astype(int); d["gt5"] = (d.re78 > 5).astype(int); d["gt10"] = (d.re78 > 10).astype(int); d["chg"] = d.re78 - d.re75
# TODO: loop over outcomes, collect robust p-values, apply multipletests
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
d = wd.data("jtrain2")
d["pos78"] = (d.re78 > 0).astype(int); d["gt5"] = (d.re78 > 5).astype(int)
d["gt10"] = (d.re78 > 10).astype(int); d["chg"] = d.re78 - d.re75
outs = ["re78", "unem78", "pos78", "gt5", "gt10", "chg"]
rows = []
for y in outs:
    r = smf.ols(f"{y} ~ train", data=d).fit(cov_type="HC1")
    rows.append((y, r.params["train"], r.pvalues["train"]))
tab = pd.DataFrame(rows, columns=["outcome", "effect", "p"]).set_index("outcome")
for meth, lab in (("bonferroni", "Bonferroni"), ("holm", "Holm"), ("fdr_bh", "BH (FDR)")):
    tab[lab] = multipletests(tab["p"], method=meth)[1]
print(tab.round(4).to_string())

c = d[d.train == 0]; z = []
for y in outs:
    s = (d[y] - c[y].mean()) / c[y].std()
    z.append(-s if y == "unem78" else s)
d["index"] = np.mean(z, axis=0)
ri = smf.ols("index ~ train", data=d).fit(cov_type="HC1")
print(f"\nsummary index: effect {ri.params['train']:.3f} control-SD units (robust se {ri.bse['train']:.3f}, p = {ri.pvalues['train']:.4f})")
```

Output:

```text
effect       p  Bonferroni    Holm  BH (FDR)
outcome                                              
re78     1.7943  0.0075      0.0449  0.0449    0.0216
unem78  -0.1106  0.0108      0.0648  0.0540    0.0216
pos78    0.1106  0.0108      0.0648  0.0540    0.0216
gt5      0.0771  0.1042      0.6252  0.2084    0.1158
gt10     0.0601  0.1158      0.6951  0.2084    0.1158
chg      1.5292  0.0324      0.1947  0.0973    0.0487

summary index: effect 0.227 control-SD units (robust se 0.085, p = 0.0074)
```

Four of the six outcomes are individually significant at 5%. After adjusting for six tests, only 1978 earnings survives
Bonferroni and Holm (Holm's adjusted p-values are smaller, but here they cross 5% at the same point), while the
Benjamini–Hochberg FDR procedure keeps four. Notice that `unem78` and `pos78` give identical results: they are almost the
same variable (positive earnings = employed), which is a reminder to define the outcome family carefully before testing. The summary index answers the
question "did training improve labor-market outcomes overall?" with a single test and more power than any one outcome.
Pre-specify which family of outcomes will be adjusted and which index will be primary *before* seeing the data
(Lecture 14, §1).

</details>

## Reading
**Required:** Christensen, Freese and Miguel, *Transparent and Reproducible Social Science Research* (2019), chapters 4–7 and 11; the Social Science Data Editors' template README and the AEA Data and Code Availability Policy.
**Further:** Cameron and Trivedi, *Microeconometrics* (2005), chapter 11 (bootstrap); Horowitz (2001), "The bootstrap", *Handbook of Econometrics* vol. 5; MacKinnon (2009), "Bootstrap hypothesis testing", in *Handbook of Computational Econometrics*; Romano and Wolf (2005), *Econometrica* 73(4); Benjamini and Hochberg (1995), *JRSS-B* 57(1); Anderson (2008), *JASA* 103(484); Olken (2015), "Promises and perils of pre-analysis plans", *Journal of Economic Perspectives* 29(3); Brodeur, Cook and Heyes (2020), "Methods matter: p-hacking and publication bias in causal analysis in economics", *AER* 110(11).
