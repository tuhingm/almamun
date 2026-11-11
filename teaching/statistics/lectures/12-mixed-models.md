# Lecture 12 — Linear Mixed Models and Hierarchical Data

## Learning objectives
Specify random-intercept and random-slope models; distinguish within- and between-group effects; derive covariance structures and ICC; understand partial pooling and likelihood-based estimation.

## 1. Hierarchical data
Observations may be nested within schools, hospitals, firms, regions, or repeated subjects. Shared group-level influences induce dependence. Row-level independence is generally implausible when observations share group-level determinants.

## 2. Random-intercept model
For unit \(i\) in group \(j\),
\[
y_{ij}=x_{ij}'\beta+b_j+\varepsilon_{ij},\quad
b_j\sim N(0,\tau^2),\quad\varepsilon_{ij}\sim N(0,\sigma^2).
\]
Assuming independence of random effects and residuals, marginal variance is \(\tau^2+\sigma^2\), and
\[
ICC=\frac{\tau^2}{\tau^2+\sigma^2}.
\]
Two observations in the same group have covariance \(\tau^2\) under this model.

## 3. Random slopes
Allow coefficients to vary:
\[
y_{ij}=\beta_0+b_{0j}+(\beta_1+b_{1j})x_{ij}+\varepsilon_{ij}.
\]
The random-effect vector has covariance matrix \(G\), which can allow intercept-slope correlation. The implied covariance depends on covariate values.

## 4. Within- and between-group effects
Decompose \(x_{ij}=(x_{ij}-\bar x_j)+\bar x_j\). Including both components separates within-group and between-group associations. A pooled coefficient can conflate the two.

## 5. Estimation and partial pooling
Mixed models are commonly estimated by maximum likelihood or REML. Empirical Bayes group predictions shrink estimates toward the population mean, with greater shrinkage for groups with less information. This is partial pooling, distinct from complete pooling and separate estimation.

## 6. Diagnostics
Inspect residuals, random-effect distributions, variance estimates, convergence, singular fits, and covariance-structure sensitivity. Variance-component tests can involve boundary conditions, so ordinary chi-square reference distributions may not apply directly. Report the number of groups as well as observations.

## Worked example
If between-school variance is 16 and within-school variance is 64, then \(ICC=16/(16+64)=0.20\). Under the random-intercept model, two students from the same school have correlation 0.20 conditional on fixed effects.

## Problem set
1. Derive marginal variance and within-group covariance.
2. Interpret the ICC.
3. Distinguish random intercepts from fixed group indicators.
4. Explain partial pooling.
5. Explain boundary issues in variance-component testing.

## Computational lab
Fit pooled OLS, group fixed-effects, and random-intercept models to nested data. Compare estimates, variance components, ICC, and group predictions; then fit a random-slope model.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 12.1 — Random intercepts and slopes for wage growth

*Data:* `wagepan` — 545 men, 1980–1987.

1. Fit a linear mixed model `lwage ~ exper + educ + black + hisp + union` with a random intercept for each worker (REML).
2. Add a random slope on `exper`. Report the variance components and the correlation between intercepts and slopes.
3. Compare the two models by AIC (refit with ML). Interpret the random-slope variance: how much do individual wage-growth rates differ?
4. Compare the fixed-effect estimate of `union` with the FE (within) estimator.

```python
import wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wagepan")
f = "lwage ~ exper + educ + black + hisp + union"
ri = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=True)
# TODO: rs = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit()
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wagepan")
f = "lwage ~ exper + educ + black + hisp + union"
ri = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=True)
rs = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit(reml=True, method="lbfgs")
icc = ri.cov_re.iloc[0, 0] / (ri.cov_re.iloc[0, 0] + ri.scale)
print(f"random intercept: sd(intercept) = {np.sqrt(ri.cov_re.iloc[0,0]):.3f}, residual sd = {np.sqrt(ri.scale):.3f}, ICC = {icc:.3f}")
V = rs.cov_re.to_numpy()
print(f"random slope:     sd(intercept) = {np.sqrt(V[0,0]):.3f}, sd(exper slope) = {np.sqrt(V[1,1]):.4f}, corr = {V[0,1]/np.sqrt(V[0,0]*V[1,1]):.3f}")
print(f"  mean exper slope {rs.fe_params['exper']:.4f}; middle 95% of individual slopes [{rs.fe_params['exper'] - 1.96*np.sqrt(V[1,1]):.4f}, {rs.fe_params['exper'] + 1.96*np.sqrt(V[1,1]):.4f}]")
a1 = smf.mixedlm(f, d, groups=d["nr"]).fit(reml=False).aic
a2 = smf.mixedlm(f, d, groups=d["nr"], re_formula="~exper").fit(reml=False, method="lbfgs").aic
print(f"AIC (ML): random intercept {a1:.1f}, random slope {a2:.1f}")
fe = smf.ols("lwage ~ exper + union + C(nr)", data=d).fit()
print(f"union: mixed model {ri.fe_params['union']:.4f}, within (FE) {fe.params['union']:.4f}")
```

Output:

```text
random intercept: sd(intercept) = 0.334, residual sd = 0.354, ICC = 0.470
random slope:     sd(intercept) = 0.452, sd(exper slope) = 0.0544, corr = -0.669
  mean exper slope 0.0631; middle 95% of individual slopes [-0.0436, 0.1698]
AIC (ML): random intercept 4470.1, random slope 4301.7
union: mixed model 0.1119, within (FE) 0.0856
```

Almost half of the variance is between workers (ICC 0.47). Allowing individual experience slopes improves the fit
substantially (AIC falls by about 170). The SD of individual slopes (0.054) is almost as large as the mean slope (0.063):
the middle 95% of workers have wage growth between about \(-4\%\) and \(+17\%\) per year of experience. The negative
intercept–slope correlation means workers who start low catch up. The mixed-model union effect (0.11) lies between pooled
OLS and FE (0.09), because the random-intercept model assumes worker effects uncorrelated with union status — the
assumption the Mundlak test rejected in Econometrics Lecture 7.

</details>

### Exercise 12.2 — Shrinkage: partial pooling of school means

*Data:* `meap93` — 408 Michigan high schools; `math10` = % of 10th graders passing a math test. We randomly divide the schools into 20 groups of very different sizes (4 to 78 schools) to mimic clusters with unequal information.

Treat each group as a "cluster" whose mean pass rate we want to estimate. Because groups are formed at random, every group's true mean is the same — so any spread in the raw group means is pure noise.

1. Compute the raw group means and their sizes.
2. Fit a random-intercept model `math10 ~ 1` with groups and compute the **empirical Bayes (BLUP)** group means.
3. Show that small groups are shrunk more toward the grand mean, and compute each group's shrinkage factor \(\tau^2/(\tau^2+\sigma^2/n_j)\).

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("meap93")
d = d.sample(frac=1, random_state=4).reset_index(drop=True)
sizes = np.r_[np.repeat(4, 8), np.repeat(10, 6), np.repeat(40, 4), np.repeat(78, 2)]
d["grp"] = np.repeat(np.arange(20), sizes)
# TODO: smf.mixedlm("math10 ~ 1", d, groups=d["grp"]).fit(); random_effects
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("meap93")
d = d.sample(frac=1, random_state=4).reset_index(drop=True)
sizes = np.r_[np.repeat(4, 8), np.repeat(10, 6), np.repeat(40, 4), np.repeat(78, 2)]   # unequal cluster sizes, total 408
d["grp"] = np.repeat(np.arange(20), sizes)
m = smf.mixedlm("math10 ~ 1", d, groups=d["grp"]).fit(reml=True)
tau2, s2, mu = m.cov_re.iloc[0, 0], m.scale, m.fe_params["Intercept"]
g = d.groupby("grp")["math10"].agg(n="size", raw="mean")
g["blup"] = mu + np.array([m.random_effects[k].iloc[0] for k in g.index])
g["shrink"] = tau2 / (tau2 + s2 / g["n"])
print(f"grand mean {mu:.2f}, between-group sd {np.sqrt(tau2):.2f}, within sd {np.sqrt(s2):.2f}")
print(g.iloc[[0, 1, 2, 8, 9, 14, 15, 18, 19]].round(3).to_string())
```

Output:

```text
grand mean 24.26, between-group sd 1.45, within sd 10.40
      n     raw    blup  shrink
grp                            
0     4  21.875  24.091   0.073
1     4  19.925  23.949   0.073
2     4  17.450  23.769   0.073
8    10  28.590  24.971   0.164
9    10  23.410  24.124   0.164
14   40  28.268  26.021   0.439
15   40  21.858  23.208   0.439
18   78  23.450  23.772   0.604
19   78  22.838  23.403   0.604
```

Groups were formed at random, so all true group means equal the grand mean; the spread of raw means is noise. The
between-group SD is estimated at only about 1.5 points against a within-group SD of 10.4. Small groups (4 schools) are
shrunk about 93% of the way to the grand mean — a raw mean of 17.5 becomes 23.8 — while the largest groups keep about 60%
of their deviation. Empirical Bayes estimates are biased toward the mean but have lower mean squared error; this is the
logic behind value-added models for teachers and schools.

</details>

## Reading
Pinheiro and Bates, *Mixed-Effects Models in S and S-PLUS*; Gelman and Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*; McCulloch et al., *Generalized, Linear, and Mixed Models*.