# Lecture 11 — Statistical Power, Sample Size, and Equivalence

## Learning objectives
Define power and minimum detectable effects; calculate sample sizes; account for clustering and attrition; distinguish non-significance from evidence of equivalence.

## 1. Power
For rejection region \(R\), power at parameter \(\theta\) is \(\pi(\theta)=P_\theta(T\in R)\). At a specified alternative it equals \(1-\beta\). Power depends on effect size, variance, sample size, significance level, design, and test. It is a design property under an assumed model, not a post hoc explanation for a non-significant finding.

## 2. Two independent means
For equal group sizes \(n\), common variance \(\sigma^2\), and difference \(\delta\), the standard error is approximately \(\sigma\sqrt{2/n}\). The standardized effect is \(d=\delta/\sigma\), and the normal-approximation noncentrality parameter is \(d\sqrt{n/2}\). Exact planning uses the noncentral t distribution and the planned sidedness.

## 3. Regression and proportions
Regression power depends on residual variance, the variance of the focal regressor after partialling out controls, and the design matrix. Collinearity reduces information even when total sample size is fixed. Proportion comparisons depend on null and alternative proportions and the selected test.

## 4. Clustered designs
For equal cluster size \(m\) and intraclass correlation \(\rho\), the approximate design effect is
\[
DE=1+(m-1)\rho.
\]
Effective sample size is approximately total sample size divided by this factor. Unequal cluster sizes and few clusters require more careful planning and small-sample corrections.

## 5. Attrition
If the required analyzable sample is \(n_a\) and expected attrition is \(r\), recruit approximately \(n_a/(1-r)\). This inflation does not remove bias from informative attrition; the missing-data mechanism must be considered separately.

## 6. Equivalence and non-inferiority
For equivalence margin \(\Delta>0\), test \(H_0:\theta\le-\Delta\text{ or }\theta\ge\Delta\) against \(H_1:-\Delta<\theta<\Delta\). The two-one-sided-tests procedure assesses whether the confidence interval lies inside the equivalence bounds. Failure to reject a zero-effect null is not evidence of equivalence.

## Worked example
If \(\sigma=10\) and the target difference is 5, then \(d=0.5\). Use the noncentral t distribution to find the group size achieving target power at the chosen alpha, then verify sensitivity to variance and attrition assumptions.

## Problem set
1. Define power as a function of the parameter.
2. Derive the two-group mean standard error.
3. Calculate a design effect for given cluster size and ICC.
4. Inflate an analyzable sample for attrition.
5. Formulate equivalence hypotheses for a chosen margin.

## Computational lab
Calculate sample sizes over a grid of effect sizes and variances. Plot power curves and compare individual versus cluster randomization.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 11.1 — Planning a job-training trial from pilot data

*Data:* `jtrain2` — use the NSW control group as pilot data on the outcome's variability.

1. Using the control-group SD of `re78`, compute the sample size per arm needed to detect an effect of \$1,000 with 80% power at \(\alpha=0.05\) (two-sided), analytically with `statsmodels`.
2. Compute the minimum detectable effect (MDE) for the actual NSW design (185 treated, 260 controls).
3. Simulate the power of the Welch \(t\)-test for a \$1,800 effect in the actual design by resampling control outcomes (2,000 replications). Compare with the analytic power. Why might they differ?

```python
import numpy as np, wooldridge as wd
from statsmodels.stats.power import TTestIndPower
c = wd.data("jtrain2").query("train == 0")["re78"].to_numpy(); sd = c.std(ddof=1)
pw = TTestIndPower()
# TODO: pw.solve_power(effect_size=1/sd, power=0.8, alpha=0.05); MDE with nobs1=185, ratio=260/185
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
c = wd.data("jtrain2").query("train == 0")["re78"].to_numpy(); sd = c.std(ddof=1)
pw = TTestIndPower()
n = pw.solve_power(effect_size=1.0 / sd, power=0.8, alpha=0.05)
print(f"control SD of re78 = {sd:.3f} thousand;  n per arm for a $1,000 effect at 80% power: {np.ceil(n):.0f}")
es = pw.solve_power(nobs1=185, ratio=260/185, power=0.8, alpha=0.05)
print(f"MDE for the NSW design (185 vs 260): {es*sd:.3f} thousand dollars ({es:.3f} SD)")
print(f"analytic power for a $1,800 effect: {pw.power(effect_size=1.8/sd, nobs1=185, ratio=260/185, alpha=0.05):.3f}")
rng = np.random.default_rng(9); rej = 0
for _ in range(2000):
    y0 = rng.choice(c, 260); y1 = rng.choice(c, 185) + 1.8
    rej += stats.ttest_ind(y1, y0, equal_var=False).pvalue < 0.05
print(f"simulated power for a $1,800 effect: {rej/2000:.3f} (MC se {np.sqrt(rej/2000*(1-rej/2000)/2000):.3f})")
```

Output:

```text
control SD of re78 = 5.484 thousand;  n per arm for a $1,000 effect at 80% power: 474
MDE for the NSW design (185 vs 260): 1.481 thousand dollars (0.270 SD)
analytic power for a $1,800 effect: 0.926
simulated power for a $1,800 effect: 0.929 (MC se 0.006)
```

Earnings are very noisy (SD of about \$5,500), so detecting a \$1,000 effect needs about 474 men per arm. The actual NSW
design could detect effects of about \$1,500 (0.27 SD) with 80% power, and had about 93% power for the \$1,800 effect it
found. The simulation reproduces the analytic power closely here. Simulated power is more trustworthy in general, because
it uses the real, skewed, zero-inflated outcome distribution instead of a normal approximation.

</details>

### Exercise 11.2 — Power under clustering: the design effect

*Data:* `wagepan` — 545 workers observed for 8 years; outcome `lwage`.

Suppose a wage subsidy is randomized **by worker** and wages are measured in all 8 years.

1. Estimate the intra-class correlation (ICC) of `lwage` within workers using a one-way random-effects ANOVA.
2. Compute the design effect \(1+(m-1)\rho\) with \(m=8\) and the effective sample size.
3. How many workers are needed to detect a 5% wage effect with 80% power, accounting for clustering? Compare with the naive calculation that treats person-years as independent.

```python
import numpy as np, wooldridge as wd
d = wd.data("wagepan")
g = d.groupby("nr")["lwage"]; m = 8; k = d["nr"].nunique()
# TODO: MSB = m * var(group means) ; MSW = pooled within variance; ICC = (MSB - MSW) / (MSB + (m-1) MSW)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from statsmodels.stats.power import TTestIndPower
d = wd.data("wagepan"); m = 8; k = d["nr"].nunique()
gm = d.groupby("nr")["lwage"].mean()
msb = m * ((gm - d["lwage"].mean())**2).sum() / (k - 1)
msw = ((d["lwage"] - d.groupby("nr")["lwage"].transform("mean"))**2).sum() / (k * (m - 1))
icc = (msb - msw) / (msb + (m - 1) * msw)
deff = 1 + (m - 1) * icc; sd = d["lwage"].std()
print(f"ICC of log wages within workers = {icc:.3f};  design effect = {deff:.2f};  effective n = {len(d)/deff:.0f} of {len(d)} person-years")
n_naive = TTestIndPower().solve_power(effect_size=0.05 / sd, power=0.8, alpha=0.05)
print(f"naive: {np.ceil(n_naive):.0f} person-years per arm = {np.ceil(n_naive/m):.0f} workers per arm")
print(f"with clustering: {np.ceil(n_naive*deff/m):.0f} workers per arm")
```

Output:

```text
ICC of log wages within workers = 0.472;  design effect = 4.30;  effective n = 1013 of 4360 person-years
naive: 1783 person-years per arm = 223 workers per arm
with clustering: 959 workers per arm
```

Almost half of the variance in log wages is between workers (ICC 0.47), so 8 observations per worker carry the information
of only about 2.3 independent ones: the design effect is 4.3. Ignoring clustering would suggest 223 workers per arm;
accounting for it requires about 959 per arm — more than four times as many. When treatment is assigned at the worker level,
repeated measurements of the same worker add much less information than new workers.

</details>

## Reading
Cohen, *Statistical Power Analysis for the Behavioral Sciences*; Chow et al., *Sample Size Calculations in Clinical Research*; Lakens, *Equivalence Tests*.