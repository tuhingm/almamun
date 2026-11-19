title: Distributional Effects: Quantile Treatment Effects and KS Tests
short: Quantile effects
when: You want to know who gains from a treatment, not only the average gain, and whether the whole outcome distribution shifted
data: National Supported Work demonstration (LaLonde 1986): 445 workers randomly assigned to job training; outcome is earnings in 1978 in thousands of dollars
tools: Python · OES engine · NumPy · statsmodels
oes: qte
lead: Use this guide when the average effect hides the story: a programme may lift the bottom of the distribution, the top, or leave the middle unchanged. The quantile treatment effect (QTE) is the difference between the treated and control outcome distributions at a given quantile. The guide estimates it with bootstrap intervals, adds a conditional quantile regression with covariates, and tests with Kolmogorov-Smirnov statistics whether the two distributions differ and whether one lies entirely to the right of the other (first-order stochastic dominance).
---

## The question and the assumptions

**Where in the distribution does the treatment act?** For a binary treatment \(T\) the quantile treatment effect at quantile \(\tau\) is

\[
\Delta(\tau) = Q_{Y(1)}(\tau) - Q_{Y(0)}(\tau),
\]

the difference between the \(\tau\)-th quantiles of the outcome with and without treatment. With random assignment the two quantiles are estimated from the two arms and need no covariates. A QTE of 3 at the 75th percentile means that the person at the 75th percentile of the treated outcomes earns 3 more than the person at the 75th percentile of the control outcomes. It is **not** the effect on the person who is at the 75th percentile without treatment unless treatment leaves everybody's rank unchanged (rank invariance). The Kolmogorov-Smirnov statistic \(\sup_y |F_1(y) - F_0(y)|\) tests whether the two distributions are equal, and its one-sided versions test dominance.

<div class="read" markdown="1">
**Three checks before you report quantile effects**

1. **Is treatment randomised or ignorable?** Without it the difference in quantiles is not causal.
2. **Does a rank-preserving reading make sense?** Say whether you mean a shift of the distribution or an effect on individuals at a rank.
3. **Are many quantiles tested?** Pointwise intervals do not control the error across the grid, so do not pick the quantile with the largest effect after looking.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We ask where in the earnings distribution job training acts. The treatment is random assignment to the National Supported Work programme and the outcome is earnings in 1978, in thousands of dollars. We estimate quantile treatment effects at the 10th, 25th, 50th, 75th and 90th percentiles as differences between the quantiles of the treated and control groups, with percentile confidence intervals from 500 bootstrap resamples. We compare them with conditional quantile regressions on age, education, race, marital status, degree status and earnings in 1974 and 1975, and test equality and first-order dominance of the two distributions with Kolmogorov-Smirnov statistics and permutation p-values.
</div>

## Load the data and look at the outcome

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("nsw_mediation.csv").read()
df = pd.read_csv("nsw_mediation.csv")
covs = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
print(len(df), "workers;", int(df.train.sum()), "treated")
print(df.groupby("train").re78.describe().round(2))
print("share with zero earnings:", (df.re78 == 0).groupby(df.train).mean().round(3).to_dict())
```

<div class="read" markdown="1">
**How to read it**

Mean earnings are 6.35 for the trained and 4.55 for the controls, an average gain of 1.79 thousand dollars. A large part of each group has no earnings at all: 35.4% of the controls and 24.3% of the trained. The outcome is therefore a mass at zero followed by a long right tail, and the mean is driven by the tail. This is the kind of distribution where an average effect says little about who gains, and the lower quantiles cannot move if the programme does not get people out of zero earnings.
</div>

## Estimate the quantile treatment effects

```python
spec = dict(outcome="re78", treat="train", covariates=covs, boot=500, taus="0.1,0.25,0.5,0.75,0.9")
r = oes.run("qte", data, spec)
oes.summary(r)
oes.show(r, "Quantile treatment")
oes.plot(r, "Quantile treatment effect by", ncols=1, size=(6.5, 3))
```

<div class="read" markdown="1">
**How to read it**

The effect is zero at the 10th percentile, where both groups have zero earnings. At the 25th percentile it is 0.49 (0.00 against 0.49, interval 0.00 to 1.08): the share of zeros is 35% among the controls and 24% among the trained, so the 25th percentile of the controls is still zero while that of the trained is positive. At the median the effect is 1.09 (3.14 against 4.23, interval -0.48 to 2.99), at the 75th percentile 2.35 (7.29 against 9.64, interval 0.62 to 3.85) and at the 90th percentile 3.20 (11.36 against 14.55, interval 0.07 to 6.54). The gains grow along the distribution and their intervals exclude zero only at the 75th and 90th percentiles, so the programme moves the upper part of the earnings distribution most. The average of 1.79 is the weighted total of these.
</div>

## Add covariates: conditional quantile regression

```python
oes.show(r, "Conditional quantile")
```

<div class="read" markdown="1">
**How to read it**

With the covariates the coefficient on treatment is 0.00 at the 10th and 25th percentiles, 1.07 at the median (p = 0.083), 2.65 at the 75th percentile (p = 0.0007) and 2.13 at the 90th (p = 0.060). The pattern matches the unconditional effects and the 75th percentile is now sharply estimated, because the covariates, mainly earnings in 1974 and 1975, explain much of the spread. A conditional quantile effect is the effect at a quantile of the outcome among workers with the same covariates, so it equals the unconditional QTE only if the covariates affect all quantiles alike. Here the two agree closely, which is what randomisation implies on average.
</div>

## Test whether the distributions differ

```python
oes.show(r, "Tests on the whole")
```

<div class="read" markdown="1">
**How to read it**

The Kolmogorov-Smirnov statistic is 0.132, with a permutation p-value of 0.039, so the earnings distributions of the two groups differ at the 5% level. The one-sided test of dominance rejects that the treated outcome is not larger (p = 0.012), while the test in the other direction cannot be rejected (statistic 0.000, p = 1.0). The treated distribution lies to the right of the control distribution at every earnings level: there is no earnings level at which the trained workers are more likely than the controls to be below it. This is a stronger statement than a positive average, because it implies that every utility function that increases in earnings gives a higher value for training.
</div>

## Robustness to the quantile grid and the bootstrap

```python
rows = {}
for name, tau in {"deciles": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", "upper tail": "0.6,0.7,0.8,0.9,0.95"}.items():
    for seed in (0, 1):
        rr = oes.run("qte", data, dict(outcome="re78", treat="train", taus=tau, boot=300, seed=seed))
        for x in rr["tables"][0]["rows"]: rows.setdefault((float(x[0])), []).append(float(x[3]))
print(pd.DataFrame({"QTE": {k: round(np.mean(v), 3) for k, v in rows.items()}, "range across runs": {k: round(max(v) - min(v), 3) for k, v in rows.items()}}))
```

<div class="read" markdown="1">
**How to read it**

The point estimates do not depend on the grid or the seed, because they are differences of sample quantiles, so the range across runs is zero; only the bootstrap intervals move with the seed. The effect is zero up to the 20th percentile, 0.94 at the 30th, about 1.1 to 1.5 through the middle of the distribution and rising in the upper tail and reaching 3.20 at the 90th and 3.93 at the 95th percentile. A sensible summary is the shape (zero in the bottom fifth, increasing above) and not any single quantile, and the intervals at individual quantiles should not be read as simultaneous.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Assignment to job training raises mean earnings by 1.79 thousand dollars, but the gain is not spread evenly. The quantile treatment effect is zero at the 10th percentile and 0.49 at the 25th, where a third of the controls have no earnings, 1.09 at the median (95% bootstrap interval -0.48 to 2.99), 2.35 at the 75th percentile (0.62 to 3.85) and 3.20 at the 90th (0.07 to 6.54). Conditional quantile regressions with covariates give the same pattern (2.65 at the 75th percentile, p = 0.0007). A Kolmogorov-Smirnov test rejects equality of the earnings distributions (p = 0.039) and the one-sided test rejects that the control distribution lies to the right (p = 0.012), so the treated distribution dominates the control distribution in the first-order sense. We read the effects as differences between distributions, not as effects on individuals at a given rank, which would need rank invariance, and we do not adjust the quantile intervals for the number of quantiles examined.
</div>

| Report | What | Where |
|---|---|---|
| Design | Quantile effect, rank invariance | Step 1 |
| Data | Outcome distribution, mass at zero | Step 2 |
| QTE | Quantiles, bootstrap intervals, chart | Step 3 |
| Conditional | Quantile regression with covariates | Step 4 |
| Tests | KS equality and dominance | Step 5 |
| Robustness | Quantile grid and seed | Step 6 |

**References.** Doksum, K. (1974), "Empirical Probability Plots and Statistical Inference for Nonlinear Models in the Two-Sample Case," *Annals of Statistics* 2. Koenker, R. and Bassett, G. (1978), "Regression Quantiles," *Econometrica* 46. Abadie, A. (2002), "Bootstrap Tests for Distributional Treatment Effects in Instrumental Variable Models," *Journal of the American Statistical Association* 97. Firpo, S. (2007), "Efficient Semiparametric Estimation of Quantile Treatment Effects," *Econometrica* 75. Chernozhukov, V., Fernández-Val, I. and Melly, B. (2013), "Inference on Counterfactual Distributions," *Econometrica* 81.
