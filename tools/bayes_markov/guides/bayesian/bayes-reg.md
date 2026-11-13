title: Bayesian Linear Regression with a g-Prior
short: Bayesian regression
when: A cross-section regression where you want posterior probabilities, prior information or model comparison
data: Wooldridge WAGE2, 663 U.S. men with complete data: log wage, education, experience, tenure, IQ and demographics
tools: Python · OES engine · statsmodels
oes: bayes_reg
lead: Use this guide when you want a regression that reports the probability that an effect is positive, uses information from earlier studies, or compares specifications by how well they explain the data. With a reasonable sample the posterior means are close to OLS; the value is in the interpretation, the prior sensitivity table and the marginal likelihood. The guide covers the prior, estimation, model comparison, diagnostics, robustness and the write-up.
---

## The question and the model

**How much does an extra year of education raise wages, once experience, tenure, ability and demographics are held fixed?** This is the Mincer wage regression. The model is

\[
\ln w_i = \alpha + x_i'\beta + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2),
\]

with \(x_i\) containing education, experience, tenure, IQ and dummies for marriage, race, region and city. OLS gives one number per coefficient and a standard error. A Bayesian regression gives a whole **posterior distribution** for each coefficient: you can read off the probability that the return to schooling exceeds 5%, combine the data with the results of earlier studies, and compare specifications without a p-value threshold.

The OES engine uses a **conjugate** Normal-inverse-gamma prior. Everything is exact: no simulation, no convergence to check, and the log marginal likelihood is available in closed form. Two priors are offered for the slopes:

| Prior | What it says | When to use it |
|---|---|---|
| **Zellner's g-prior** | \(\beta \sim N(\beta_0, g\,\sigma^2 (X'X)^{-1})\): the prior is as informative as \(1/g\) of an observation's worth of the sample, scaled to the regressors | Default. Unit-free, so it needs no judgement about the scale of each variable. \(g = n\) is the "unit-information prior" |
| **Normal** | \(\beta_j \sim N(\beta_{0j}, s^2)\), the same standard deviation for every slope | When the variables are in comparable units, or when you have a prior from earlier work |

The constant has a flat prior and \(\sigma^2\) an inverse-gamma(0.01, 0.01) prior in both cases. A large \(g\) (or large \(s\)) lets the data speak; a small one pulls the slopes towards the prior mean \(\beta_0\).

<div class="read" markdown="1">
**Frequentist or Bayesian?**

With 663 observations and a weak prior, the two give almost the same numbers, and that is the point: a Bayesian regression is a way of *reporting* the answer (probabilities, intervals that mean what people think they mean) and of *using* extra information, not a different estimate. Use it when the sample is small, when you have genuine prior information, or when model comparison is part of the question.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate the log-wage equation by Bayesian linear regression with a Zellner g-prior on the slopes (\(g = n\), the unit-information prior), a flat prior on the constant and an inverse-gamma(0.01, 0.01) prior on the error variance (Zellner 1986; Liang et al. 2008). With this conjugate prior the marginal posterior of each coefficient is a Student-\(t\) distribution, available in closed form, and we report its mean, standard deviation, 95% credible interval and the posterior probability that the coefficient is positive.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("wage2.csv").read()
df = pd.read_csv("wage2.csv")
x = ["educ", "exper", "tenure", "IQ", "married", "black", "south", "urban"]
print(df[["lwage"] + x].describe().round(2).loc[["count", "mean", "std", "min", "max"]])
```

<div class="read" markdown="1">
**How to read it**

663 men, with log wages averaging 6.81 (about \$900 a month) and a standard deviation of 0.41. Education averages 13.7 years. Nine in ten are married, 8% are Black, 32% live in the South and 72% in a city. Dummy variables like these have a standard deviation below 0.5, while IQ has 14.7: the **g-prior** is useful here because it scales the prior for each coefficient by the spread of its regressor, so you do not have to choose a different prior standard deviation for IQ and for a dummy.
</div>

## Estimate

```python
spec = dict(y="lwage", x=x)
r = oes.run("bayes_reg", data, spec)
oes.summary(r)
oes.show(r, "Posterior")
oes.show(r, "OLS")
```

<div class="read" markdown="1">
**How to read it**

Each row is a posterior distribution. Take education: the posterior mean is 0.055 with a standard deviation of 0.008, so a year of schooling raises wages by about 5.5%, and there is a 95% probability that the effect lies between 3.9% and 7.1%. That is a direct statement about the parameter; a frequentist confidence interval cannot be read that way. The last column, \(P(>0)\), is the posterior probability that the effect is positive: 1.000 for education, experience, marriage and city; 0.999 for tenure and IQ; and 0.012 and 0.019 for Black and South, whose effects are negative with high probability.

Compare with OLS: coefficients agree to the third decimal place. The g-prior with \(g = n\) carries the weight of a single observation, so with 663 observations it moves nothing. This is the expected result with a weak prior and a decent sample; Step 6 shows what happens with a stronger prior.
</div>

## Compare specifications by marginal likelihood

The **log marginal likelihood** is the probability of the data under the model with the coefficients integrated out. It rewards fit and automatically penalises a model for having parameters it does not need. The difference between two models is the log **Bayes factor**.

```python
base = oes.run("bayes_reg", data, spec)
lml = {"full model": base["summary"][4]["value"]}
for drop in ["IQ", "tenure", "black", "south", "urban"]:
    rr = oes.run("bayes_reg", data, dict(spec, x=[v for v in x if v != drop]))
    lml[f"without {drop}"] = rr["summary"][4]["value"]
out = pd.Series(lml, name="log marginal likelihood").to_frame()
out["log Bayes factor (full vs. model)"] = out.iloc[0, 0] - out["log marginal likelihood"]
print(out.round(2))
```

<div class="read" markdown="1">
**How to read it**

A log Bayes factor above about 1 is "some evidence" for the full model, above 2.3 is "strong" and above 4.6 "decisive" (Jeffreys' scale). Dropping `urban` costs almost 17 log points: the city dummy is decisive. Dropping `IQ` costs 2.0 and `tenure` 1.3: moderate evidence for both. Dropping `south` or `black` *raises* the marginal likelihood by 1.1 and 0.7: the data, after the penalty for extra parameters, slightly prefer the model without them. This does not mean the coefficients are zero (both have \(P(>0)\) below 0.02); it means they are not worth their cost in predictive terms.

Report the table if the specification is a topic in the paper, and describe the scale you use. Do not use the Bayes factor as a mechanical rule; use it with the posterior intervals and with economic reasoning.
</div>

## Use prior information

If earlier studies find a return to education of about 7%, you can say so. `prior_mean` sets the prior mean of each slope: one number for all, or a list in the order of the regressors.

```python
r7 = oes.run("bayes_reg", data, dict(spec, prior_mean="0.07, 0, 0, 0, 0, 0, 0, 0"))
print("log marginal likelihood:", r7["summary"][4]["value"], "versus", base["summary"][4]["value"], "with a zero prior mean")
oes.show(r7, "Posterior", rows=3)
```

<div class="read" markdown="1">
**How to read it**

The posterior mean of the education coefficient barely moves (0.0551 against 0.0550), because this prior carries the weight of one observation. To let earlier work matter, use a **small g** or a Normal prior with a small standard deviation, and report the sensitivity table below. In a small sample the prior would matter much more, and writing it down becomes part of the paper's identification of the result.
</div>

## Robustness: how much does the prior matter?

```python
oes.show(r, "Prior sensitivity")
```

<div class="read" markdown="1">
**How to read it**

Each row re-estimates the slopes under a different prior. For \(g \ge 663\) and for the Normal priors with standard deviation 1 or 10, the slopes are the same as OLS: the data dominate. With \(g = 10\) the coefficients shrink by about 9% (education 0.050), and with \(g = 1\) by half (education 0.028; marriage 0.10 against 0.20). The tight Normal prior (standard deviation 0.1) pulls every slope towards zero, with the largest effect on marriage (0.075) and race (\(-0.037\)), which have the largest coefficients.

What to say: the signs and ranking of the effects do not depend on the prior unless it is much more informative than the data. If a result flips only for a prior that acts like a handful of observations, it is not robust. The table is cheap evidence to show a referee.
</div>

For a non-conjugate prior (a Normal prior with the *same* standard deviation for each slope and independent of \(\sigma^2\)), the posterior is obtained by a Gibbs sampler, which needs a convergence check:

```python
rg = oes.run("bayes_reg", data, dict(spec, method="gibbs", prior="normal", prior_sd=1, draws=3000, burn=500))
oes.show(rg, "MCMC")
```

<div class="read" markdown="1">
**How to read it**

The effective sample size equals the number of draws (3000) for every coefficient, because a Gibbs sampler for a linear model draws from two exact conditional distributions and the chain mixes immediately. Geweke's test compares the first 10% and the last 50% of the chain: a statistic with |z| above 2 would mean the chain had not settled. It is borderline for `black` (z = 2.0): with nine parameters one such value is expected by chance, and increasing the burn-in confirms nothing changes. Report effective sample sizes for any sampler you use.
</div>

## Diagnostics: residuals

```python
import statsmodels.api as sm
oes.show(r, "Residual diagnostics")
ols = sm.OLS(df["lwage"], sm.add_constant(df[x])).fit(cov_type="HC3")
cmp = pd.DataFrame({"posterior sd": oes.frame(next(t for t in r["tables"] if t["title"].startswith("Posterior")))["Std. dev."],
                    "OLS HC3 s.e.": ols.bse})
cmp["ratio"] = cmp["OLS HC3 s.e."] / cmp["posterior sd"]
print(cmp.round(4))
```

<div class="read" markdown="1">
**How to read it**

Residual autocorrelation is rejected by neither the portmanteau nor the Ljung-Box test, as expected in a cross-section. Residuals are not normal (Jarque-Bera rejects) and the variance is not constant (Breusch-Pagan, p = 0.003). The regression coefficients stay consistent, but the posterior standard deviations assume a constant variance and are too small if the variance is unevenly spread: the table compares the posterior standard deviations with heteroskedasticity-robust (HC3) OLS standard errors. The ratios run from 0.90 (Black) to 1.06 (South), so the robust errors are within about 10% of the posterior ones and the conclusions stand, with one qualification: with robust errors the South coefficient is borderline (t about 1.96). Report the robust errors next to the posterior as a check; the point estimates are the same.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Table 1 reports posterior summaries. A year of education raises hourly wages by 5.5% (posterior standard deviation 0.8 percentage points; 95% credible interval 3.9% to 7.1%), with a posterior probability of one that the effect is positive. Experience, tenure, IQ, marriage and living in a city have positive effects (probability above 0.999), while the effects of being Black or living in the South are negative with probabilities of 0.988 and 0.981. The posterior means are almost identical to OLS because the g-prior carries the weight of one observation. The log Bayes factors indicate decisive support for the city dummy (16.8), and moderate support for IQ (2.0) and tenure (1.3); the data slightly prefer a model without the South and Black dummies, though their credible intervals exclude zero. The signs and ranking of the effects are unchanged for all priors except those that carry the weight of fewer than ten observations (Table 2). Residual diagnostics show heteroskedasticity, so we also compare with heteroskedasticity-robust OLS standard errors, which are within 10% of the posterior standard deviations; the South coefficient is then only borderline significant.
</div>

| Report | What | Where |
|---|---|---|
| Prior | Family, \(g\) or \(s\), \(\beta_0\), the \(\sigma^2\) prior | Step 1 |
| Results | Posterior mean, standard deviation, 95% interval, \(P(>0)\) | Step 3 |
| Comparison | Log marginal likelihood or Bayes factors, with the scale used | Step 4 |
| Robustness | Posterior means under different \(g\) and \(s\) | Step 6 |
| Diagnostics | Residual tests; effective sample size and Geweke for Gibbs | Steps 6 and 7 |

**References.** Zellner (1986), *Bayesian Inference and Decision Techniques*; Liang, Paulo, Molina, Clyde and Berger (2008), *Journal of the American Statistical Association*; Kass and Raftery (1995), *Journal of the American Statistical Association*; Mincer (1974), *Schooling, Experience, and Earnings*; Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed.
