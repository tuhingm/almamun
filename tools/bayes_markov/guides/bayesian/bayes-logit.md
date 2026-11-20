title: Bayesian Logit and Probit
short: Bayesian logit/probit
when: A yes/no outcome where you want probabilities for effects, a prior, or an estimate that survives a small or separated sample
data: Wooldridge MROZ, 753 married women in 1975: labour force participation, with income, schooling, experience, age and children
tools: Python · OES engine · statsmodels
oes: bayes_binary
lead: Use this guide when the outcome is binary (works or not, defaults or not, adopts or not) and you want the answer as posterior probabilities, with the effect on the probability of the outcome and a prior you can write down. With a reasonable sample the posterior means are close to maximum likelihood; the value is in the interpretation, the marginal effects with credible intervals and the cases where maximum likelihood breaks down. The guide covers the priors, the two samplers, marginal effects, convergence, fit, robustness, separation and the write-up.
---

## The question and the model

**What makes a married woman more likely to be in the labour force, and by how much?** The data are 753 married women in 1975; 568 in every thousand were working. The outcome \(y_i\) is 1 if the woman works, and the model for the probability is

\[
\Pr(y_i = 1 \mid x_i) = F(\alpha + x_i'\beta), \qquad F = \Lambda \ (\text{logit}) \ \text{or} \ \Phi \ (\text{probit}).
\]

The Bayesian version adds a prior on \(\beta\) and reports its posterior distribution. The engine uses an independent normal prior on each slope, \(\beta_j \sim N(0, s^2)\) **per one-standard-deviation change in \(x_j\)** (Gelman et al. 2008), so a single number \(s\) is meaningful whatever the units of the variables; the default \(s = 2.5\) is weakly informative: a one-s.d. change in a regressor is unlikely to move the log odds by more than about 5. The constant has an \(N(0, 10^2)\) prior.

| Link | Sampler | Properties |
|---|---|---|
| **Probit** | Albert-Chib (1993) Gibbs sampler: draw a latent normal variable for each observation, then the coefficients | Exact conditional distributions: no tuning, no rejection |
| **Logit** | Independence Metropolis-Hastings with a Student-\(t\) proposal centred at the posterior mode | Acceptance rate is reported; draws are correlated |

Both link functions give almost the same probabilities; the logit coefficients are about 1.6 to 1.9 times the probit ones, and the logit has odds ratios. The engine runs several chains from different starting points and reports the R-hat statistic.

<div class="read" markdown="1">
**Why Bayesian?**

Three reasons. (1) The marginal effect on the probability is a nonlinear function of the coefficients and the data; with draws you get its full posterior, with an interval that means what people think it means, and no delta method. (2) A prior can include earlier evidence. (3) With a rare outcome, a small sample or **separation** (a regressor that predicts the outcome perfectly), maximum likelihood either fails or produces estimates that are infinite; a prior keeps the problem well posed (Step 8).
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate the probability of labour force participation by Bayesian logit (Polson et al. 2013 and Gelman et al. 2008 for the prior). Each slope has an independent normal prior with standard deviation 2.5 per one-standard-deviation change in its regressor, and the constant a normal prior with standard deviation 10. The posterior is sampled by independence Metropolis-Hastings with a Student-\(t\) proposal at the posterior mode, in two chains of 2,000 draws after 1,000 burn-in draws. We report posterior means, 95% credible intervals, odds ratios and average marginal effects computed draw by draw.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("mroz.csv").read()
df = pd.read_csv("mroz.csv")
x = ["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
print(df.groupby("inlf")[x].mean().round(2).T.rename(columns={0: "not working", 1: "working"}))
print("share working:", df["inlf"].mean().round(3), " n =", len(df))
```

<div class="read" markdown="1">
**How to read it**

Of 753 women, 57% were in the labour force. Working women are better educated (12.7 years against 11.8), have much more experience (13.0 years against 7.5), are slightly younger (42.0 against 43.3) and have far fewer small children (0.14 against 0.37 children under six on average), and live in households with a somewhat lower income from other sources (\$18,900 against \$21,700). `nwifeinc` is the husband's and other household income in thousands of dollars, so it is the income a woman has if she does not work. `kidslt6` counts children under six and `kidsge6` children aged six to eighteen.
</div>

## Estimate

```python
spec = dict(y="inlf", x=x, link="logit")
r = oes.run("bayes_binary", data, spec)
oes.summary(r)
oes.show(r, "Posterior of the coefficients")
oes.show(r, "Maximum likelihood")
```

<div class="read" markdown="1">
**How to read it**

Each row is a posterior distribution for a coefficient in the log odds. An extra year of education raises the log odds of working by 0.229 (posterior standard deviation 0.044; 95% credible interval 0.146 to 0.316), and a year of experience by 0.121. Age lowers it by 0.092 a year, and each child under six by 1.46, a large effect. Household income from others lowers it by 0.020 per thousand dollars, with a posterior probability of 0.994 that the effect is negative. Children aged six or older have a coefficient of 0.058, with a posterior probability of being positive of only 0.775: the data cannot tell the sign.

The second table sets the maximum likelihood estimates against the posterior means and standard deviations. They agree to within 0.12 standard errors for every coefficient (education 0.227 against 0.229; children under six \(-1.439\) against \(-1.456\)). With 753 observations and a weak prior this is the expected result: the Bayesian logit reports the same answer with a better interpretation.
</div>

## Odds ratios and marginal effects

The coefficient is not the effect on the probability. The odds ratio \(e^{\beta}\) is the factor by which a one-unit change multiplies the odds of working. The **average marginal effect** is the change in the probability, averaged over the women in the sample; for a dummy variable it is the change from 0 to 1. Both are computed for each posterior draw, so they come with credible intervals.

```python
oes.show(r, "Posterior of the odds ratios")
oes.show(r, "Posterior of the average marginal effects")
```

<div class="read" markdown="1">
**How to read it**

A year of education multiplies the odds of working by 1.26 (95% interval 1.16 to 1.37); each child under six multiplies them by 0.24 (0.15 to 0.35), so the odds fall by about three-quarters.

In probability terms, the average woman's probability of working rises by 4.2 percentage points per year of education (95% interval 2.7 to 5.5), by 2.2 points per year of experience, and falls by 1.6 points per year of age and by 25.7 points per child under six (interval \(-31.6\) to \(-19.5\)). An extra thousand dollars of other household income lowers it by 0.4 points (\(-0.6\) to \(-0.1\)). The effect of children over six, 1.1 points, has an interval from \(-1.7\) to 3.8 points that includes zero. State effects on the probability in the paper, because readers cannot interpret log odds, and report the interval.
</div>

## Check the sampler

```python
oes.show(r, "MCMC convergence diagnostics")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

R-hat compares the variance between the two chains with the variance within each; values near one, as here (at most 1.002), mean the chains have found the same distribution. The effective sample size is the number of independent draws the 4,000 correlated draws are worth: between 1,810 and 2,341 for every parameter, which is plenty for means and intervals. The Metropolis-Hastings acceptance rate, in the note under the table, is 0.73: the proposal fits the posterior closely, so most proposed draws are accepted. The trace plot shows the draws of experience, the best-determined coefficient, wandering around a stable level, and the histogram is the posterior density.

If R-hat is above 1.05 or the effective sample size is a few dozen, do not report the results: run longer, drop highly collinear regressors or tighten the prior. Step 8 shows that case.
</div>

## Diagnostics: does the model fit?

```python
oes.show(r, "Fit and posterior predictive check")
```

<div class="read" markdown="1">
**How to read it**

A **posterior predictive check** simulates new data from the fitted model and asks whether they look like the real data. The model predicts that 57.1% of women work, with a 95% range from 53.0% to 61.1% across posterior draws, and the sample share (56.8%) lies inside it. This is a weak check, since a logit with a constant reproduces the share by construction, but it confirms that nothing is wrong with the constant.

Discrimination is moderate. The model classifies 74.0% of women correctly at a 0.5 cut-off, against 56.8% by always predicting that a woman works, and the area under the ROC curve is 0.797 (0.5 is chance, 1 is perfect). The Brier score is 0.181; always predicting the sample share would give 0.245. Participation depends on much that the model does not observe, which is why 26% are still misclassified.
</div>

## Robustness: the prior and the link

```python
oes.show(r, "Prior sensitivity")
rp = oes.run("bayes_binary", data, dict(spec, link="probit"))
ame = lambda res: oes.frame(next(t for t in res["tables"] if t["title"].startswith("Posterior of the average")))["Mean"]
print(pd.DataFrame({"logit": ame(r), "probit": ame(rp)}).round(4))
```

<div class="read" markdown="1">
**How to read it**

Each row of the first table is a separate run with a different prior standard deviation, from 0.5 (tight) to 100 (almost flat). The slopes for \(s \ge 2.5\) are within 0.005 of each other, and the maximum likelihood row is within about 0.01 of them. The tight prior pulls the large coefficients towards zero: children under six \(-1.35\) at \(s = 0.5\) against \(-1.44\), and education 0.218 against 0.227. No conclusion changes: the signs and sizes are the same under every prior.

The second table compares the average marginal effects under the two links: they differ by less than 0.01 for every variable (education 0.0415 and 0.0404; children under six \(-0.257\) and \(-0.266\)). Logit and probit coefficients differ by a scale factor; marginal effects are the same quantities and are the sensible thing to compare.
</div>

## When maximum likelihood fails: separation

Women with three or more children under six are rare: three women, none of whom works. A dummy for them predicts the outcome perfectly. Maximum likelihood cannot estimate its coefficient.

```python
import statsmodels.api as sm
df["kids3"] = (df["kidslt6"] >= 3).astype(float)
x2 = ["educ", "exper", "age", "kids3"]
ml = sm.Logit(df["inlf"], sm.add_constant(df[x2])).fit(disp=0, maxiter=200)
print("ML  kids3 coefficient:", ml.params["kids3"].round(1), " standard error:", ml.bse["kids3"].round(0))
rows = []
for s in (0.5, 1, 2.5, 10):
    rs = oes.run("bayes_binary", df.to_csv(index=False), dict(y="inlf", x=x2, prior_sd=s, draws=4000))
    co = oes.frame(next(t for t in rs["tables"] if t["title"].startswith("Posterior of the coefficients")))
    am = oes.frame(next(t for t in rs["tables"] if t["title"].startswith("Posterior of the average")))
    rows.append([s, co.loc["kids3", "Mean"], co.loc["kids3", "Median"], am.loc["kids3", "Mean"], am.loc["kids3", "2.5%"], am.loc["kids3", "97.5%"]])
print(pd.DataFrame(rows, columns=["prior s.d.", "coef. mean", "coef. median", "AME mean", "AME 2.5%", "AME 97.5%"]).round(3))
oes.show(rs, "MCMC convergence diagnostics")
```

<div class="read" markdown="1">
**How to read it**

The likelihood keeps rising as the coefficient falls, so the maximum does not exist: the fitting routine stops at \(-18.7\) with a standard error of 5,566, which means nothing. A Bayesian estimate exists, because the prior stops the posterior from running off to \(-\infty\), but it **is the prior's answer, not the data's**. The posterior mean of the coefficient moves from \(-7.8\) to \(-140\) as the prior standard deviation goes from 0.5 to 10 per standard deviation of the dummy (which is only 0.063, so a prior s.d. of 2.5 is a coefficient s.d. of about 40), and the diagnostics sound the alarm: the final run (prior s.d. 10) has an effective sample size of about 60 for the dummy and an R-hat of 1.44. Do not report the coefficient.

What the data do identify is the effect on the **probability**. The average marginal effect is stable: \(-0.53\) at \(s = 0.5\), \(-0.55\), \(-0.57\) and \(-0.57\) at \(s = 10\), with an interval whose upper end is \(-0.26\) at the tightest prior and between \(-0.35\) and \(-0.53\) at the others. Having three small children lowers the probability of working by 53 to 57 points, which is the sample's finding (none of three works); but with three women the result is a statement about those three. Say so, and either pool categories (two or more children) or treat the dummy as uninformative and drop it.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Table 1 reports the Bayesian logit for labour force participation among 753 married women. Education and experience raise the probability of working and age, household income from other sources and young children lower it. An extra year of schooling multiplies the odds of working by 1.26 (95% credible interval 1.16 to 1.37) and raises the probability by 4.2 percentage points (2.7 to 5.5); a child under six multiplies the odds by 0.24 (0.15 to 0.35) and lowers the probability by 25.7 points (19.5 to 31.6). Children aged six to eighteen have no clear effect (posterior probability of a positive effect 0.78). The two chains converge (R-hat at most 1.002, effective sample sizes above 1,800). The posterior means are within 0.12 standard errors of maximum likelihood, and the conclusions are the same under priors with standard deviation from 0.5 to 100 per standard deviation of the regressor, and under a probit link (average marginal effects within 0.01). The model classifies 74% of women correctly (area under the ROC curve 0.80), against 57% for always predicting participation.
</div>

| Report | What | Where |
|---|---|---|
| Prior | Normal, s.d. per one-s.d. change in \(x\); the constant's prior | Step 1 |
| Results | Posterior mean, interval; odds ratios | Steps 3 and 4 |
| Effects | Average marginal effects with credible intervals | Step 4 |
| Sampler | Chains, draws, R-hat, effective sample size, acceptance rate | Step 5 |
| Fit | Posterior predictive share, accuracy against the majority rate, AUC | Step 6 |
| Robustness | Posterior means under different priors; the other link | Step 7 |

**References.** Albert and Chib (1993), *Journal of the American Statistical Association*; Gelman, Jakulin, Pittau and Su (2008), *Annals of Applied Statistics*; Polson, Scott and Windle (2013), *Journal of the American Statistical Association*; Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed.; Mroz (1987), *Econometrica*.
