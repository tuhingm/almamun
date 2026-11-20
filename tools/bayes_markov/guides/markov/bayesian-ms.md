title: Bayesian Markov-Switching Regression
short: Bayesian MS regression
when: Hidden regimes (recession and expansion, calm and turbulent, high and low inflation) where you want probabilities for the regimes, the transition matrix and the durations, not only point estimates
data: U.S. quarterly GNP growth, 1951Q2-1984Q4 (Hamilton 1989), with the NBER recession dates; U.S. quarterly inflation and unemployment, 1959-2009
tools: Python · OES engine · statsmodels
oes: bayes_ms
lead: Use this guide when a series or a regression seems to move between hidden regimes, such as expansion and recession or a low-inflation and a high-inflation era, and you want the answer as probabilities: how likely each period is to be in each regime, how likely the regime is to persist, how long it lasts and how different the regimes are. The Bayesian sampler gives a full posterior for all of these, with small-sample intervals and a prior that you can state and test. The guide covers the model, estimation on Hamilton's GNP data, dating the regimes against the NBER, switching variance, the prior on persistence, a warning about autoregressive terms, convergence, a coverage simulation, a Phillips curve with switching slopes and the write-up.
---

## The question and the model

**Can the data find the recessions without being told where they are?** In a Markov-switching regression the regime \(s_t \in \{1, \dots, K\}\) is hidden and follows a first-order Markov chain with transition probabilities \(p_{ij} = \Pr(s_t = j \mid s_{t-1} = i)\); conditional on the regime the series is an ordinary regression:

\[
y_t = \mu_{s_t} + w_t'\gamma_{s_t} + \sigma_{s_t}\varepsilon_t, \qquad \varepsilon_t \sim N(0, 1).
\]

The intercept always switches; slopes and the variance may or may not. The maximum-likelihood version (the Markov-switching guides) gives point estimates and asymptotic standard errors, which are often unreliable here because the likelihood is multi-modal and the effective sample is the number of regime changes, which is small. The **Bayesian version** (Albert and Chib 1993; Chib 1996; Kim and Nelson 1999) samples everything jointly:

1. the whole regime path \(s_{1:T}\), by forward filtering and backward sampling (Hamilton filter, then a draw for each period from the last backwards);
2. the regression coefficients, by a normal update given the regimes;
3. the variances, by inverse-gamma draws;
4. each row of the transition matrix, by a Dirichlet draw from the counts of transitions.

The result is a posterior for every parameter, for the duration \(1/(1 - p_{jj})\) of each regime, for the long-run share of time in each regime, and for the regime of every period. **Label switching** (the regimes can swap names between draws) is fixed by ordering the regimes by their intercepts (or by their variances), which the output reports.

| Output | What it answers |
|---|---|
| **Regime parameters** | Posterior mean, interval of the intercept, slopes and standard deviation in each regime, next to the maximum likelihood |
| **Transition probabilities** | Probability of moving between regimes, with intervals |
| **Durations and long-run shares** | How long each regime lasts and how much of the time it occupies |
| **Regime dates** | Runs of periods in which a regime is more likely than not, and agreement with a reference series |
| **Diagnostics** | R-hat and effective sample sizes, and the share of draws in which the labels had to be reordered |

<div class="write" markdown="1">
**How to write it: the model**

We model U.S. quarterly GNP growth as a two-regime Markov-switching regression with a regime-dependent mean and a common variance, estimated by Gibbs sampling (Albert and Chib 1993; Chib 1996). The regime path is drawn by forward filtering and backward sampling, the regime means by a normal update, the variance by an inverse-gamma update and each row of the transition matrix by a Dirichlet update. The priors are N(0, 5²) on the standardised regime means, inverse gamma(2, 1) on the standardised variance and Dirichlet(1, 1) on each row of the transition matrix. The regimes are ordered by their means to fix the labelling. We run two chains of 3,000 draws after 1,000 burn-in draws and check convergence with R-hat and effective sample sizes.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("hamilton_gnp.csv").read()
d = pd.read_csv("hamilton_gnp.csv")
print(d.shape, d.date.iloc[0], "to", d.date.iloc[-1])
print(d.gnp_growth.describe().round(2).loc[["mean", "std", "min", "max"]])
print("quarters the NBER calls a recession:", int(d.nber_recession.sum()), "of", len(d))
print("mean growth in NBER recession quarters:", d.gnp_growth[d.nber_recession == 1].mean().round(2), " in other quarters:", d.gnp_growth[d.nber_recession == 0].mean().round(2))
```

<div class="read" markdown="1">
**How to read it**

The 135 quarters of growth (in percent, annualised) have a mean of 0.74 and a standard deviation of 1.07, ranging from \(-2.39\) to 3.11. The NBER calls 27 of the quarters recessions; growth in those was \(-0.58\) on average and 1.08 in the rest. The NBER dates are not used in the estimation; they are the check against which the method's dates are judged.
</div>

## Estimate

```python
spec = dict(y="gnp_growth", time="date", reference="nber_recession", switching_variance="no", draws=3000, burn=1000)
r = oes.run("bayes_ms", data, spec)
oes.summary(r)
oes.show(r, "Posterior of the regime parameters")
oes.show(r, "Transition probabilities")
oes.show(r, "Expected duration")
```

<div class="read" markdown="1">
**How to read it**

The sampler is sound: R-hat is at most 1.003 and the smallest effective sample size is 481 out of 6,000 draws, and no draw needed relabelling, so the two regimes are well separated.

*Regime means.* Regime 1, the slow regime, has a mean growth of \(-0.50\) with a posterior standard deviation of 0.36 (95% interval \(-1.32\) to 0.11); regime 2, the fast regime, has 1.11 (0.86 to 1.40). The maximum-likelihood values (\(-0.49\) and 1.10) are close to the posterior means. The interval of the slow regime is wide because only a quarter of the quarters belong to it, and it just reaches zero: the data say that growth in the slow regime is probably negative, but not with certainty. The common standard deviation is 0.85 (0.73 to 0.98).

*Persistence.* The probability that the fast regime continues is 0.89 (0.74 to 0.97): an expected duration of 11 quarters (95% interval 4 to 29). The slow regime continues with probability 0.65 (0.34 to 0.86), for an expected duration of 3.3 quarters (1.5 to 7.2). These are asymmetric, as the business cycle is: expansions are long and recessions short. The long-run share of the slow regime is 25% (7% to 51%). The intervals are wide, and the maximum-likelihood standard errors would hide this; the Bayesian interval for a duration is the honest one.
</div>

## Date the regimes

```python
oes.show(r, "Episodes of regime 1")
oes.show(r, "Agreement with the reference")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The method finds eight slow-growth episodes, among them 1953Q3 to 1954Q2, 1957Q3 to 1958Q1, 1969Q4 to 1970Q2, 1974Q1 to 1975Q1 and 1981Q2 to 1982Q3. They are the recessions that the NBER dates as 1953 to 1954, 1957 to 1958, 1969 to 1970, 1973 to 1975 and 1981 to 1982. The posterior probability averages 0.86 over the first episode and 0.89 over the 1974 to 1975 one: the dating is clear for the large recessions and less so for the marginal ones (1960 at 0.62 and the single quarter 1970Q4 at 0.56).

Against the NBER series, the method classifies 93.3% of the quarters in the same way. It places 81.5% of the NBER recession quarters in regime 1 and only 3.7% of the other quarters, and the correlation between the probability of regime 1 and the recession indicator is 0.83. The first chart shows the probability of each regime over time; the second the series with the regime-weighted mean.
</div>

## Diagnostics: the sampler

```python
oes.show(r, "MCMC convergence")
```

<div class="read" markdown="1">
**How to read it**

R-hat is between 1.001 and 1.003 for every parameter and the effective sample sizes range from 481 (the mean of the slow regime) to 2,093. The staying probabilities, 952 and 660, are the quantities that mix most slowly in Markov-switching models because they depend on the sampled regime path. The last line of the note reports the share of draws in which the labels had to be reordered: 0.0% here. A high share (more than 10%) means that the regimes are barely separated and that the ordering is doing a lot of work; then reduce the number of regimes or add information.
</div>

## Let the variance switch

Recessions are often more volatile as well as slower. Let each regime have its own standard deviation.

```python
rv = oes.run("bayes_ms", data, dict(spec, switching_variance="yes", reference="nber_recession"))
oes.show(rv, "Posterior of the regime parameters")
oes.show(rv, "Transition probabilities")
oes.show(rv, "Agreement with the reference")
```

<div class="read" markdown="1">
**How to read it**

The standard deviation is 0.97 in the slow regime (95% interval 0.70 to 1.27) and 0.78 in the fast one (0.58 to 0.96), the same as the maximum-likelihood values (0.97 and 0.79), but the intervals overlap heavily: the data do not say that the regimes differ in volatility. The means move too, to \(-0.21\) and 1.21. The regime dating changes in a revealing way: the share of the NBER recession quarters in regime 1 rises from 0.82 to 0.96, but the share of other quarters in regime 1 also rises from 0.04 to 0.13, so the overall agreement falls from 93% to 89%. Letting the variance switch gives the slow regime more room to absorb volatile quarters that are not recessions. The constant-variance model is the better dating device for these data.
</div>

## Robustness: the prior on staying

The Dirichlet prior on each row of the transition matrix is flat by default. Because regimes are persistent you may want to add prior weight to the diagonal.

```python
rows = []
for stay in (0, 2, 5):
    rs = oes.run("bayes_ms", data, dict(spec, prior_stay=stay, draws=2000, burn=800))
    t = {q[0]: q for q in rs["tables"][1]["rows"]}
    dur = rs["tables"][2]["rows"]
    rows.append([stay, t["1 -> 1"][1], t["2 -> 2"][1], dur[0][1], dur[1][1]])
print(pd.DataFrame(rows, columns=["extra prior weight on staying", "P(1->1)", "P(2->2)", "duration regime 1", "duration regime 2"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Adding 2 or 5 extra prior observations of staying in a regime raises the posterior probability of staying in the slow regime from 0.65 to 0.70 and 0.74, and the expected duration from 3.4 to 6.9 and 4.5 quarters (the duration is a ratio of a small number and is noisy across runs). The fast regime hardly moves (0.886, 0.892, 0.899). With 135 quarters and about 25 in the slow regime the prior does matter for the slow regime, and it should be reported. The regime means and the dating, which depend mostly on the data, change little.
</div>

## A warning about autoregressive terms

Hamilton's own model has the regime in the mean of an autoregression. Here the lag of the series enters as a regressor with a common coefficient and only the intercept switches, which is a different model.

```python
ra = oes.run("bayes_ms", data, dict(spec, ar=1, draws=6000, burn=2000))
oes.show(ra, "Posterior of the regime parameters")
oes.show(ra, "Agreement with the reference")
oes.show(ra, "MCMC convergence")
```

<div class="read" markdown="1">
**How to read it**

With one lag, the autoregressive coefficient is 0.34 (95% interval 0.09 to 0.57) and the regimes blur: the slow regime's level is \(-0.14\) (interval \(-1.20\) to 0.75), the agreement with the NBER falls to 79%, the share of the NBER recession quarters in regime 1 falls to 63%, and in 3.8% of the draws the labels had to be reordered. The lag soaks up the persistence that the regimes were capturing. The effective sample sizes are 239 to 1,441 after 6,000 draws. With only 135 quarters the data cannot sharply identify the regimes and an autoregression at the same time. If you need Hamilton's mean-switching autoregression, use the **Markov-switching autoregression** guide; if you add lags here, check the agreement, the share of reordered draws and the effective sample sizes, as in this step.
</div>

## Diagnostics: do the intervals cover?

Simulate 30 series of 250 observations from a two-regime chain with known means (\(-1\) and 2), standard deviation 1 and transition probabilities of 0.9 and 0.85.

```python
rng = np.random.default_rng(0)
cover, acc = [], []
for s in range(30):
    n = 250; st = np.zeros(n, int)
    for t in range(1, n):
        st[t] = rng.choice(2, p=[0.9, 0.1] if st[t - 1] == 0 else [0.15, 0.85])
    y = np.where(st == 0, -1.0, 2.0) + rng.normal(size=n)
    sim = pd.DataFrame({"t": np.arange(n), "y": y})
    rs = oes.run("bayes_ms", sim.to_csv(index=False), dict(y="y", time="t", switching_variance="no", draws=800, burn=400, seed=s))
    t1, t2 = [q for q in rs["tables"][0]["rows"] if q[1] == "Intercept"]
    cover.append([t1[4] <= -1 <= t1[5], t2[4] <= 2 <= t2[5]])
    p1 = np.array(rs["charts"][0]["series"][0]["y"], float); acc.append(((p1 > 0.5) == (st == 0)).mean())
print("coverage of the 95% intervals of the two regime means:", np.mean(cover, axis=0).round(2))
print("average share of periods classified correctly:", np.mean(acc).round(3))
```

<div class="read" markdown="1">
**How to read it**

The 95% intervals of the two regime means contain the truth in 90% and 93% of the 30 samples (the standard error with 30 samples is about 4 points), and on average 97.4% of the periods are assigned to the right regime. The test suite adds a check against maximum likelihood, recovery of transition probabilities, variance switching, common and switching slopes, and that a series of pure noise does not produce two separated regimes.
</div>

## Switching slopes: the Phillips curve

The same sampler with switching slopes and variance. The relation between inflation and unemployment, 1959 to 2009; the intercept is the level of inflation at the average rate of unemployment.

```python
m = open("us_macro.csv").read()
rp = oes.run("bayes_ms", m, dict(y="inflation", x=["unemp"], time="date", switching_slopes="yes", switching_variance="yes", draws=15000, burn=3000))
oes.show(rp, "Posterior of the regime parameters")
oes.show(rp, "Transition probabilities")
oes.show(rp, "Episodes of regime 1")
oes.show(rp, "MCMC convergence")
```

<div class="read" markdown="1">
**How to read it**

The model separates two eras. The low-inflation regime holds 1959Q2 to 1972Q4 and 1983Q2 to 2009Q3: inflation at average unemployment is 2.8% (95% interval 2.4% to 3.2%), its standard deviation is 1.8 and the slope on unemployment is \(-0.18\) with an interval of \(-0.57\) to 0.21: a flat Phillips curve. The high-inflation regime holds 1973Q1 to 1983Q1, the Great Inflation: the level is 9.0% (5.9% to 11.3%), the volatility 3.4 and the slope \(-1.18\) (\(-2.18\) to 0.29), steeper but imprecise, since only 41 quarters belong to it.

Both regimes are persistent: the low-inflation regime continues with probability 0.97 and the high one with 0.93 (0.81 to 0.99). The convergence report shows R-hat up to 1.01 and effective sample sizes from 144 to 4,440 after 15,000 draws: with long regimes there are few transitions to learn from, so the variance and persistence parameters mix slowly. Use long runs for this model and report the diagnostics; the run here takes a minute or two.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Two regimes are identified in U.S. quarterly GNP growth. Mean growth is \(-0.50\) in the slow regime (95% interval \(-1.32\) to 0.11) and 1.11 in the fast regime (0.86 to 1.40), with a common standard deviation of 0.85. The fast regime persists with probability 0.89 (expected duration 11 quarters, 95% interval 4 to 29) and the slow regime with probability 0.65 (3.3 quarters, 1.5 to 7.2). The regimes match the NBER recessions: 93% of the quarters are classified the same way, 81.5% of the NBER recession quarters fall in the slow regime and 3.7% of the others. The sampler's R-hat is at most 1.003 and the effective sample sizes exceed 480. The posterior means are close to the maximum-likelihood estimates. Letting the variance switch, or adding an autoregressive term, blurs the regimes (agreement 89% and 79%), so the constant-variance specification is used. In simulation, the 95% intervals of the regime means covered the truth in 90% to 93% of the samples. Applied to inflation and unemployment, the model separates a low-inflation regime (2.8% at average unemployment, slope \(-0.18\)) from the 1973 to 1983 high-inflation regime (9.0%, slope \(-1.18\)).
</div>

| Report | What | Where |
|---|---|---|
| Model | Regimes, priors, sampler, labelling | Step 1 |
| Regime parameters | Means, standard deviation, intervals | Step 3 |
| Persistence | Transition probabilities, durations, long-run shares | Step 3 |
| Dates | Regime episodes, agreement with a reference | Step 4 |
| Convergence | R-hat, effective sample sizes, relabelled draws | Step 5 |
| Robustness | Switching variance, prior on staying, autoregressive terms | Steps 6, 7 and 8 |
| Coverage | Simulation | Step 9 |
| Switching slopes | Phillips curve regimes | Step 10 |

**References.** Hamilton (1989), *Econometrica*; Albert and Chib (1993), *Journal of Business and Economic Statistics*; Chib (1996), *Journal of Econometrics*; Kim and Nelson (1999), *State-Space Models with Regime Switching*, MIT Press; Frühwirth-Schnatter (2006), *Finite Mixture and Markov Switching Models*, Springer; Stephens (2000), *Journal of the Royal Statistical Society B*; Krolzig (1997), *Markov-Switching Vector Autoregressions*, Springer.
