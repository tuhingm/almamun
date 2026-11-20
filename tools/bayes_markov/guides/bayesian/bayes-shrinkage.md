title: Bayesian Shrinkage Regression (Horseshoe, Lasso, Ridge)
short: Bayesian shrinkage
when: Many regressors relative to the sample (interactions, polynomials, dozens of controls), and you want estimates that do not overfit
data: Wooldridge HPRICE2, 506 Boston-area census tracts: log house price, eight regressors, their squares and all 28 pairwise products (44 regressors)
tools: Python · OES engine · statsmodels
oes: bayes_shrink
lead: Use this guide when you have many regressors relative to the number of observations, because you add squares and interactions to capture nonlinearity or include dozens of controls, and ordinary least squares overfits. A shrinkage prior pulls the coefficients that the data do not support towards zero and leaves the ones they do support alone, and the posterior says how sure you are about each. The guide covers the three priors, estimation, which effects are real, the comparison with OLS, the sampler check, prior sensitivity, an out-of-sample test and the write-up.
---

## The question and the model

**How much do neighbourhood characteristics affect house prices, when you allow every regressor to enter nonlinearly and to interact with the others?** The hedonic price equation of Harrison and Rubinfeld (1978) has eight regressors: air pollution (`lnox`), rooms, distance to employment centres, access to highways, the property tax rate, the pupil-teacher ratio, the share of lower-status residents (`lowstat`) and crime. Adding their squares and all 28 pairwise products gives **44 regressors**. With 506 tracts OLS handles this; with 120 it does not: the model has almost as many parameters as observations, the coefficients are noisy and the predictions are worse than those of the simple model.

A **shrinkage prior** is a prior on the coefficients that puts most of its mass near zero. With standardised regressors,

\[
y_i = \alpha + x_i'\beta + \varepsilon_i, \qquad \beta_j \mid \lambda_j, \tau, \sigma \sim N\!\left(0, \; \sigma^2 \tau^2 \lambda_j^2\right),
\]

and the three priors differ in how \(\lambda_j\) (the scale of each coefficient) and \(\tau\) (the overall scale) are distributed:

| Prior | Scales | What it does |
|---|---|---|
| **Horseshoe** (Carvalho et al. 2010) | \(\lambda_j\) and \(\tau\) half-Cauchy | Shrinks small coefficients hard and leaves large ones nearly untouched: built for a few large effects among many noise terms |
| **Bayesian lasso** (Park and Casella 2008) | \(\lambda_j^2\) exponential: Laplace prior on \(\beta_j\) | Shrinks every coefficient by roughly the same amount towards zero |
| **Ridge** | \(\lambda_j = 1\): common scale, half-Cauchy \(\tau\) | Shrinks all coefficients by the same proportion: best when there are many small effects |

All three are sampled by Gibbs samplers whose conditional distributions are exact (Makalic and Schmidt 2016 for the horseshoe), and the regressors are standardised so that one prior suits every variable. The engine reports for each coefficient the posterior mean and interval, the **probability that the effect is non-negligible** (a one-standard-deviation change in the regressor moves the outcome by more than 5% of its standard deviation; you can change the 5%) and the **shrinkage factor** \(\kappa_j = 1/(1 + n\tau^2\lambda_j^2)\), which is 1 when the coefficient is pulled all the way to zero and 0 when the data are left alone (Piironen and Vehtari 2017).

<div class="read" markdown="1">
**Shrinkage is not selection**

A coefficient that is shrunk is not set to zero. The posterior reports how much is left and how uncertain it is. If you need a list of variables, use the probability of a non-negligible effect or the interval, and say which rule you used; if you need a list of variables with a stated probability of belonging, use the model-averaging guide.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate the hedonic price equation with 44 regressors (eight characteristics, their squares and all pairwise products, each standardised) on a random sample of 120 tracts, with a horseshoe prior on the coefficients (Carvalho et al. 2010): each coefficient has a normal prior with standard deviation \(\sigma\tau\lambda_j\), and \(\lambda_j\) and \(\tau\) have half-Cauchy priors. The error variance has the usual improper prior. We sample the posterior by Gibbs sampling with the scale-mixture representation of Makalic and Schmidt (2016), in two chains of 2,000 draws after 1,000 burn-in draws, and compare with the Bayesian lasso, a ridge prior and OLS.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes

df = pd.read_csv("hprice2_poly.csv")
x = [c for c in df.columns if c != "lprice"]
base = x[:8]
print(df.shape, "-> 8 regressors, 8 squares, 28 products:", len(x), "regressors")
train = df.sample(120, random_state=7)
test = df.drop(train.index)
data = train.to_csv(index=False)
print("training sample:", len(train), " test sample:", len(test))
print(train[base + ["lprice"]].describe().round(2).loc[["mean", "std"]].T)
```

<div class="read" markdown="1">
**How to read it**

The file has 506 tracts and 44 regressors already built: the eight characteristics, centred and divided by their standard deviations, their eight squares and the 28 products of pairs. In a random training sample of 120 tracts, the regressors have means near zero and standard deviations near one, as they should; log price has mean 9.92 and standard deviation 0.39. The other 386 tracts are held out to test the predictions in Step 8. Standardisation matters for shrinkage: a prior of one scale only treats all coefficients alike if the regressors have the same scale (the engine does this itself, and reports the coefficients in the original units).
</div>

## Estimate

```python
spec = dict(y="lprice", x=x)
r = oes.run("bayes_shrink", data, spec)
oes.summary(r)
oes.show(r, "Posterior of the coefficients", rows=12)
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The model explains 88.4% of the variance of log price in the training sample, with a residual standard deviation of 0.150 (log points) against 0.39 for log price itself. The table lists the 12 coefficients with the largest standardised posterior means. Three stand out: the share of lower-status residents (\(-0.206\), 95% interval \(-0.279\) to \(-0.135\)), its interaction with the property tax (\(-0.154\), \(-0.257\) to \(-0.066\)) and the number of rooms (0.109, 0.038 to 0.177). A one-standard-deviation increase in the lower-status share lowers the price by about 21 log points, other things at their means. Their shrinkage factors are 0.010, 0.019 and 0.043, so the prior left them almost untouched. The next nine have shrinkage between 0.15 and 0.31, intervals that include zero, and posterior probabilities of a non-negligible effect between 0.51 and 0.79.

The first chart shows the 20 largest standardised coefficients, posterior mean against OLS (OLS is far larger for almost all of them), and the second their shrinkage, which rises from near 0 for the three clear effects to between 0.15 and 0.35 for the rest. The "expected number of non-negligible coefficients" in the summary (20.3) is the sum of the 44 probabilities; it is large because many coefficients have probabilities near one half, which means *undecided*, not *present*.
</div>

## Which effects are real?

```python
co = oes.frame(next(t for t in r["tables"] if t["title"].startswith("Posterior of the coefficients")))
print("intervals that exclude zero:", (co["Interval excludes 0"] == "yes").sum(), "of", len(co))
print("P(non-negligible) above 0.9:", (co["P(non-negligible)"].astype(float) > 0.9).sum())
print(co.loc[base, ["Mean", "2.5%", "97.5%", "P(non-negligible)", "Shrinkage"]])
```

<div class="read" markdown="1">
**How to read it**

Three of the 44 intervals exclude zero, and the same three have a probability above 0.9 of being non-negligible. Look at the eight linear terms. Rooms (0.109) and the lower-status share (\(-0.206\)) are clear. The other six are undecided: pollution \(-0.021\) (interval \(-0.111\) to 0.024), distance \(-0.034\), highway access 0.024, property tax \(-0.027\), the pupil-teacher ratio \(-0.018\) and crime \(-0.049\), with probabilities of a non-negligible effect between 0.43 and 0.58 and shrinkage between 0.28 and 0.37. The data do not say these effects are zero; with 120 tracts they cannot tell. A paper should say "not distinguishable from zero in this sample", not "no effect". With all 506 tracts the picture would be sharper.
</div>

## Compare with OLS

```python
oes.show(r, "Ordinary least squares", rows=8)
```

<div class="read" markdown="1">
**How to read it**

With 44 regressors and 120 observations, OLS is unstable. The standard errors on highway access (0.264) and the property tax (0.312) are five to eight times the posterior standard deviations (0.047 and 0.041), and the coefficients are large and erratic: crime \(-0.205\) against a posterior mean of \(-0.049\), pollution \(-0.105\) against \(-0.021\), distance \(-0.135\) against \(-0.034\). The ratio column shows the share of the OLS coefficient that survives: about 0.2 to 0.27 for those four, so OLS overstates them by a factor of four or five, which is the overfitting that the prior corrects. The two clear effects are not shrunk much: the lower-status share (ratio 1.08) and rooms (1.36; the posterior is larger than OLS, because OLS splits the rooms effect over correlated interaction terms that the prior shrinks). Ignore the ratios for the property tax (7.4) and the pupil-teacher ratio (\(-12.5\)): the OLS coefficients are almost exactly zero (\(-0.004\) and 0.001), so a ratio is meaningless.
</div>

## Check the sampler

```python
oes.show(r, "MCMC convergence diagnostics")
```

<div class="read" markdown="1">
**How to read it**

The table lists the ten parameters with the smallest effective sample sizes out of 45. The worst is 280 (the interaction of rooms and highway access) out of 4,000 draws, and the largest R-hat in the whole run is 1.011, so the two chains agree. A few hundred effective draws are enough for means and standard deviations but rather few for the tails of a 95% interval: the horseshoe prior has heavy tails and mixes more slowly than a normal prior. For a paper, use 10,000 draws per chain, which takes a few seconds. If R-hat exceeded 1.05, or the effective sample size fell below about 100 for a parameter you report, increase `draws` and `burn` before trusting the intervals.
</div>

## Robustness: the prior

```python
oes.show(r, "Prior sensitivity")
```

<div class="read" markdown="1">
**How to read it**

Each row re-estimates the model under a different prior, and the columns are the eight largest coefficients. The three leading terms keep their size and sign whatever the prior: the lower-status share is \(-0.204\) (horseshoe), \(-0.189\) (lasso) and \(-0.170\) (ridge), and rooms is 0.111, 0.114 and 0.118. The horseshoe leaves the large coefficient closest to its OLS value (\(-0.191\)), and ridge shrinks it most, as expected: ridge applies the same proportional shrinkage to everything. For the small coefficients the three priors agree and all disagree with OLS: crime is \(-0.071\), \(-0.065\) and \(-0.066\) against \(-0.205\) in OLS, and the property-tax interaction with crime and the others are halved or more. The conclusions about which effects matter do not depend on the prior; their size depends a little.
</div>

## Diagnostics: does shrinkage predict better?

Estimate on 120 random tracts, predict the other 386, and repeat for ten random training samples. The measure is the root mean squared error of log price.

```python
import statsmodels.api as sm
rows = []
for s in range(10):
    tr = df.sample(120, random_state=100 + s); te = df.drop(tr.index); d_ = tr.to_csv(index=False); row = []
    for pr in ("horseshoe", "lasso", "ridge"):
        rr = oes.run("bayes_shrink", d_, dict(spec, prior=pr, draws=1000, burn=500))
        b = oes.frame(next(t for t in rr["tables"] if t["title"].startswith("Posterior of the coefficients")))["Mean"].reindex(x).astype(float)
        pred = tr["lprice"].mean() + (te[x] - tr[x].mean()).to_numpy() @ b.to_numpy()
        row.append(np.sqrt(((te["lprice"] - pred) ** 2).mean()))
    for cols in (x, base):
        fit = sm.OLS(tr["lprice"], sm.add_constant(tr[cols])).fit()
        row.append(np.sqrt(((te["lprice"] - fit.predict(sm.add_constant(te[cols], has_constant="add"))) ** 2).mean()))
    row.append(np.sqrt(((te["lprice"] - tr["lprice"].mean()) ** 2).mean()))
    rows.append(row)
out = pd.DataFrame(rows, columns=["Horseshoe", "Lasso", "Ridge", "OLS, 44", "OLS, 8 linear", "Mean only"])
print(out.mean().round(4))
print("horseshoe beats OLS (44):", (out["Horseshoe"] < out["OLS, 44"]).mean(), " lasso:", (out["Horseshoe"] < out["Lasso"]).mean(), " ridge:", (out["Horseshoe"] < out["Ridge"]).mean())
```

<div class="read" markdown="1">
**How to read it**

The three shrinkage priors predict about 15% better than OLS: root mean squared errors of 0.183 (horseshoe), 0.180 (lasso) and 0.180 (ridge), against 0.218 for OLS with all 44 regressors and 0.210 for OLS with only the eight linear terms. Using all 44 regressors without shrinkage is *worse* than using the eight (0.218 against 0.210): extra terms hurt when they are not shrunk, and help when they are. The horseshoe beats OLS in all ten samples. It does **not** beat the lasso or the ridge here (it wins 1 and 2 of ten): the signal in this problem is dense, with many small effects and only a few large ones, and the horseshoe's advantage is in sparse problems where nearly all coefficients are zero. Choose the prior from the structure you expect, and report the comparison.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Table 1 reports the posterior of the hedonic price equation with 44 regressors, estimated on 120 randomly drawn tracts with a horseshoe prior. Three terms are clearly different from zero: the share of lower-status residents (posterior mean \(-0.206\) per standard deviation, 95% credible interval \(-0.279\) to \(-0.135\)), its interaction with the property tax (\(-0.154\); \(-0.257\) to \(-0.066\)) and the number of rooms (0.109; 0.038 to 0.177). The other 41 coefficients have intervals that include zero and posterior probabilities of a non-negligible effect between 0.4 and 0.8, so the sample cannot tell whether they are zero. OLS on the same sample gives coefficients that are four to five times larger for pollution, distance, access and crime, with standard errors up to eight times larger. The chains converge (R-hat at most 1.011, smallest effective sample size 280). The three leading effects are the same under a Bayesian lasso and a ridge prior. In ten random splits, the shrinkage priors predict log prices of the 386 held-out tracts with a root mean squared error of 0.18 against 0.22 for OLS with all regressors and 0.21 for OLS with the linear terms only.
</div>

| Report | What | Where |
|---|---|---|
| Prior | Family, hyperpriors, standardisation, the negligible-effect rule | Step 1 |
| Results | Posterior mean, interval, probability of a non-negligible effect, shrinkage | Steps 3 and 4 |
| Comparison | OLS coefficients and standard errors on the same sample | Step 5 |
| Sampler | Chains, draws, R-hat, effective sample size | Step 6 |
| Robustness | The three priors side by side | Step 7 |
| Prediction | Out-of-sample root mean squared error over repeated splits | Step 8 |

**References.** Carvalho, Polson and Scott (2010), *Biometrika*; Makalic and Schmidt (2016), *IEEE Signal Processing Letters*; Park and Casella (2008), *Journal of the American Statistical Association*; Piironen and Vehtari (2017), *Electronic Journal of Statistics*; Bhadra, Datta, Polson and Willard (2019), *Statistical Science*; Harrison and Rubinfeld (1978), *Journal of Environmental Economics and Management*.
