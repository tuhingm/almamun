title: Bayesian VAR with Minnesota and Normal-Wishart Priors
short: Bayesian VAR
when: A VAR with many coefficients and a short sample
data: U.S. quarterly GDP growth, inflation and the 3-month T-bill rate, 1959–2009
tools: Python · OES engine · statsmodels
oes: bvar
lead: Use this guide when a VAR has more coefficients than the data can pin down, which is almost always the case with four lags of three or more quarterly variables. A Bayesian VAR shrinks the coefficients towards a simple benchmark, gives sharper impulse responses and better forecasts, and reports its uncertainty directly. The guide covers the choice of prior, lag length and tightness, estimation, diagnostics, robustness and the write-up.
---

## The question and the model

**What happens to output and inflation after an unexpected rise in the short-term interest rate?** This is the classic question of the monetary VAR. The model is a VAR(\(p\)) in three variables, quarterly GDP growth, CPI inflation and the 3-month Treasury bill rate:

\[
y_t = c + A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t, \qquad u_t \sim N(0, \Sigma).
\]

With \(p = 4\) lags each equation has \(1 + 3 \times 4 = 13\) coefficients, 39 in all, plus 6 in \(\Sigma\). Fifty years of quarterly data give about 200 observations: enough to estimate the coefficients, but not to estimate them *precisely*. Unrestricted OLS fits the noise, impulse responses come out jagged, and forecasts are poor.

A **Bayesian VAR** adds a prior that pulls every coefficient towards a sensible benchmark. The **Minnesota prior** (Litterman 1986) says that each variable behaves like its own random walk or white noise, that recent lags matter more than distant ones, and that other variables' lags matter less than a variable's own. The data then move the coefficients away from this benchmark only as far as they have evidence for.

| Term | Meaning | Default here |
|---|---|---|
| \(\delta_i\) | prior mean of variable \(i\)'s own first lag | 1 for an I(1) series in levels, 0 for a stationary one |
| \(\lambda_1\) | overall tightness: how strongly all coefficients are pulled to the prior | 0.2, then chosen by the data |
| \(\lambda_2\) | extra tightness on the lags of *other* variables | 0.5 |
| \(\lambda_3\) | how fast the prior tightens with the lag length (\(1/l^{\lambda_3}\)) | 1 |
| \(\lambda_4\) | tightness on the constant | 100 (essentially flat) |

<div class="read" markdown="1">
**Four priors, one idea**

The OES engine offers the four priors of the ECB's BEAR toolbox. They share the Minnesota means and scales and differ in how they treat \(\Sigma\):

- **Minnesota (Litterman):** \(\Sigma\) is fixed at the residual variances of univariate AR models. Exact, fast, equation by equation. Good for forecasting.
- **Normal-Wishart:** \(\Sigma\) is estimated with an inverse-Wishart prior, and the posterior and the *marginal likelihood* are analytic, so lag length and tightness can be chosen by the data. It forces \(\lambda_2 = 1\).
- **Independent Normal-Wishart:** frees \(\lambda_2\) and allows block exogeneity; estimated by Gibbs sampling.
- **Normal-diffuse:** Minnesota prior on the coefficients, no information on \(\Sigma\); Gibbs sampling.

Start with Normal-Wishart to choose the lags and \(\lambda_1\), then report the prior you prefer and show the others as robustness.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a Bayesian VAR in real GDP growth, CPI inflation and the 3-month Treasury bill rate, \(y_t = c + \sum_{l=1}^{p} A_l y_{t-l} + u_t\), \(u_t \sim N(0,\Sigma)\), on quarterly U.S. data for 1959Q3–2009Q3. Following Litterman (1986) and Kadiyala and Karlsson (1997), we use a Normal-Wishart prior with Minnesota moments: the prior mean of each variable's own first lag is one for the interest rate, which is persistent, and zero for growth and inflation; all other coefficients have prior mean zero, with prior standard deviations that shrink with the lag length.
</div>

## Load the data and test for unit roots

The prior mean of the own first lag depends on whether a series is persistent. Test first.

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("us_macro.csv").read()
df = pd.read_csv("us_macro.csv")
print(df[["date", "gdp_growth", "inflation", "tbilrate"]].describe().round(2).loc[["count", "mean", "std", "min", "max"]])

spec = dict(vars=["gdp_growth", "inflation", "tbilrate"], time="date", lags=4, prior="nw", lambda1=0.2,
            delta="0, 0, 1", sens_shock="tbilrate", sens_response="gdp_growth")
r = oes.run("bvar", data, spec)
oes.show(r, "Unit-root")
```

<div class="read" markdown="1">
**How to read it**

GDP growth and inflation reject a unit root in levels (ADF p-values below 0.05) and are treated as stationary: prior mean 0 on their own first lag. The T-bill rate does not reject; it is stationary only after differencing, so it gets a random-walk prior (prior mean 1). Notice that nothing is differenced: a Bayesian VAR is estimated in levels, and the prior handles the persistence. Sims, Stock and Watson (1990) show that inference on impulse responses remains valid in levels even with unit roots or cointegration.
</div>

## Choose the lag length and the tightness

The Normal-Wishart prior gives the **marginal likelihood**, the probability of the data under the model with the coefficients integrated out. It rewards fit and penalises complexity automatically, so it can choose both the number of lags and \(\lambda_1\) (Giannone, Lenza and Primiceri 2015).

```python
oes.show(r, "Lag order")
oes.show(r, "Overall tightness")
```

<div class="read" markdown="1">
**How to read it**

- **Lags.** The marginal likelihood rises quickly up to three or four lags and then flattens: its maximum is at seven lags, but the gain beyond four is about two log points, small next to the ten points gained from one to two lags. Extra lags add little once the prior has shrunk them. BIC, which has no prior to shrink with, prefers fewer lags. With quarterly data four lags is the convention, and the marginal likelihood supports it; more lags would be harmless under the prior.
- **Tightness.** The marginal likelihood peaks at \(\lambda_1 = 0.2\): looser priors overfit, tighter ones ignore the data. This is the value used below. Always report how \(\lambda_1\) was chosen; a value picked to deliver a result is a red flag for referees.
</div>

## Estimate the BVAR

```python
oes.summary(r)
oes.show(r, "Equation: tbilrate")
```

<div class="read" markdown="1">
**How to read it**

Each row is a posterior: its mean, standard deviation, 95% credible interval and the posterior probability that the coefficient is positive. The interest-rate equation is a monetary policy rule in reduced form: the T-bill rate is highly persistent (own first lag about 0.93) and rises after strong GDP growth and higher inflation, with posterior probabilities above 0.9 for several of these lags. "Distance from OLS" in the summary shows how much the prior moved the coefficients: about half, which is typical for a VAR of this size.

A credible interval is read literally: given the model and the prior, there is a 95% probability that the coefficient lies in it. Bayesian results do not need p-values, but P(> 0) close to 0 or 1 plays the same role.
</div>

## Impulse responses: a monetary policy shock

The policy shock is identified recursively (Cholesky), with the interest rate ordered **last**: the Fed reacts within the quarter to output and inflation, while output and prices respond to the interest rate only with a lag. This is the standard ordering of Christiano, Eichenbaum and Evans (1999).

```python
fig = oes.plot(r, "Response of gdp_growth to a shock in tbilrate")
fig = oes.plot(r, "Response of inflation to a shock in tbilrate")
```

<div class="read" markdown="1">
**How to read it**

After a one-standard-deviation rise in the T-bill rate (about 0.6 percentage point), GDP growth ticks up in the first quarter and then falls by about 0.15 percentage point in the second and third quarters, where the 68% credible band is below zero; after that the response stays negative but the band includes zero. Inflation, however, **rises**, by about 0.25 point, and returns only slowly towards zero: the well-known *price puzzle* (Sims 1992). It usually means the VAR omits information the Fed uses to forecast inflation, such as commodity prices. Adding many indicators through factors, as in a FAVAR (see the FAVAR guide), is the usual remedy when you have a large panel of series. Report the puzzle honestly rather than hiding it; it is a known limitation of small monetary VARs.

Bayesian VAR papers report 68% bands (16th to 84th percentiles) by convention, as here; add 90% bands if referees ask.
</div>

## How important is the shock? Variance decomposition

```python
oes.show(r, "Forecast-error variance")
```

<div class="read" markdown="1">
**How to read it**

Monetary policy shocks explain only a few percent of the forecast-error variance of GDP growth and inflation, even after five years. Most movements in output and inflation are due to other shocks; the interest rate itself is driven increasingly by output and inflation, as the policy rule implies. This is the standard result: policy matters for *how* the economy responds, but surprises in policy are a small source of fluctuations.
</div>

## Diagnostics: residuals, stability and convergence

```python
oes.show(r, "Residual diagnostics")
oes.show(r, "Stability")
```

<div class="read" markdown="1">
**How to read it**

- **Stability.** Almost every posterior draw has all roots inside the unit circle, and the largest root at the posterior mean is below one: the VAR is stationary and impulse responses die out.
- **Autocorrelation.** Ljung-Box does not reject for GDP growth and inflation but does for the T-bill rate, and the joint portmanteau test rejects. The usual remedy is more lags; re-estimate with six and check that the responses do not change.
- **Normality and ARCH.** Strongly rejected for inflation and the interest rate. The 1970s oil shocks and the Volcker disinflation produce large residuals, and their variance changes over time. The coefficient estimates remain valid, but bands that assume constant variance are too narrow in volatile periods. The SV-VAR guide models this directly; mention it as a limitation.
</div>

With the Normal-Wishart prior the posterior draws are independent, so there is nothing to converge. The Gibbs samplers (Independent Normal-Wishart, Normal-diffuse) need a check:

```python
r_inw = oes.run("bvar", data, dict(spec, prior="inw", draws=2000, burn=1000))
oes.show(r_inw, "MCMC")
```

<div class="read" markdown="1">
**How to read it**

Effective sample sizes are close to the number of draws and the Geweke statistics are small: the chain mixes well and has settled. If |z| were above 2 for several parameters, increase the burn-in.
</div>

## Robustness: prior, tightness and forecasts

A result that depends on the prior is not a result. Three checks:

```python
oes.show(r, "Prior sensitivity")
for prior in ["minnesota", "nw", "inw", "nd"]:
    rr = oes.run("bvar", data, dict(spec, prior=prior))
    irf = next(c for c in rr["charts"] if c["title"] == "Response of gdp_growth to a shock in tbilrate")["series"][0]["y"]
    print(f"{prior:10s} response of GDP growth after 4, 8, 12 quarters: " + ", ".join(f"{irf[h]:+.3f}" for h in (4, 8, 12)))
```

<div class="read" markdown="1">
**How to read it**

- **Tightness.** With tight and moderate priors (\(\lambda_1 \le 0.2\)) the output response is negative at every horizon. With very loose priors and with OLS, the response at four quarters turns *positive*: without shrinkage the VAR fits noise, and the sign is not robust. This is exactly why a Bayesian VAR is used, and the table is the evidence a referee needs.
- **Prior family.** All four priors give a fall in GDP growth of similar size. Report the Normal-Wishart results and these as robustness.
</div>

```python
oes.show(r, "Out-of-sample")
```

<div class="read" markdown="1">
**How to read it**

Recursive one- and four-quarter-ahead forecasts over the last 20% of the sample (2000–2009). The BVAR beats the OLS VAR for inflation at both horizons and is close for the other variables. A random walk is hard to beat for the interest rate at one quarter and for GDP growth at four quarters, a common finding for this period. In a forecasting paper, test the differences formally with the Diebold-Mariano test and report several evaluation windows.
</div>

## Block exogeneity

When some variables cannot be affected by the others, such as world variables in a model of a small open economy, the Independent Normal-Wishart and Normal-diffuse priors can impose it: the coefficients of the other variables in the block's equations get (almost) zero prior variance.

```python
rb = oes.run("bvar", data, dict(spec, prior="inw", block=["gdp_growth"], draws=1000, burn=500))
oes.show(rb, "Equation: gdp_growth", rows=7)
```

Here GDP growth is (for illustration) made block-exogenous: the lags of inflation and the interest rate in its equation are pushed to zero. In a real application use it for foreign variables, oil prices or anything determined outside the economy you model.

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The marginal likelihood selects an overall tightness of \(\lambda_1 = 0.2\) and changes little beyond four lags, the conventional choice for quarterly data. Figure 1 shows the responses to a one-standard-deviation monetary policy shock, identified recursively with the interest rate ordered last. GDP growth falls by about 0.15 percentage point in the second and third quarters after the shock, when the 68% credible band excludes zero, and remains below its baseline thereafter. Inflation rises by about 0.25 percentage point and returns only slowly: the price puzzle familiar from small monetary VARs (Sims 1992). Policy shocks account for less than 3% of the forecast-error variance of output growth at all horizons. The output response is negative under all four priors (Minnesota, Normal-Wishart, Independent Normal-Wishart and Normal-diffuse) and for any \(\lambda_1 \le 0.2\); with much looser priors, and with OLS, the response becomes erratic, which illustrates the value of shrinkage in a sample of this length.
</div>

| Report | What | Where |
|---|---|---|
| Data | Sources, sample, transformations, unit-root tests | Step 2 |
| Prior | Family, \(\delta_i\), \(\lambda_1\)–\(\lambda_4\), how \(\lambda_1\) and \(p\) were chosen | Steps 1 and 3 |
| Identification | Ordering and its economic justification | Step 5 |
| Results | Impulse responses with 68% (and 90%) bands, variance decomposition | Steps 5 and 6 |
| Diagnostics | Stability, residual tests, MCMC convergence for Gibbs samplers | Step 7 |
| Robustness | Other priors, other tightness, more lags, forecast comparison | Step 8 |

**References.** Litterman (1986), *Journal of Business and Economic Statistics*; Kadiyala and Karlsson (1997), *Journal of Applied Econometrics*; Sims, Stock and Watson (1990), *Econometrica*; Christiano, Eichenbaum and Evans (1999), *Handbook of Macroeconomics*; Bańbura, Giannone and Reichlin (2010), *Journal of Applied Econometrics*; Giannone, Lenza and Primiceri (2015), *Review of Economics and Statistics*; Dieppe, Legrand and van Roye (2016), *The BEAR toolbox*, ECB Working Paper 1934.
