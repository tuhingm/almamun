title: Factor-Pricing Regressions and the GRS Test
short: CAPM and factor pricing
when: You want to know whether a market factor, or several factors, explain the average returns of stocks, industries or funds, and whether any alpha is left over
data: Monthly excess returns of three U.S. industry portfolios (food, durables, construction) and the market, 1960-2002 (Verbeek 2004; Kenneth French's data library)
tools: Python · OES engine · statsmodels
oes: capm
lead: Use this guide to test an asset-pricing model on a set of returns. Each asset is regressed on the factors; the intercept, alpha, is the average return the factors do not explain, and the Gibbons-Ross-Shanken (GRS) test asks whether all the alphas are zero together. The guide covers the model, the estimation with Newey-West standard errors, the joint test, a rolling beta, a market-timing check, a sub-period check, a simulation of the test's size and power and the write-up.
---

## The question and the model

**Do the factors price the assets?** The factor model is

\[
r_{it} - r_{ft} = \alpha_i + \beta_i' f_t + \varepsilon_{it}, \qquad i = 1, \dots, N,
\]

where \(f_t\) are excess returns of the factors (for the CAPM, the market). If the factors price the assets, every \(\alpha_i\) is zero. A t-test on one alpha says little when there are many assets, so Gibbons, Ross and Shanken (1989) test all alphas at once:

\[
\text{GRS} = \frac{T - N - K}{N}\,\frac{\hat\alpha'\hat\Sigma^{-1}\hat\alpha}{1 + \hat\mu'\hat\Omega^{-1}\hat\mu} \sim F(N,\; T - N - K),
\]

where \(\hat\Sigma\) is the residual covariance, \(\hat\mu\) the mean of the factors and \(\hat\Omega\) their covariance. The denominator is the squared Sharpe ratio of the factors; the numerator measures how much the alphas would add to it.

<div class="write" markdown="1">
**How to write it: the model**

We test whether the market factor prices three industry portfolios by regressing each portfolio's monthly excess return on the market excess return. The alpha of each portfolio is tested with Newey-West standard errors, and the hypothesis that all alphas are jointly zero is tested with the Gibbons, Ross and Shanken (1989) F test, which is exact for normal homoskedastic errors.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate alphas and betas

```python
@@EST@@
```

@@READ_EST@@

## Test the alphas jointly

```python
@@GRS@@
```

@@READ_GRS@@

## Stability: rolling beta

```python
@@ROLL@@
```

@@READ_ROLL@@

## Robustness: market timing and sub-periods

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: size and power of the GRS test

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Sharpe (1964), *Journal of Finance*; Lintner (1965), *Review of Economics and Statistics*; Gibbons, Ross and Shanken (1989), *Econometrica*; Fama and French (1993), *Journal of Financial Economics*; Newey and West (1987), *Econometrica*; Treynor and Mazuy (1966), *Harvard Business Review*; Verbeek (2004), *A Guide to Modern Econometrics*, Wiley.
