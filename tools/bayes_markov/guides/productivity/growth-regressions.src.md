title: Growth Regressions: Solow and Convergence
short: Growth regressions
when: You have cross-country (or cross-region) data on income, investment, population growth and schooling and want to test the Solow model and the speed of convergence
data: Income per working-age person in 1960 and 1985, investment rate, population growth and schooling for 121 countries (Durlauf and Johnson 1995; Ecdat, GrowthDJ); the Mankiw-Romer-Weil non-oil sample has 98 countries
tools: Python · OES engine · NumPy
oes: growth
lead: Use this guide to test whether the Solow growth model explains differences in income across countries and whether poor countries catch up. The guide follows Mankiw, Romer and Weil (1992): the level regression with the restriction that the coefficients on saving and on population growth are equal and opposite, the implied capital share, the augmented model with human capital, and the regression of growth on initial income for unconditional and conditional convergence. It covers robustness to the sample, a simulation with a known capital share and the write-up.
---

## The question and the model

**Why are some countries rich, and do poor countries grow faster?** In the Solow model, steady-state income per worker depends on the saving rate \(s\) and on \(n + g + \delta\), the sum of population growth, technical progress and depreciation:

\[
\ln y = a + \frac{\alpha}{1-\alpha}\ln s - \frac{\alpha}{1-\alpha}\ln(n+g+\delta) .
\]

The coefficients on \(\ln s\) and \(\ln(n+g+\delta)\) are equal and opposite, and the ratio \(\alpha/(1-\alpha)\) identifies the capital share \(\alpha\). Mankiw, Romer and Weil (1992) add human capital \(h\) (schooling), which adds \(\frac{\beta}{1-\alpha-\beta}\ln h\). Convergence is a statement about growth: with speed \(\lambda\), \(\ln y_T - \ln y_0 = -(1 - e^{-\lambda T})\ln y_0 + \dots\); it is *unconditional* if growth falls with initial income alone, *conditional* if it does so once \(s\), \(n\) and \(h\) are held constant. As usual \(g + \delta = 0.05\).

<div class="write" markdown="1">
**How to write it: the model**

We estimate the Mankiw-Romer-Weil model on 98 non-oil countries: the logarithm of 1985 income per working-age person on the logarithms of the investment rate, of \(n + 0.05\) and of the schooling rate, with robust standard errors; we test the Solow restriction that the first two coefficients sum to zero and compute the implied capital share. We then regress growth from 1960 to 1985 on initial income, alone and with the three controls, and report the speed of convergence and the half-life.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## The Solow regression in levels

```python
@@EST@@
```

@@READ_EST@@

## Convergence

```python
@@CONV@@
```

@@READ_CONV@@

## Robustness: samples and specification

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: does the test recover a known capital share?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Solow (1956), *Quarterly Journal of Economics*; Mankiw, Romer and Weil (1992), *Quarterly Journal of Economics*; Barro and Sala-i-Martin (1992), *Journal of Political Economy*; Durlauf and Johnson (1995), *Journal of Applied Econometrics*; Durlauf, Johnson and Temple (2005), *Handbook of Economic Growth*.
