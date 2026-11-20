title: Explosive Bubble Tests (SADF and GSADF)
short: Bubble tests
when: You want to test whether a price or a price ratio has episodes of explosive growth faster than a random walk, and date the start and end of each episode
data: Monthly S&P 500 price and dividends, January 1871 to June 2023 (Shiller); 1,830 months
tools: Python · OES engine · NumPy
oes: bubble
lead: Use this guide to test for bubbles in asset prices and to date them. The right-tailed unit root tests of Phillips, Wu and Yu (2011) and Phillips, Shi and Yu (2015) ask whether a series has a mildly explosive root in some part of the sample, and the backward SADF sequence says when. The guide covers the tests, the simulated critical values, the dated episodes, what a ratio to dividends adds, robustness to the minimum window, a simulation of the size and the dating accuracy, and the write-up.
---

## The question and the model

**Did prices ever grow faster than a random walk?** For a series \(y_t\) the ADF regression is

\[
\Delta y_t = a + \rho\, y_{t-1} + u_t ,
\]

and a unit root (\(\rho = 0\)) is tested against the **right-tail** alternative \(\rho > 0\): an explosive root, growth that feeds on itself. Three statistics use different windows:

| Statistic | Windows |
|---|---|
| **ADF** | The whole sample |
| **SADF** (Phillips, Wu and Yu 2011) | Sup of the ADF over windows that start at the first observation and end at \(r_2\) |
| **GSADF** (Phillips, Shi and Yu 2015) | Sup over all windows \([r_1, r_2]\) with at least \(r_0 = T(0.01 + 1.8/\sqrt T)\) observations: it can find several bubbles |

The **backward SADF** at \(r_2\), \(\text{BSADF}(r_2) = \sup_{r_1}\text{ADF}_{r_1}^{r_2}\), is compared with its critical value at every date to **date-stamp** the episodes: an episode starts when BSADF crosses above its critical value and ends when it falls below. Critical values are simulated from random walks (with a drift that vanishes with the sample size) of the same length, so they depend on the sample and the minimum window.

A rise in the price can reflect rising fundamentals. The **price-dividend ratio** holds dividends fixed, so an explosive ratio is harder to attribute to fundamentals.

<div class="write" markdown="1">
**How to write it: the model**

We test for explosive behaviour in the monthly price-dividend ratio of the S&P 500 with the SADF and GSADF tests of Phillips, Wu and Yu (2011) and Phillips, Shi and Yu (2015). The ADF regression has a constant and no lags, the minimum window is the rule \(T(0.01 + 1.8/\sqrt T)\), and critical values come from 200 simulated random walks with a vanishing drift of the same length. Episodes are dated where the backward SADF exceeds its 95% critical value for at least log(T) consecutive months.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## The tests

```python
@@EST@@
```

@@READ_EST@@

## Date the episodes

```python
@@DATE@@
```

@@READ_DATE@@

## Robustness: the minimum window and the price alone

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: size and dating accuracy

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Phillips, Wu and Yu (2011), *International Economic Review*; Phillips, Shi and Yu (2015), *International Economic Review*; Phillips and Shi (2018), *Peking University Working Paper*; Homm and Breitung (2012), *Journal of Financial Econometrics*; Shiller (2000), *Irrational Exuberance*, Princeton University Press; Evans (1991), *American Economic Review*.
