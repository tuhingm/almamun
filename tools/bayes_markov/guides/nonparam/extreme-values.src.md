title: Extreme Value Theory
short: Extreme value theory
when: You care about the far tail of a distribution (large losses, floods, claims) and need probabilities and levels beyond most of the data
data: 2,167 Danish fire insurance claims, 1980-1990, in millions of Danish krone (McNeil 1997; evir, danish)
tools: Python · OES engine · SciPy
oes: evt
lead: Use this guide to model the tail of a distribution with a peaks-over-threshold approach. Above a high threshold the excesses follow a generalised Pareto distribution (Pickands 1975; Balkema and de Haan 1974). The guide covers the model, the choice of the threshold, the Hill estimator, return levels, value at risk and expected shortfall, a goodness-of-fit test, a simulation of the estimates and the write-up.
---

## The question and the model

**How large can the losses be, and how often?** For a high threshold \(u\) the excesses \(y = x - u\) are approximately generalised Pareto with shape \(\xi\) and scale \(\sigma\), \(P(y > t) = (1 + \xi t/\sigma)^{-1/\xi}\). A positive \(\xi\) means a heavy tail: moments of order above \(1/\xi\) do not exist. With \(\zeta\) the share of observations above \(u\), the level exceeded once in \(m\) observations is \(u + (\sigma/\xi)\bigl((m\zeta)^{\xi} - 1\bigr)\). The 99% value at risk is the 1-in-100 level and expected shortfall is the mean loss beyond it. The threshold trades bias (too low) against variance (too high).

<div class="write" markdown="1">
**How to write it: the model**

We fit a generalised Pareto distribution to the excesses of Danish fire claims over a threshold of 10 million krone by maximum likelihood. We check the stability of the shape across thresholds, compare it with the Hill estimator of the tail index, and report return levels with delta-method intervals, value at risk and expected shortfall at 99 and 99.5 per cent. We test the fit and check the estimator in a simulation from a known generalised Pareto distribution.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Fit the tail

```python
@@EST@@
```

@@READ_EST@@

## Return levels, value at risk and expected shortfall

```python
@@RISK@@
```

@@READ_RISK@@

## Robustness: the threshold

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: a simulation with a known tail

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Pickands (1975), *Annals of Statistics*; Balkema and de Haan (1974), *Annals of Probability*; Hill (1975), *Annals of Statistics*; McNeil (1997), *ASTIN Bulletin*; Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*; McNeil, Frey and Embrechts (2015), *Quantitative Risk Management*.
