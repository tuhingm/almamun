title: Cluster-Robust Inference and the Wild Bootstrap
short: Wild cluster bootstrap
when: Errors are correlated within groups, and with few groups the usual cluster-robust test is unreliable
data: Petersen's (2009) simulated panel of 500 firms over 10 years, used to compare standard errors; plus a simulation with few clusters
tools: Python · OES engine · NumPy
oes: wildboot
lead: Use this guide when observations come in groups (firms, schools, villages) and the errors within a group are correlated. Conventional standard errors are then too small. Cluster-robust standard errors (Liang and Zeger 1986) fix this when there are many groups; with few groups the wild cluster bootstrap-t (Cameron, Gelbach and Miller 2008) is more reliable. The guide covers the problem, the three kinds of inference, confidence intervals, a simulation of size with few clusters and the write-up.
---

## The question and the model

**How much do we overstate precision if we ignore the grouping, and how do we test with few groups?** In a regression \(y_{gi} = x_{gi}'\beta + u_{gi}\) with \(u\) correlated within group \(g\), the variance of \(\hat\beta\) is estimated by summing the scores within each group, \(\hat V = (X'X)^{-1}\bigl(\sum_g X_g' \hat u_g \hat u_g' X_g\bigr)(X'X)^{-1}\), with a small-sample factor (CR1). This is valid when the number of groups is large. The wild cluster bootstrap imposes the null, multiplies the residuals of each group by a random sign (or a Webb weight, which has six points and suits fewer than 12 groups) and uses the distribution of the t statistic.

<div class="write" markdown="1">
**How to write it: the model**

We regress y on x in Petersen's (2009) panel of 500 firms and 10 years and report conventional, heteroskedasticity-robust and cluster-robust (CR1) standard errors and a wild cluster bootstrap-t p-value that imposes the null. We then draw a simulation with 8 clusters and no true effect to compare the rejection rates of the tests at the 5 per cent level.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Three kinds of inference

```python
@@EST@@
```

@@READ_EST@@

## Confidence intervals and clusters

```python
@@CI@@
```

@@READ_CI@@

## Robustness: a different null and weights

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: size with few clusters

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Liang and Zeger (1986), *Biometrika*; Cameron, Gelbach and Miller (2008), *Review of Economics and Statistics*; Webb (2014); Petersen (2009), *Review of Financial Studies*; Cameron and Miller (2015), *Journal of Human Resources*; MacKinnon and Webb (2017), *Journal of Applied Econometrics*.
