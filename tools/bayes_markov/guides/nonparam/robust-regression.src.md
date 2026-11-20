title: Robust Regression
short: Robust regression
when: A few observations may be outliers or high-leverage points and you want estimates that they cannot move
data: Record times of 35 Scottish hill races against distance and climb (Atkinson 1986; MASS, hills)
tools: Python · OES engine · NumPy
oes: robreg
lead: Use this guide when least squares may be driven by a few unusual observations. It compares ordinary least squares with a Huber M estimator, a bisquare M estimator and an MM estimator (Yohai 1987) that combines a high breakdown point with high efficiency. The guide covers the estimators, the weights that identify the outliers, leverage, a simulation with contamination and the write-up.
---

## The question and the model

**Are the estimates driven by a few observations, and what are they without them?** Least squares minimises squared residuals, so one large residual can dominate. An M estimator minimises \(\sum \rho(e_i/s)\) with a \(\rho\) that grows more slowly: Huber's is quadratic near zero and linear beyond a cut-off, and Tukey's bisquare gives zero weight to very large residuals. These are not robust to points that are far out in \(x\) (leverage). The MM estimator starts from an S estimate of scale with a 50% breakdown point and then takes an efficient M step. Observations with a robust residual above 2.5 scales are flagged.

<div class="write" markdown="1">
**How to write it: the model**

We regress race time on distance and climb with ordinary least squares, a Huber M estimator, a bisquare M estimator and an MM estimator (Yohai 1987). We compare the coefficients and the residual scale, list the observations with the largest robust residuals and the highest leverage, and check the estimators against a simulation in which 20 per cent of the data are bad leverage points.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate

```python
@@EST@@
```

@@READ_EST@@

## Which observations are unusual?

```python
@@OUT@@
```

@@READ_OUT@@

## Robustness: dropping the flagged races

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: contamination in a simulation

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Huber (1964), *Annals of Mathematical Statistics*; Rousseeuw and Yohai (1984); Yohai (1987), *Annals of Statistics*; Maronna, Martin and Yohai (2006), *Robust Statistics*; Atkinson (1986), *Journal of the Royal Statistical Society C*.
