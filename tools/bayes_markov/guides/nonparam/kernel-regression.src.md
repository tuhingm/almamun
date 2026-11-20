title: Kernel Regression
short: Kernel regression
when: You do not know the shape of the relation between an outcome and a regressor and want to estimate it without assuming a functional form
data: Log wages and age of 205 Canadian workers with common education (Pagan and Ullah 1999; Ecdat/np, cps71)
tools: Python · OES engine · NumPy
oes: kernreg
lead: Use this guide to let the data draw the regression function. Local-linear kernel regression (Nadaraya 1964; Watson 1964; Fan and Gijbels 1996) fits a straight line in a window around every point and joins the fits. The guide covers the estimator, the choice of the bandwidth by leave-one-out cross-validation, a test of the linear model, sensitivity to the bandwidth, a simulation and the write-up.
---

## The question and the model

**What is the shape of E(y | x) when we do not want to assume it?** The local-linear estimator at a point \(x_0\) solves a weighted least-squares problem, with weights that fall with the distance from \(x_0\):

\[
\hat m(x_0) = \hat a,\qquad (\hat a,\hat b) = \arg\min \sum_i K\!\left(\tfrac{x_i - x_0}{h}\right)\bigl(y_i - a - b(x_i - x_0)\bigr)^2 .
\]

The bandwidth \(h\) controls the trade-off: a small \(h\) follows the noise, a large \(h\) approaches a straight line. We choose it by minimising the leave-one-out prediction error, which has a closed form for a linear smoother. A wild bootstrap compares the straight-line model with the kernel fit (Härdle and Mammen 1993).

<div class="write" markdown="1">
**How to write it: the model**

We estimate the relation between log wages and age by local-linear kernel regression with a Gaussian kernel. The bandwidth is chosen by leave-one-out cross-validation. We compare the cross-validated error with polynomials of degree one to four, test the linear specification with a wild bootstrap, check the sensitivity to the bandwidth and to local-constant fitting, and check the method in a simulation with a known function.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the regression function

```python
@@EST@@
```

@@READ_EST@@

## The estimated curve

```python
@@CURVE@@
```

@@READ_CURVE@@

## Robustness: bandwidth and degree

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: a simulation with a known function

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Nadaraya (1964), *Theory of Probability and its Applications*; Watson (1964), *Sankhya*; Fan and Gijbels (1996), *Local Polynomial Modelling and its Applications*; Härdle and Mammen (1993), *Annals of Statistics*; Pagan and Ullah (1999), *Nonparametric Econometrics*.
