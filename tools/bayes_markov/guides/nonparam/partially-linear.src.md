title: Partially Linear Model
short: Partially linear model
when: You want the effect of some regressors in a linear way while letting one control enter in an unknown, smooth way
data: Food spending of 1,655 Canadian single-person households (Engel curve; Ecdat/np, Engel95); food share on the number of kids and log wages, with log expenditure entering freely
tools: Python · OES engine · NumPy
oes: plm
lead: Use this guide when the effect of one variable is well described by a coefficient and another variable needs a flexible shape. The partially linear model (Robinson 1988) removes the unknown function by subtracting kernel regressions on the control from both sides and runs least squares on the residuals. The guide covers the estimator, the shape of the unknown function, the comparison with a linear control, the bias when the shape matters, a simulation and the write-up.
---

## The question and the model

**What is the effect of x on y when another variable z has an unknown effect?** The model is

\[
y_i = x_i'\beta + g(z_i) + \varepsilon_i ,
\]

with \(g\) unknown and smooth. Taking the expectation given \(z\) and subtracting gives \(y - E(y|z) = (x - E(x|z))'\beta + \varepsilon\). Robinson (1988) replaces the two conditional means by kernel regressions and applies least squares to the residuals; \(\hat\beta\) is root-n consistent and has the usual standard errors. The function \(g\) is then estimated from the remaining variation.

<div class="write" markdown="1">
**How to write it: the model**

We estimate a partially linear model of food spending on the number of kids and log wages, with log total expenditure entering through an unknown function. Following Robinson (1988) we remove the function by kernel regression with a cross-validated bandwidth and estimate the coefficients by least squares on the residuals. We plot the estimated function, compare the fit with linear and polynomial controls, check the coefficients against different bandwidths, and show in a simulation that a linear control biases the coefficient when the true function is curved.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the model

```python
@@EST@@
```

@@READ_EST@@

## Is the control linear?

```python
@@CURVE@@
```

@@READ_CURVE@@

## Robustness: the bandwidth

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: bias from a linear control

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Robinson (1988), *Econometrica*; Engle, Granger, Rice and Weiss (1986), *Journal of the American Statistical Association*; Yatchew (1997), *Economics Letters*; Delgado and Mora (1998), *Journal of Applied Econometrics* (the Engel95 data).
