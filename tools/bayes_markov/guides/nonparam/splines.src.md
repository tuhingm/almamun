title: Penalised Splines
short: Penalised splines
when: You want a smooth curve through noisy data, with the smoothness chosen from the data and a test of whether a straight line would do
data: Head acceleration in 133 simulated motorcycle crashes against time after impact (Silverman 1985; MASS, mcycle)
tools: Python · OES engine · NumPy
oes: pspline
lead: Use this guide to fit a flexible curve when the shape changes quickly in some places and not in others. A penalised spline (Eilers and Marx 1996) uses many B-spline basis functions and a penalty on the second differences of their coefficients; one smoothing parameter sets the roughness and is chosen by generalised cross-validation. The guide covers the model, the effective degrees of freedom, a test against a straight line, the comparison with polynomials, sensitivity to the knots and the penalty, a simulation and the write-up.
---

## The question and the model

**What smooth curve describes y as a function of x?** The fit minimises

\[
\sum_i \bigl(y_i - B(x_i)'\alpha\bigr)^2 + \lambda\, \alpha' D'D\, \alpha ,
\]

where \(B\) is a B-spline basis and \(D\) takes second differences of neighbouring coefficients. A large \(\lambda\) gives a straight line and \(\lambda \to 0\) interpolates the noise. The effective degrees of freedom, the trace of the smoother matrix, say how many parameters the curve is worth. We pick \(\lambda\) by generalised cross-validation and compare the straight line with the smooth curve with an approximate F test.

<div class="write" markdown="1">
**How to write it: the model**

We fit head acceleration against time after impact with a penalised cubic B-spline (Eilers and Marx 1996), with 20 knots and a second-order difference penalty. The smoothing parameter is chosen by generalised cross-validation, and we report the effective degrees of freedom. We test the straight line against the smooth curve with an approximate F test, compare the fit with polynomials, and check the dependence on the number of knots and on the penalty in a simulation with a known function.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Fit the spline

```python
@@EST@@
```

@@READ_EST@@

## The curve and the comparison with polynomials

```python
@@CURVE@@
```

@@READ_CURVE@@

## Robustness: knots and the smoothing parameter

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

**References.** Eilers and Marx (1996), *Statistical Science*; Wood (2017), *Generalized Additive Models: An Introduction with R*; Silverman (1985), *Journal of the Royal Statistical Society B*; Ruppert, Wand and Carroll (2003), *Semiparametric Regression*.
