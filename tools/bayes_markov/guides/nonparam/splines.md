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
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("mcycle.csv")
print(len(d), "observations; time from", d.times.min(), "to", d.times.max(), "ms")
print("acceleration: mean", round(d.accel.mean(), 1), ", min", d.accel.min(), ", max", d.accel.max())
def s(res, label):
    return next(x["value"] for x in res["summary"] if x["label"] == label)
```

<div class="read" markdown="1">
**How to read it**

There are 133 observations of head acceleration for times from 2.4 to 57.6 milliseconds after impact. Acceleration averages -25.5 and ranges from -134 to 75: it is flat before the impact is felt, falls sharply, rebounds and decays.
</div>

## Fit the spline

```python
spec = dict(y="accel", x="times")
r = oes.run("pspline", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Penalised spline fit")
oes.show(r, "Test of nonlinearity")
```

<div class="read" markdown="1">
**How to read it**

Generalised cross-validation chooses lambda = 0.56, which gives 11.6 effective degrees of freedom (10.6 for the smooth part); a straight line would have 2. R-squared is 0.80 and the residual standard deviation is 22.7. The approximate F test of a straight line against the smooth curve is 44.1 on 9.6 and 121.4 degrees of freedom, with p below 0.0001: the relation is clearly not linear.
</div>

## The curve and the comparison with polynomials

```python
oes.show(r, "Comparison with polynomials")
oes.show(r, "Estimated smooth function")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The cross-validated squared error is 540.7 for the spline, against 1,633 for the best polynomial (degree 3) and 2,162 for a straight line, a reduction of about two thirds. The curve is flat at about -1 to -7 until 14 ms, drops to -104 at 19 ms, climbs through -74 at 25 ms to +31.5 at 30 ms, and decays to about zero (-4.7 at 47 ms). The credible bands are about ±24 at 2.4 ms, where the process is quiet, and ±11 at 19 ms, where it is violent: the method assumes constant noise, and in these data the noise is clearly larger in the middle, so the bands are too wide at the ends and too narrow in the middle.
</div>

## Robustness: knots and the smoothing parameter

```python
print("setting                      edf     R-squared")
for k in (8, 12, 20, 40):
    e = oes.run("pspline", d.to_csv(index=False), dict(spec, knots=k))
    print(f"{k:3d} knots, GCV lambda     {s(e, 'Effective df'):6.2f}  {s(e, 'R-squared'):8.3f}")
for lam in (0.001, 0.05, 0.5623, 5, 500, 100000):
    e = oes.run("pspline", d.to_csv(index=False), dict(spec, **{"lambda": lam}))
    print(f"lambda {lam:10g}          {s(e, 'Effective df'):6.2f}  {s(e, 'R-squared'):8.3f}")
```

<div class="read" markdown="1">
**How to read it**

The number of knots matters little once the penalty is chosen from the data: with 8, 12, 20 and 40 knots the effective degrees of freedom are 9.8, 10.4, 11.6 and 12.3 and R-squared is between 0.784 and 0.799. The penalty matters a lot when it is fixed: lambda = 0.001 gives 21.7 degrees of freedom and follows the noise, lambda = 5 gives 7.8 and R-squared falls to 0.747, lambda = 500 gives 3.3 and 0.315, and lambda = 100,000 gives 2.02 degrees of freedom, a straight line with R-squared 0.09.
</div>

## Diagnostics: a simulation with a known function

```python
rng = np.random.default_rng(2)
print("   n   edf   mean |error| spline   mean |error| cubic polynomial")
for n in (100, 300, 1000):
    a = []; p = []; ed = []
    for _ in range(10):
        x = np.sort(rng.uniform(0, 6, n)); y = np.sin(2 * x) + rng.normal(0, 0.4, n)
        e = oes.run("pspline", pd.DataFrame(dict(x=x, y=y)).to_csv(index=False), dict(y="y", x="x"))
        g = np.array(e["tables"][3]["rows"], float)
        a.append(np.mean(np.abs(g[:, 1] - np.sin(2 * g[:, 0])))); ed.append(s(e, "Effective df"))
        c = np.polyfit(x, y, 3); p.append(np.mean(np.abs(np.polyval(c, g[:, 0]) - np.sin(2 * g[:, 0]))))
    print(f"{n:5d} {np.mean(ed):6.1f} {np.mean(a):17.3f} {np.mean(p):24.3f}")
```

<div class="read" markdown="1">
**How to read it**

With the true function sin(2x) and noise 0.4, the spline uses 11 to 13 effective degrees of freedom and its mean absolute error falls from 0.102 with 100 observations to 0.064 with 300 and 0.042 with 1,000. A cubic polynomial stays at 0.58 to 0.59, because the function is too wavy for it and more data does not remove its bias.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We fit head acceleration against time after impact in 133 simulated crashes with a penalised cubic B-spline (20 knots, second-difference penalty). Generalised cross-validation selects a smoothing parameter of 0.56 and 11.6 effective degrees of freedom. The fitted curve is flat until about 14 ms, falls to -104 at 19 ms, rebounds to +31 at 30 ms and settles near zero after 45 ms. A straight line is rejected (F = 44.1, p < 0.0001), and the cross-validated error is 541, against 1,633 for the best polynomial. The fit is stable to the number of knots (8 to 40), but the bands assume constant variance and are too wide where the data are quiet and too narrow where they are noisy. In simulations the spline recovers a known function with an error that falls with the sample, while a cubic polynomial does not.
</div>

**References.** Eilers and Marx (1996), *Statistical Science*; Wood (2017), *Generalized Additive Models: An Introduction with R*; Silverman (1985), *Journal of the Royal Statistical Society B*; Ruppert, Wand and Carroll (2003), *Semiparametric Regression*.
