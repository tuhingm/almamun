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
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("cps71.csv")
print(len(d), "workers; age from", d.age.min(), "to", d.age.max())
print(d.groupby(pd.cut(d.age, [20, 30, 40, 50, 60, 70])).logwage.agg(["count", "mean"]).round(2))
def s(res, label):
    return next(x["value"] for x in res["summary"] if x["label"] == label)
```

<div class="read" markdown="1">
**How to read it**

There are 205 workers aged 21 to 65. Mean log wage rises from 13.16 in the 21-30 band to 13.73 in the 31-40 band, stays near 13.6 up to age 60 and falls to 13.09 in the last band, which has only 8 workers. A straight line cannot describe this pattern: it rises, flattens and falls.
</div>

## Estimate the regression function

```python
spec = dict(y="logwage", x="age", reps=199, seed=1)
r = oes.run("kernreg", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Fit and leave-one-out")
oes.show(r, "Bandwidth")
oes.show(r, "Test of a linear")
```

<div class="read" markdown="1">
**How to read it**

Cross-validation selects a bandwidth of h = 3.27 years (the rule of thumb 1.06 s n^(-1/5) gives 4.47, which smooths more). The smoother uses 6.8 effective parameters. Its leave-one-out error is 0.303, against 0.391 for a straight line: the straight line predicts held-out workers about 29% worse. The wild-bootstrap test rejects linearity (statistic 18.8, p = 0.005 with 199 replications). The residual standard deviation is 0.54 and the in-sample R-squared is 0.32. A quartic polynomial has a slightly lower cross-validated error (0.295), so the kernel fit is not the only flexible model that works here, and the choice among them would matter little for prediction.
</div>

## The estimated curve

```python
oes.show(r, "Estimated regression function")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

The fitted log wage climbs from 12.50 at age 22 to 13.53 at age 29 and 13.72 at age 36, where the slope reaches zero. It then stays between 13.64 and 13.73 until age 53 and falls to 13.31 by age 60. Pointwise 95% bands are about ±0.23 at the ends and ±0.13 to ±0.14 in the middle, and they ignore the smoothing bias. The fall after 55 rests on few workers: only 8 are older than 60.
</div>

## Robustness: bandwidth and degree

```python
print("bandwidth   effective parameters   CV error")
for h in (1.5, 2.5, 3.27, 5, 8, 20):
    e = oes.run("kernreg", d.to_csv(index=False), dict(y="logwage", x="age", bandwidth=h, reps=0))
    print(f"{h:8.2f} {s(e, 'Effective parameters'):14.2f} {s(e, 'CV error'):18.4f}")
e0 = oes.run("kernreg", d.to_csv(index=False), dict(y="logwage", x="age", degree=0, reps=0))
print("local constant: h", s(e0, "Bandwidth"), ", CV error", s(e0, "CV error"), "; local linear: h", s(r, "Bandwidth"), ", CV error", s(r, "CV error"))
for k in ("epanechnikov", "gaussian"):
    ek = oes.run("kernreg", d.to_csv(index=False), dict(y="logwage", x="age", kernel=k, reps=0))
    print(k, "kernel: h", s(ek, "Bandwidth"), ", CV error", s(ek, "CV error"))
```

<div class="read" markdown="1">
**How to read it**

The cross-validation error is flat near its minimum: 0.305 at h = 2.5, 0.303 at 3.27 and 0.308 at 5, while h = 1.5 (13 effective parameters) gives 0.319 and h = 20 (2.3 parameters, almost a straight line) gives 0.361. A local-constant fit needs a smaller bandwidth (1.86) and has a higher error (0.316) than the local-linear fit (0.303). The Epanechnikov kernel gives the same error (0.302), with a larger bandwidth because its window ends at h. The shape and the conclusion are the same for all reasonable choices.
</div>

## Diagnostics: a simulation with a known function

```python
rng = np.random.default_rng(0)
print("   n   mean |error| kernel   mean |error| line   selected h")
for n in (100, 300, 1000):
    ek = []; el = []; hh = []
    for _ in range(10):
        x = rng.uniform(0, 6, n); y = np.sin(x) + rng.normal(0, 0.4, n)
        e = oes.run("kernreg", pd.DataFrame(dict(x=x, y=y)).to_csv(index=False), dict(y="y", x="x", reps=0))
        g = np.array(e["tables"][3]["rows"], float)
        ek.append(np.mean(np.abs(g[:, 1] - np.sin(g[:, 0])))); hh.append(s(e, "Bandwidth"))
        b = np.polyfit(x, y, 1); el.append(np.mean(np.abs(np.polyval(b, g[:, 0]) - np.sin(g[:, 0]))))
    print(f"{n:5d} {np.mean(ek):18.3f} {np.mean(el):18.3f} {np.mean(hh):12.2f}")
rej = 0
for k in range(30):
    x = rng.uniform(0, 1, 100); y = 1 + 2 * x + rng.normal(0, 0.5, 100)
    e = oes.run("kernreg", pd.DataFrame(dict(x=x, y=y)).to_csv(index=False), dict(y="y", x="x", reps=99, seed=k))
    rej += s(e, "Linearity test p-value") < 0.05
print("linearity test, true line, rejections at 5%:", rej, "of 30")
```

<div class="read" markdown="1">
**How to read it**

With a known sine function and noise of 0.4, the mean absolute error of the kernel fit falls with the sample: 0.092 with 100 observations, 0.053 with 300 and 0.028 with 1,000, while a straight line stays at 0.36 because its bias does not shrink. The selected bandwidth falls from 0.30 to 0.23 as the sample grows. When the true regression is a straight line, the linearity test rejects in 3 of 30 samples at the 5% level, which is consistent with the nominal rate for this number of samples.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We estimate the relation between log wages and age for 205 workers by local-linear kernel regression with a Gaussian kernel and a bandwidth of 3.3 years chosen by leave-one-out cross-validation. The estimated profile rises steeply from age 22 (12.50) to about age 36 (13.72), is flat until the early fifties and falls afterwards (13.31 at age 60). A straight line predicts held-out workers worse than the kernel fit (cross-validated error 0.39 against 0.30), and a wild-bootstrap test rejects linearity (p = 0.005). A quartic polynomial predicts about as well as the kernel fit, the shape is stable for bandwidths between 2.5 and 5, and the estimate at the oldest ages rests on few workers. In a simulation the kernel fit recovers a known nonlinear function, with an error that falls as the sample grows.
</div>

**References.** Nadaraya (1964), *Theory of Probability and its Applications*; Watson (1964), *Sankhya*; Fan and Gijbels (1996), *Local Polynomial Modelling and its Applications*; Härdle and Mammen (1993), *Annals of Statistics*; Pagan and Ullah (1999), *Nonparametric Econometrics*.
