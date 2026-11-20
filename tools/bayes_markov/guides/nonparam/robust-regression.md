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
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("hills.csv")
print(len(d), "races")
print(d[["time", "dist", "climb"]].describe().round(1).loc[["mean", "min", "max"]])
def s(res, label):
    return next(x["value"] for x in res["summary"] if x["label"] == label)
```

<div class="read" markdown="1">
**How to read it**

There are 35 races with record times averaging 57.9 minutes (from 16 to 204.6), distances from 2 to 28 miles (mean 7.5) and climbs from 300 to 7,500 feet (mean 1,815). The largest race is far from the others in both distance and climb.
</div>

## Estimate

```python
spec = dict(y="time", x=["dist", "climb"], label="race")
r = oes.run("robreg", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Coefficients by estimator")
oes.show(r, "Scale and fit")
```

<div class="read" markdown="1">
**How to read it**

Five of the 35 races are flagged as outliers. The residual scale is 14.7 minutes for least squares and 3.8 for the MM estimator, a ratio of almost four. The distance coefficient is 6.22 minutes per mile by least squares and 6.64 by MM; the climb coefficient is 0.0110 by least squares and 0.0064 by MM, so least squares overstates the effect of climb by 70% and the largest shift is 2.2 least-squares standard errors. The MM standard errors are about one third as large (0.199 against 0.601 for distance, 0.0007 against 0.0021 for climb). Huber, bisquare and median regressions are in between or close to MM (distance 6.55 to 6.68).
</div>

## Which observations are unusual?

```python
oes.show(r, "Largest robust residuals")
oes.show(r, "Highest-leverage")
```

<div class="read" markdown="1">
**How to read it**

Knock Hill (time 78.65 minutes, fitted 14.1) has a robust residual of 17.2 scales, but a least-squares residual of only 4.4 standard deviations. Bens of Jura (204.6 against 146.5) has 15.5 against 2.1, and Two Breweries (170.3 against 145.0) has 6.7 against 0.7: least squares is pulled toward these races, which hides them (masking), and the robust fit reveals them. All three get zero weight. Black Hill and Goatfell are flagged more mildly (-2.9 and 2.6). Lairig Ghru has the highest leverage (0.69 against a rule of thumb of 0.17) but a small residual (0.33), so it is a good leverage point that supports the fit. Bens of Jura has a leverage of 0.42 and a large residual: it is a bad leverage point, the kind that moves least squares.
</div>

## Robustness: dropping the flagged races

```python
drop = ["Knock Hill", "Bens of Jura"]
dd = d[~d.race.isin(drop)]
r2 = oes.run("robreg", dd.to_csv(index=False), spec)
print("coefficients (distance, climb) by estimator, all races:")
for row in r["tables"][0]["rows"][1:]:
    print(f"  {row[0]:6s} OLS {row[1]:7.3f}  Huber {row[2]:7.3f}  bisquare {row[3]:7.3f}  LAD {row[4]:7.3f}  MM {row[5]:7.3f}")
print("without Knock Hill and Bens of Jura:")
for row in r2["tables"][0]["rows"][1:]:
    print(f"  {row[0]:6s} OLS {row[1]:7.3f}  Huber {row[2]:7.3f}  bisquare {row[3]:7.3f}  LAD {row[4]:7.3f}  MM {row[5]:7.3f}")
print("residual scale: OLS", s(r, "OLS scale"), "-> ", s(r2, "OLS scale"), "; MM", s(r, "MM scale"), "->", s(r2, "MM scale"))
```

<div class="read" markdown="1">
**How to read it**

Dropping Knock Hill and Bens of Jura moves least squares a lot (distance 6.22 to 6.69, climb 0.0110 to 0.0080, residual scale 14.7 to 6.1) and the MM estimator hardly at all (distance 6.641 to 6.645, climb 0.0064 to 0.0064, scale 3.8 to 3.2). Huber, bisquare and median regressions end up within 0.05 of MM on distance after the deletion. Two Breweries remains in the sample and keeps the least-squares scale well above the robust one.
</div>

## Diagnostics: contamination in a simulation

```python
rng = np.random.default_rng(3)
def draw(n, frac, kind):
    x = rng.normal(size=n); y = 1 + 2 * x + rng.normal(size=n)
    k = int(frac * n); i = rng.choice(n, k, replace=False)
    if kind == "bad leverage":
        x[i] = rng.normal(3, 0.5, k); y[i] = rng.normal(-8, 1, k)
    elif kind == "vertical":
        y[i] = y[i] + 15
    return pd.DataFrame(dict(y=y, x=x))
print("true slope = 2;  mean slope over 20 samples of 200")
print("contamination        OLS   Huber  bisquare    LAD     MM")
for kind, frac in (("none", 0), ("vertical", 0.2), ("bad leverage", 0.2)):
    est = []
    for _ in range(20):
        e = oes.run("robreg", draw(200, frac, kind).to_csv(index=False), dict(y="y", x=["x"], seed=1))
        est.append(e["tables"][0]["rows"][1][1:6])
    m = np.mean(est, axis=0)
    print(f"{kind:14s} {frac:5.0%} " + " ".join(f"{v:7.2f}" for v in m))
```

<div class="read" markdown="1">
**How to read it**

The true slope is 2. Without contamination all five estimators give 1.99. With 20% of the observations shifted vertically by 15 units the slope hardly changes for any of them (1.97 to 2.05), because the outliers are unrelated to x. With 20% bad leverage points (large x and very low y) the slope of least squares is -1.22, Huber -1.31, bisquare -1.30 and median regression -1.27: none of them is robust to this contamination. The MM estimator gives 1.99. M estimators and median regression protect against outliers in y only; protection against bad leverage needs a high-breakdown start such as MM.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We regress race time on distance and climb for 35 Scottish hill races. Least squares gives 6.22 minutes per mile and 0.0110 minutes per foot of climb with a residual standard deviation of 14.7; the MM estimator (Yohai 1987) gives 6.64 and 0.0064 with a scale of 3.8, and its standard errors are about one third as large. Knock Hill, Bens of Jura and Two Breweries have zero weight in the robust fit; Bens of Jura is also a high-leverage race. Removing Knock Hill and Bens of Jura moves the least-squares coefficients by 8% (distance) and 27% (climb) but leaves the MM estimates unchanged. In a simulation with 20% bad leverage points, only the MM estimator recovers the true slope. We report MM as the main estimate and least squares as the comparison.
</div>

**References.** Huber (1964), *Annals of Mathematical Statistics*; Rousseeuw and Yohai (1984); Yohai (1987), *Annals of Statistics*; Maronna, Martin and Yohai (2006), *Robust Statistics*; Atkinson (1986), *Journal of the Royal Statistical Society C*.
