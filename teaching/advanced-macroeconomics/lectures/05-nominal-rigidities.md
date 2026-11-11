# Lecture 05 — Nominal Rigidities, Calvo Pricing, and the New Keynesian Phillips Curve

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive the firm's reset-price problem under Calvo pricing; interpret the New Keynesian Phillips curve and its slope.

## 1. Price-setting environment
Each period, a fraction \(1-\theta\) of firms can reset prices; fraction \(\theta\) keep prices unchanged. A resetting firm chooses \(P_t^*\) to maximize expected discounted profits while its price may remain fixed for multiple periods.

## 2. Reset-price condition
The optimal reset price is a discounted markup over a weighted average of future nominal marginal costs. In log-linear form around a zero-inflation steady state, the aggregate pricing condition yields
\[
\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t,
\]
where \(\pi_t\) is inflation, \(x_t\) the output gap, and \(u_t\) a cost-push disturbance. Under the standard Calvo model,
\[
\kappa=\frac{(1-\theta)(1-\beta\theta)}{\theta}\times \text{real marginal-cost scaling}.
\]
The scaling depends on preferences and technology; it should not be silently set to one across parameterizations.

## 3. Economic interpretation
A larger \(\theta\) means fewer firms reset prices each period and typically a flatter Phillips curve. Expected future inflation matters because today's reset price is chosen for a potentially long spell. Marginal cost, rather than output itself, is the primitive pricing determinant; the output-gap representation uses a relation between marginal cost and the gap.

## 4. Rotemberg alternative
With quadratic price-adjustment costs, firms choose prices every period but pay a cost proportional to squared inflation. This yields a forward-looking Phillips curve with a slope governed by the adjustment-cost parameter. Calvo and Rotemberg models are alternative microfoundations, not identical mechanisms.

## 5. Worked example
For \(\beta=0.99\) and \(\theta=0.75\), compute the Calvo component \((1-\theta)(1-\beta\theta)/\theta\). Interpret how increasing \(\theta\) changes price flexibility.

## 6. Computational lab
Plot the Phillips curve slope over a grid of \(\theta\). Simulate inflation following a cost-push shock under alternative degrees of price rigidity.

## Problem set
1. Explain the reset-price firm's forward-looking choice.
2. Interpret each term in the NK Phillips curve.
3. Compare Calvo and Rotemberg pricing.
4. Explain why marginal cost is central to price-setting.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 5.1 — From price stickiness to the slope of the Phillips curve

*Data:* Calibration: \(\beta=0.99\), \(\sigma=1\), \(\varphi=1\) (Galí, 2015, ch. 3, without strategic complementarities).

In the Calvo model, a share \(\theta\) of firms cannot reset prices each quarter, so the average price duration is \(1/(1-\theta)\), and the NKPC slope on the output gap is
\(\kappa=\frac{(1-\theta)(1-\beta\theta)}{\theta}(\sigma+\varphi)\).

1. Compute \(\kappa\) for average durations of 2, 3, 4, 6 and 8 quarters.
2. Micro evidence (Bils and Klenow, 2004; Nakamura and Steinsson, 2008) suggests durations of 4–11 months for regular prices. What range of \(\kappa\) does that imply?
3. Invert the mapping numerically: what duration is implied by a slope of \(\kappa=0.01\), roughly what many macro estimates find?

```python
import numpy as np
from scipy import optimize
b, s, phi = 0.99, 1.0, 1.0
kappa = lambda th: (1 - th) * (1 - b * th) / th * (s + phi)
# TODO: durations D -> theta = 1 - 1/D; invert kappa(theta) = 0.01 with brentq
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy import optimize
b, s, phi = 0.99, 1.0, 1.0
kappa = lambda th: (1 - th) * (1 - b * th) / th * (s + phi)
for D in (2, 3, 4, 6, 8):
    th = 1 - 1 / D
    print(f"duration {D} quarters: theta = {th:.3f}, kappa = {kappa(th):.4f}")
lo, hi = kappa(1 - 1 / (4 / 3)), kappa(1 - 1 / (11 / 3))
print(f"durations of 4-11 months (1.33-3.67 quarters): kappa between {hi:.3f} and {lo:.3f}")
th = optimize.brentq(lambda t: kappa(t) - 0.01, 0.5, 0.9999)
print(f"kappa = 0.01 requires theta = {th:.4f}, i.e. an average price duration of {1/(1-th):.1f} quarters ({1/(1-th)/4:.1f} years)")
```

Output:

```text
duration 2 quarters: theta = 0.500, kappa = 1.0100
duration 3 quarters: theta = 0.667, kappa = 0.3400
duration 4 quarters: theta = 0.750, kappa = 0.1717
duration 6 quarters: theta = 0.833, kappa = 0.0700
duration 8 quarters: theta = 0.875, kappa = 0.0382
durations of 4-11 months (1.33-3.67 quarters): kappa between 0.210 and 4.515
kappa = 0.01 requires theta = 0.9361, i.e. an average price duration of 15.7 quarters (3.9 years)
```

The slope \(\kappa\) falls steeply with price duration: from 1.0 at 2 quarters to 0.17 at one year and 0.04 at two years.
Micro evidence on regular price durations of 4–11 months implies \(\kappa\) of 0.2 or more. Yet a slope of 0.01, typical of
aggregate estimates, would require prices to stay fixed for almost four years on average. This micro–macro gap is why
models add strategic complementarities (real rigidities), which flatten the Phillips curve for a given frequency of price
changes.

</details>

### Exercise 5.2 — Estimating a New Keynesian Phillips curve by GMM

*Data:* `macrodata` — U.S. quarterly CPI inflation and unemployment, 1960–2007.

Estimate the hybrid NKPC \(\pi_t=\gamma_f E_t\pi_{t+1}+\gamma_b\pi_{t-1}+\kappa\,x_t+\varepsilon_t\), with the output gap proxied by the negative unemployment gap \(x_t=-(u_t-\bar u_t)\), \(\bar u_t\) the HP trend of unemployment.

1. Replace \(E_t\pi_{t+1}\) with realized \(\pi_{t+1}\) and estimate by 2SLS with instruments dated \(t-1\) and earlier (lags 1–4 of inflation, lags 1–2 of the gap and of the T-bill rate).
2. Report \(\gamma_f\), \(\gamma_b\) and \(\kappa\). Is the curve flat? Is inflation mostly forward- or backward-looking?
3. Discuss why weak instruments plague NKPC estimation (Mavroeidis, Plagborg-Møller and Stock, 2014).

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
# TODO: x = -(u - HP trend of u); pi_lead = pi.shift(-1); instruments: lags
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"year": m["year"], "pi": m["infl"], "x": -hpfilter(m["unemp"], 1600)[0], "i": m["tbilrate"]})
d["pi_f"], d["pi_b"] = d["pi"].shift(-1), d["pi"].shift(1)
Z = []
for k in (2, 3, 4):
    d[f"pi_l{k}"] = d["pi"].shift(k); Z.append(f"pi_l{k}")
for v in ("x", "i"):
    for k in (1, 2):
        d[f"{v}_l{k}"] = d[v].shift(k); Z.append(f"{v}_l{k}")
d = d[(d.year >= 1960) & (d.year <= 2007)].dropna()
iv = IV2SLS.from_formula(f"pi ~ 1 + pi_b + x + [pi_f ~ {' + '.join(Z)}]", data=d).fit(cov_type="kernel")
print(iv.params[["pi_f", "pi_b", "x"]].round(3).to_string())
print("HAC se:", iv.std_errors[["pi_f", "pi_b", "x"]].round(3).to_dict())
fs = smf.ols(f"pi_f ~ pi_b + x + {' + '.join(Z)}", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test(", ".join(f"{z} = 0" for z in Z), scalar=True).statistic / len(Z)
print(f"first-stage F for pi_(t+1) on excluded instruments: {F:.1f};  gamma_f + gamma_b = {iv.params['pi_f'] + iv.params['pi_b']:.3f}")
```

Output:

```text
pi_f    0.854
pi_b    0.090
x       0.152
HAC se: {'pi_f': 0.119, 'pi_b': 0.109, 'x': 0.145}
first-stage F for pi_(t+1) on excluded instruments: 12.9;  gamma_f + gamma_b = 0.944
```

The estimates put most weight on expected future inflation (\(\gamma_f\approx0.85\)) and little on lagged inflation, with the
two summing to slightly below one. The slope on the unemployment gap is positive but imprecise — not significantly
different from zero: the Phillips curve is flat and poorly identified. The first-stage \(F\) of about 13 looks acceptable,
but NKPC estimates are notoriously sensitive to the instrument set, the gap measure and the sample, because lagged variables
predict future inflation only weakly once current inflation is controlled for (Mavroeidis, Plagborg-Møller and Stock, 2014).
Use weak-instrument-robust methods before drawing strong conclusions.

</details>

## Further reading
Galí, *Monetary Policy, Inflation, and the Business Cycle*; Woodford, *Interest and Prices*.
