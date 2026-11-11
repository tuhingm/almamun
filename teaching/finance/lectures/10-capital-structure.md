# Lecture 10 — Capital Structure, Leverage, and Payout Policy

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Understand Modigliani-Miller propositions; derive tax-shield effects; compare trade-off and pecking-order theories; analyze dividends and repurchases.

## 1. Modigliani-Miller benchmark
In frictionless markets without taxes, distress costs, or agency frictions, firm value is independent of capital structure. The proposition is a benchmark separating financing effects from real investment decisions.

## 2. Corporate taxes
With deductible interest and permanent debt under simplified assumptions, the value of the tax shield is often represented as \(T_CD\). More general valuation depends on debt policy, risk, tax capacity, and discounting of shield cash flows.

## 3. Trade-off theory
Firms balance tax benefits of debt against expected financial distress and agency costs. Optimal leverage is model-dependent and can vary with asset tangibility, profitability, growth opportunities, and cash-flow risk.

## 4. Pecking order and information
With asymmetric information, firms may prefer internal funds, then debt, then equity. Equity issuance can be interpreted as a signal when managers have private information, but the prediction depends on model assumptions.

## 5. Payout policy
Dividends and repurchases distribute cash to shareholders. In frictionless benchmark models, payout form is irrelevant; taxes, signaling, agency costs, and investor clientele can make it matter.

## 6. Worked example
A firm with debt \(D=40\) and corporate tax rate 25% has a simplified perpetual tax-shield value of 10 under the permanent-riskless-debt benchmark. State why this is not a universal valuation formula.

## 7. Computational lab
Build a capital-structure valuation under alternative debt policies, tax rates, distress costs, and cost-of-debt schedules. Plot firm value and WACC against leverage.

## Problem set
1. State MM propositions I and II.
2. Derive the simplified tax-shield value.
3. Compare trade-off and pecking-order predictions.
4. Explain why payout irrelevance requires restrictive assumptions.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 10.1 — Leverage, risk and return across firms

*Data:* `return` — 142 U.S. firms: stock return 1990–1994 `return` (%), return on equity `roe`, debt-to-capital ratio `dkr`, earnings per share `eps`.

Modigliani–Miller (Proposition II) says that the expected return on equity rises with leverage, \(r_E=r_A+(r_A-r_D)D/E\).

1. Describe leverage across firms. Regress the 1990–94 stock return on `dkr` and on `roe`, with robust standard errors.
2. Is there evidence that higher-leverage firms earned higher returns? How much variation do the regressors explain?
3. Discuss why a single five-year return per firm is a very noisy measure of **expected** returns, and what MM II would predict about the **riskiness** of returns.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("return")
# TODO: describe dkr; return ~ dkr + roe
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("return")
print(d[["return", "dkr", "roe", "eps"]].describe().round(2).loc[["mean", "std", "min", "50%", "max"]].to_string())
r = smf.ols("Q('return') ~ dkr + roe", data=d).fit(cov_type="HC1")
print(r.params.round(3).to_dict(), "robust t:", r.tvalues.round(2).to_dict(), f"R2 = {r.rsquared:.3f}, n = {int(r.nobs)}")
d["lev_group"] = np.where(d["dkr"] > d["dkr"].median(), "high leverage", "low leverage")
print(d.groupby("lev_group")["return"].agg(["mean", "std", "size"]).round(2).to_string())
```

Output:

```text
return    dkr    roe     eps
mean   -4.04  25.65  17.73    2.02
std    39.42  16.67   9.88   43.15
min   -84.89   0.00   1.10  -89.30
50%    -8.67  26.40  15.80    6.70
max   131.84  79.50  61.10  266.60
{'Intercept': -0.862, 'dkr': 0.208, 'roe': -0.48} robust t: {'Intercept': -0.09, 'dkr': 0.9, 'roe': -1.5} R2 = 0.028, n = 142
               mean    std  size
lev_group                       
high leverage -1.05  43.17    71
low leverage  -7.04  35.33    71
```

Leverage varies widely across firms (debt-to-capital from 0 to 80%). The regression finds no significant relation between
leverage and 1990–94 returns, and the regressors explain about 3% of the variation. Higher-leverage firms did earn more on
average (about −1% against −7%), with more dispersion, but a single five-year realized return per firm is an extremely noisy
proxy for expected returns: the cross-sectional SD of returns is about 40 points. MM II predicts that leverage raises the
**riskiness** and the expected return of equity; the higher dispersion in the high-leverage group is consistent with the
first part.

</details>

### Exercise 10.2 — The WACC curve: MM with taxes and costly distress

*Data:* Market risk premium from the Fama–French data (1963–2018); risk-free rate and the BAA–AAA spread (December 2018) to calibrate distress costs.

An all-equity firm has asset beta 0.9. Debt is risk-free up to 20% debt-to-value; beyond that, the cost of debt rises with the BAA–AAA spread scaled by leverage, and expected distress costs grow quadratically.

1. For debt-to-value ratios from 0 to 70%, compute the levered equity beta (Hamada with taxes), the cost of equity, the cost of debt, the WACC and firm value for a perpetual pre-tax cash flow of \$100m (tax rate 21%).
2. Find the value-maximizing leverage.
3. Explain which ingredient makes the WACC curve U-shaped. What would MM without taxes predict?

```python
import numpy as np, pandas as pd
from arch.data import default
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

# TODO: grid over L = D/V; beta_E = beta_A (1 + (1-t) D/E); r_D = rf + spread * f(L); V = EBIT(1-t)/WACC - distress
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import default
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

mrp = 12 * ff.loc["1963-07":, "Mkt-RF"].mean(); rf = 12 * ff["RF"].iloc[-1]
dd = default.load(); spread = (dd["BAA"] - dd["AAA"]).iloc[-1] / 100
tax, bA, ebit = 0.21, 0.9, 100.0
rows = []
for L in np.arange(0, 0.71, 0.05):
    DE = L / (1 - L) if L < 1 else np.inf
    bE = bA * (1 + (1 - tax) * DE)
    rE = rf + bE * mrp
    rD = rf + (0 if L <= 0.2 else spread * ((L - 0.2) / 0.1)**1.5)
    wacc = (1 - L) * rE + L * rD * (1 - tax)
    distress = 400 * max(L - 0.3, 0)**2                  # PV of expected distress costs, $m
    V = ebit * (1 - tax) / wacc - distress
    rows.append((L, bE, rE, rD, wacc, V))
t = pd.DataFrame(rows, columns=["D/V", "equity beta", "r_E", "r_D", "WACC", "value $m"])
print(f"MRP {100*mrp:.2f}%, rf {100*rf:.2f}%, BAA-AAA spread {100*spread:.2f}%")
print(t.iloc[::2].round(4).to_string(index=False))
best = t.loc[t["value $m"].idxmax()]
print(f"value-maximizing leverage: D/V = {best['D/V']:.2f}, WACC {100*best['WACC']:.2f}%, value ${best['value $m']:.0f}m")
```

Output:

```text
MRP 6.33%, rf 2.16%, BAA-AAA spread 1.11%
 D/V  equity beta    r_E    r_D   WACC  value $m
 0.0       0.9000 0.0786 0.0216 0.0786 1005.1144
 0.1       0.9790 0.0836 0.0216 0.0769 1026.6746
 0.2       1.0778 0.0899 0.0216 0.0753 1049.1800
 0.3       1.2047 0.0979 0.0327 0.0763 1035.6983
 0.4       1.3740 0.1086 0.0530 0.0819  960.3935
 0.5       1.6110 0.1236 0.0793 0.0931  832.2971
 0.6       1.9665 0.1461 0.1104 0.1108  677.0874
 0.7       2.5590 0.1837 0.1457 0.1357  518.2850
value-maximizing leverage: D/V = 0.25, WACC 7.52%, value $1050m
```

As leverage rises, the equity beta rises (MM II), but cheap tax-deductible debt lowers the WACC up to about 25% debt, where
firm value peaks at about \$1.05 billion, 4% above the all-equity value. Beyond that, the rising cost of risky debt and
expected distress costs push the WACC up steeply. Without taxes and distress costs, MM Proposition I says the WACC and firm
value are independent of leverage: the curve would be flat. The U-shape comes from trading off the tax shield against
distress costs — the trade-off theory. The specific optimum depends entirely on the assumed distress-cost function.

</details>

## Further reading
Modigliani and Miller (1958, 1963); Myers (1984); Berk and DeMarzo, *Corporate Finance*.
