# Lecture 09 — Corporate Finance: Capital Budgeting and Project Valuation

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Value investment projects using incremental cash flows; calculate NPV and IRR; apply WACC consistently; evaluate uncertainty and real options.

## 1. Net present value
For project cash flows \(CF_t\) and discount rate \(r\),
\[
NPV=\sum_{t=0}^{T}\frac{CF_t}{(1+r)^t}.
\]
Accept an independent project when NPV is positive, subject to financing and strategic constraints. NPV is value-additive under consistent valuation assumptions.

## 2. Incremental free cash flow
A typical unlevered project cash flow is
\[
FCF=EBIT(1-T)+D\&A-CapEx-\Delta NWC.
\]
Include opportunity costs and externalities within the firm, exclude sunk costs, and account for terminal value and asset disposal taxes.

## 3. IRR and ranking
IRR solves \(NPV(r)=0\). Multiple sign changes can generate multiple IRRs; mutually exclusive projects can be mis-ranked when scale or timing differs. Use incremental NPV or an explicitly justified equivalent method.

## 4. Cost of capital
WACC is
\[
WACC=\frac{E}{D+E}r_E+\frac{D}{D+E}r_D(1-T),
\]
under the conventional corporate-tax treatment of deductible interest. Match project risk, currency, inflation basis, and cash-flow definition to the discount rate.

## 5. Scenario and sensitivity analysis
Vary key drivers individually and jointly. Scenario analysis should preserve economically coherent dependencies; simulation can quantify the distribution of NPV. Real options arise when management can defer, expand, contract, or abandon under uncertainty.

## 6. Worked example
A project costs 100 today and generates 60 in each of the next two years. Calculate NPV at 10% and solve for IRR. Explain the decision.

## 7. Computational lab
Construct a project model with taxes, depreciation, working capital, and terminal value. Perform sensitivity analysis on price, volume, discount rate, and investment cost.

## Problem set
1. Derive project free cash flow.
2. Calculate NPV and IRR.
3. Explain an IRR ranking conflict.
4. State when WACC is an appropriate discount rate.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 9.1 — A CAPM cost of capital and NPV sensitivity

*Data:* Fama–French market excess returns (for the market risk premium) and the AAA yield (for the cost of debt), December 2018.

A firm with equity beta 1.2, a target debt-to-value ratio of 30% and a 21% tax rate considers a project costing \$100 million that generates \$9 million in year 1, growing 3% a year for 20 years, with no salvage value.

1. Estimate the market risk premium as the 1926–2018 average excess return, and alternatively as the 1963–2018 average. Use the December 2018 T-bill rate (annualized) as \(r_f\) and the AAA yield as the pre-tax cost of debt.
2. Compute the cost of equity (CAPM), the WACC and the project NPV for each premium estimate.
3. Plot (tabulate) NPV against the discount rate and find the IRR. How much does the choice of sample for the risk premium matter?

```python
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import default
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

# TODO: MRP = 12 * mean(Mkt-RF); re = rf + 1.2 MRP; wacc = 0.7 re + 0.3 rd (1 - 0.21); NPV with growing annuity
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import default
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

rf = 12 * ff["RF"].iloc[-1]; rd = default.load()["AAA"].iloc[-1] / 100
cf = 9 * 1.03**np.arange(20); t = np.arange(1, 21)
npv = lambda r: -100 + np.sum(cf / (1 + r)**t)
for lab, lo in (("1926-2018", "1926-07"), ("1963-2018", "1963-07")):
    mrp = 12 * ff.loc[lo:, "Mkt-RF"].mean(); re = rf + 1.2 * mrp; w = 0.7 * re + 0.3 * rd * (1 - 0.21)
    print(f"MRP {lab}: {100*mrp:.2f}% -> cost of equity {100*re:.2f}%, WACC {100*w:.2f}%, NPV ${npv(w):.1f}m")
irr = optimize.brentq(npv, -0.5, 1.0)
print(f"rf = {100*rf:.2f}%, cost of debt {100*rd:.2f}%;  project IRR = {100*irr:.2f}%")
for r in (0.06, 0.08, 0.10, 0.12):
    print(f"  NPV at {100*r:.0f}%: ${npv(r):6.1f}m")
```

Output:

```text
MRP 1926-2018: 7.92% -> cost of equity 11.66%, WACC 9.12%, NPV $0.7m
MRP 1963-2018: 6.33% -> cost of equity 9.76%, WACC 7.78%, NPV $12.2m
rf = 2.16%, cost of debt 4.02%;  project IRR = 9.21%
  NPV at 6%: $  31.1m
  NPV at 8%: $  10.3m
  NPV at 10%: $  -5.9m
  NPV at 12%: $ -18.7m
```

The project's IRR (about 9.2%) is very close to the WACC, so the investment decision hinges on the market risk premium.
With the 1926–2018 premium (7.9%), the WACC is 9.1% and the NPV barely positive; with the 1963–2018 premium (6.3%), the WACC
is 7.8% and the NPV is about \$12 million. The NPV falls by roughly \$8 million for each percentage point added to the
discount rate. A choice of sample that looks technical changes the decision. Sensitivity analysis on the cost of capital
should be standard practice.

</details>

### Exercise 9.2 — Monte Carlo NPV with cash flows that move with the economy

*Data:* `macrodata` — U.S. annual real GDP growth (mean and volatility used to simulate cash-flow growth).

Suppose the project's cash-flow growth each year equals 1.5 times U.S. real GDP growth minus 1.5 percentage points (a cyclical project), with GDP growth drawn from a normal distribution with the historical mean and standard deviation.

1. Simulate 20,000 cash-flow paths for the project of the previous exercise (initial cash flow \$9m) and compute the NPV distribution at a 9% discount rate.
2. Report the mean NPV, the probability that NPV is negative and the 5th percentile.
3. Compare the mean NPV with the NPV of the **mean** cash flows. Why are they nearly the same here, and when would they differ?

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
g = np.log(m.groupby("year")["realgdp"].mean()).diff().dropna()
mu, sd = g.mean(), g.std()
# TODO: simulate growth = 1.5*gdp - 0.015; cash flows cumulative product; NPV distribution
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
g = np.log(m.groupby("year")["realgdp"].mean()).diff().dropna().loc[:2008]
mu, sd = g.mean(), g.std()
print(f"U.S. real GDP growth 1960-2008: mean {100*mu:.2f}%, sd {100*sd:.2f}%")
rng = np.random.default_rng(42); n, T, r = 20000, 20, 0.09
growth = 1.5 * rng.normal(mu, sd, (n, T - 1)) - 0.015
cf = 9 * np.hstack([np.ones((n, 1)), np.cumprod(1 + growth, axis=1)])
npv = -100 + cf @ (1 + r)**-np.arange(1, T + 1)
print(f"mean NPV ${npv.mean():.1f}m, sd ${npv.std():.1f}m, P(NPV < 0) = {np.mean(npv < 0):.3f}, 5th percentile ${np.percentile(npv, 5):.1f}m")
mean_cf = 9 * np.r_[1, np.cumprod(np.full(T - 1, 1 + 1.5 * mu - 0.015))]
print(f"NPV of the mean cash-flow path: ${-100 + mean_cf @ (1 + r)**-np.arange(1, T + 1):.1f}m")
```

Output:

```text
U.S. real GDP growth 1960-2008: mean 3.21%, sd 1.97%
mean NPV $4.1m, sd $6.3m, P(NPV < 0) = 0.268, 5th percentile $-5.9m
NPV of the mean cash-flow path: $4.1m
```

With growth uncertainty calibrated to U.S. GDP, the NPV has a mean of about \$4 million but a standard deviation of \$6
million: there is about a 27% chance the project destroys value, and a 5% chance of losing \$6 million or more. The mean NPV
equals the NPV of the mean cash-flow path here, because cash flows are close to linear in the growth shocks over this range
and the discount rate is fixed. They would differ with real options (the ability to abandon or expand), which make NPV a
**convex** function of cash flows — and with a discount rate that should itself reflect the project's cyclicality.

</details>

## Further reading
Berk and DeMarzo, *Corporate Finance*; Brealey, Myers and Allen, *Principles of Corporate Finance*.
