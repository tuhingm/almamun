# Lecture 02 — Firms, Capital Accumulation, and Investment

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive factor demands, the capital Euler equation, and Tobin's \(q\); understand depreciation, adjustment costs, and investment dynamics.

## 1. Technology and static optimization
A competitive firm produces
\[
y_t=A_t k_t^\alpha n_t^{1-\alpha},\quad 0<\alpha<1.
\]
Taking factor prices as given, profit maximization implies
\[
w_t=(1-\alpha)A_t(k_t/n_t)^\alpha,\qquad
r_t^k=\alpha A_t(k_t/n_t)^{\alpha-1},
\]
where \(r_t^k\) is the rental rate of capital before depreciation.

## 2. Capital accumulation
Capital evolves according to
\[
k_{t+1}=(1-\delta)k_t+i_t,
\]
with depreciation rate \(\delta\in(0,1)\). Investment is a flow; installed capital is a stock. Confusing these objects leads to incorrect timing in dynamic models.

## 3. Investment with adjustment costs
Let investment adjustment costs be \(\frac{\phi}{2}(i_t/k_t-\delta)^2k_t\). The firm chooses investment to maximize discounted dividends. Define marginal value of installed capital as \(q_t\). The investment FOC is
\[
q_t=1+\phi(i_t/k_t-\delta).
\]
The capital valuation equation equates the value of installed capital to its marginal product plus the discounted continuation value:
\[
q_t=E_t\left[M_{t,t+1}\{r_{t+1}^k+(1-\delta)q_{t+1}\}\right],
\]
where \(M_{t,t+1}\) is the stochastic discount factor.

## 4. Balanced growth and normalization
When technology grows, detrend real quantities by the appropriate technology or labor-augmenting trend. A stationary model requires consistent scaling of output, consumption, investment, and capital. Verify that the normalized capital law preserves the original resource constraint.

## 5. Worked example
With Cobb–Douglas production, \(A=1\), \(k/n=4\), and \(\alpha=0.36\), the marginal product of capital is \(0.36(4)^{-0.64}\). The wage is \(0.64(4)^{0.36}\). These prices satisfy the competitive firm's factor-share conditions.

## 6. Computational lab
Compute factor prices and a steady-state capital stock. Add quadratic adjustment costs and simulate the response of investment and Tobin's \(q\) to a temporary productivity shock. Check the capital law period by period.

## Problem set
1. Derive both factor-demand equations.
2. Explain why investment can respond more slowly when \(\phi\) is large.
3. Derive the investment FOC from the adjustment-cost function.
4. Interpret the capital valuation equation.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 2.1 — Building a capital stock and a Solow residual for the U.S.

*Data:* `macrodata` — U.S. quarterly real GDP, real investment, unemployment rate and population, 1959–2009.

1. Build a capital stock with the perpetual-inventory method \(K_{t+1}=(1-\delta)K_t+I_t\), \(\delta=0.025\) per quarter, initializing \(K_0=I_0/(g+\delta)\) with \(g\) the average investment growth rate.
2. Proxy labor input by employment, \(N_t=pop_t(1-u_t/100)\), and compute the Solow residual \(\log z_t=\log Y_t-\alpha\log K_t-(1-\alpha)\log N_t\) with \(\alpha=1/3\).
3. Decompose average annual growth of output per worker into capital deepening and TFP growth, before and after 1973.

```python
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); d = 0.025; a = 1/3
# TODO: perpetual inventory; employment; Solow residual; growth accounting by period
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a = 0.025, 1/3
g = np.mean(np.diff(np.log(I)))
K = np.empty_like(Y); K[0] = I[0] / (g + dl)
for t in range(len(Y) - 1):
    K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
logz = np.log(Y) - a * np.log(K) - (1 - a) * np.log(N)
df = pd.DataFrame({"year": m["year"], "y": np.log(Y / N), "k": np.log(K / N), "z": logz})
print(f"capital-output ratio (annual): 1959 {K[0]/Y[0]/4:.2f}, 1973 {K[56]/Y[56]/4:.2f}, 2009 {K[-1]/Y[-1]/4:.2f}")
for lo, hi in ((1960, 1973), (1973, 2007)):
    s = df[(df.year >= lo) & (df.year <= hi)]; T = (len(s) - 1) / 4
    gy = 100 * (s.y.iloc[-1] - s.y.iloc[0]) / T; gk = 100 * a * (s.k.iloc[-1] - s.k.iloc[0]) / T; gz = 100 * (s.z.iloc[-1] - s.z.iloc[0]) / T
    print(f"{lo}-{hi}: output/worker {gy:.2f}%/yr = capital deepening {gk:.2f} + TFP {gz:.2f}")
np.save("/tmp/us_logz.npy", logz)
```

Output:

```text
capital-output ratio (annual): 1959 0.80, 1973 0.82, 2009 1.29
1960-1973: output/worker 2.77%/yr = capital deepening 1.09 + TFP 1.68
1973-2007: output/worker 1.86%/yr = capital deepening 1.00 + TFP 0.86
```

The perpetual-inventory capital stock rises relative to output over the sample. Growth accounting shows the post-1973
productivity slowdown: output per worker growth fell from about 2.8% to 1.9% a year, and almost all of the decline is
TFP (from about 1.7% to 0.9%), while the contribution of capital deepening was stable at about 1 point. This replicates
the classic finding with a crude employment measure; using hours and a capital measure that excludes residential
structures changes the levels but not the story.

</details>

### Exercise 2.2 — Tobin's q: does market value predict investment?

*Data:* `grunfeld` (statsmodels) — investment, market value and capital stock of 11 large U.S. firms, 1935–1954.

With adjustment costs, investment depends on Tobin's \(q\). Grunfeld's data give firm value \(V\) and capital \(K\); use
\(V/K\) as a (crude) average-\(q\) proxy.

1. Regress the investment rate \(I_{it}/K_{it}\) on lagged \(V/K\), with firm fixed effects.
2. Interpret the coefficient as \(1/\phi\) for quadratic adjustment costs \(\frac{\phi}{2}(I/K)^2K\), and report the implied \(\phi\).
3. Why does the \(q\) theory typically explain investment poorly, and what does measurement error in \(q\) do to the coefficient?

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
g["ik"] = g["invest"] / g["capital"]; g["q"] = g["value"] / g["capital"]
# TODO: lag q by firm; ik ~ q_lag + C(firm)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
g = sm.datasets.grunfeld.load_pandas().data.sort_values(["firm", "year"])
g = g[g["capital"] > 5].copy()
g["ik"] = g["invest"] / g["capital"]; g["q"] = g["value"] / g["capital"]
g["q_lag"] = g.groupby("firm")["q"].shift(1)
d = g.dropna(subset=["q_lag"])
r = smf.ols("ik ~ q_lag + C(firm)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["firm"]})
b = r.params["q_lag"]
print(f"investment rate on lagged V/K (firm FE): {b:.4f} (cluster se {r.bse['q_lag']:.4f}), within R2 approx {r.rsquared:.3f}")
print(f"implied adjustment-cost parameter phi = 1/b = {1/b:.1f}")
print(f"median I/K {d['ik'].median():.3f}; median V/K {d['q'].median():.2f}")
```

Output:

```text
investment rate on lagged V/K (firm FE): 0.0291 (cluster se 0.0092), within R2 approx 0.533
implied adjustment-cost parameter phi = 1/b = 34.4
median I/K 0.379; median V/K 3.51
```

The coefficient on lagged \(V/K\) is positive and significant, but small: it implies an adjustment-cost parameter of about
34, so firms would take decades to close a gap between \(q\) and one. That is the well-known failure of \(q\) theory —
implausibly slow adjustment. One reason is measurement: \(V/K\) is **average** q, the relevant concept is **marginal** q,
and market values are noisy, so the coefficient is attenuated (Erickson and Whited, 2000). With 11 firms, the clustered
standard error also rests on few clusters.

</details>

## Further reading
Hayashi (1982), “Tobin's Marginal q and Average q”; Ljungqvist and Sargent, *Recursive Macroeconomic Theory*.
