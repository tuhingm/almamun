# Lecture 04 — Real Business Cycle Models and Competitive Equilibrium

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Construct a real business cycle (RBC) model; derive equilibrium conditions; understand calibration, shock propagation, and impulse responses.

## 1. Household and firm
The household maximizes \(E_0\sum_t\beta^t[\log c_t-\chi n_t^{1+\varphi}/(1+\varphi)]\), subject to its budget and capital accumulation. The firm has
\[
y_t=A_tk_t^\alpha n_t^{1-\alpha}.
\]
Optimality gives the consumption Euler equation, labor-supply condition
\[
\chi n_t^\varphi=\frac{w_t}{c_t},
\]
and factor prices equal marginal products.

## 2. Productivity process
Let \(\log A_t=\rho_a\log A_{t-1}+\varepsilon_t^a\), with \(|\rho_a|<1\) and innovation variance \(\sigma_a^2\).

## 3. Competitive equilibrium
Given initial capital and shock processes, an equilibrium is a set of allocations and prices satisfying household and firm optimality, market clearing, and the capital law:
\[
y_t=c_t+i_t,\qquad k_{t+1}=(1-\delta)k_t+i_t.
\]
Government and external sectors are absent in the baseline model.

## 4. Quantitative analysis
Calibration assigns parameters using long-run averages or external estimates. A solution method maps shocks and states into decision rules. Impulse responses describe conditional dynamic responses; variance decompositions quantify contributions of shocks under the model; moments compare simulated data with empirical targets.

## 5. Worked steady state
At a deterministic steady state, \(1=\beta(\alpha y/k+1-\delta)\), so
\[
k/y=\frac{\alpha}{\beta^{-1}-1+\delta}.
\]
Use the resource constraint to obtain consumption and investment shares.

## 6. Computational lab
Build a Dynare or Python RBC model. Compute the steady state, verify residuals, solve at first order, simulate productivity shocks, and report IRFs for output, consumption, investment, hours, and wages. Change \(\rho_a\) and \(\sigma_a\) separately.

## Problem set
1. List all equilibrium conditions and endogenous variables.
2. Derive the steady-state capital-output ratio.
3. Explain the distinction between shock size and persistence.
4. Discuss which model moments are targeted in calibration.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 4.1 — U.S. business-cycle facts with the HP filter

*Data:* `macrodata` — U.S. quarterly real GDP, consumption, investment and government spending, 1959–2009.

1. HP-filter the logs of real GDP, consumption, investment and government spending (\(\lambda=1600\)).
2. Report for each the standard deviation (%), the standard deviation relative to output, the correlation with output, and the first-order autocorrelation.
3. Which facts must a business-cycle model reproduce? Which of them does the basic RBC model struggle with?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
# TODO: cycles = {v: hpfilter(np.log(m[v]), 1600)[0] * 100 for v in [...]}
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
names = {"realgdp": "output", "realcons": "consumption", "realinv": "investment", "realgovt": "government"}
cyc = pd.DataFrame({lab: hpfilter(np.log(m[v]), 1600)[0] * 100 for v, lab in names.items()})
cyc["employment"] = hpfilter(np.log(m["pop"] * (1 - m["unemp"] / 100)), 1600)[0] * 100
out = pd.DataFrame({"sd %": cyc.std(), "rel. sd": cyc.std() / cyc["output"].std(),
                    "corr with output": cyc.corr()["output"], "autocorr": [cyc[c].autocorr() for c in cyc]})
print(out.round(2).to_string())
```

Output:

```text
sd %  rel. sd  corr with output  autocorr
output       1.54     1.00              1.00      0.86
consumption  1.24     0.80              0.87      0.87
investment   7.19     4.66              0.91      0.81
government   2.62     1.70             -0.06      0.77
employment   0.77     0.50              0.88      0.91
```

These are the classic stylized facts: investment is about 4.7 times as volatile as output and strongly procyclical;
consumption is smoother (0.8 times as volatile) and procyclical; employment is procyclical and persistent; government
spending is essentially acyclical; all series are persistent (autocorrelations of 0.8–0.9). A business-cycle model must
reproduce these relative volatilities and co-movements. The basic RBC model does well on investment and consumption, but
it generates too little employment volatility and too much correlation between productivity and output.

</details>

### Exercise 4.2 — Simulating the stochastic growth model with full depreciation

*Data:* TFP process estimated from the U.S. Solow residual of Lecture 2 (rebuilt here); model calibration \(\alpha=1/3\), \(\beta=0.99\).

With log utility and full depreciation (Brock–Mirman), the policy functions are known in closed form:
\(k_{t+1}=\alpha\beta z_tk_t^{\alpha}\) and \(c_t=(1-\alpha\beta)z_tk_t^{\alpha}\).

1. Re-estimate an AR(1) for the (linearly detrended) U.S. log Solow residual.
2. Simulate the model for 10,000 quarters, HP-filter simulated output, consumption and investment, and compute the same moments as in the previous exercise.
3. Compare with the data. Why does the model make consumption **exactly** as volatile as output, and what feature of the full RBC model (with \(\delta<1\)) fixes this?

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
# TODO: rebuild log z (Lecture 2 exercise), detrend, AR(1); simulate k' = a b z k^a
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a, b = 0.025, 1/3, 0.99
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
lz = np.log(Y) - a * np.log(K) - (1 - a) * np.log(N)
tt = np.arange(len(lz)); res = lz - np.polyval(np.polyfit(tt, lz, 1), tt)
rho = np.corrcoef(res[1:], res[:-1])[0, 1]; sig = np.std(res[1:] - rho * res[:-1])
print(f"TFP: rho = {rho:.3f}, sigma = {100*sig:.3f}%")
rng = np.random.default_rng(1); T = 10000
z = np.zeros(T); k = np.zeros(T); k[0] = (a * b)**(1 / (1 - a))
for t in range(1, T): z[t] = rho * z[t - 1] + sig * rng.normal()
yv = np.zeros(T)
for t in range(T):
    yv[t] = np.exp(z[t]) * k[t]**a
    if t + 1 < T: k[t + 1] = a * b * yv[t]
c, inv = (1 - a * b) * yv, a * b * yv
cyc = pd.DataFrame({v: hpfilter(np.log(x), 1600)[0] * 100 for v, x in (("output", yv), ("consumption", c), ("investment", inv))})
print(pd.DataFrame({"sd %": cyc.std(), "rel. sd": cyc.std() / cyc["output"].std(), "corr with output": cyc.corr()["output"],
                    "autocorr": [cyc[x].autocorr() for x in cyc]}).round(2).to_string())
```

Output:

```text
TFP: rho = 0.964, sigma = 0.737%
             sd %  rel. sd  corr with output  autocorr
output       1.27      1.0               1.0      0.83
consumption  1.27      1.0               1.0      0.83
investment   1.27      1.0               1.0      0.83
```

The estimated TFP process (\(\rho\approx0.96\), \(\sigma\approx0.7\%\)) is close to the standard RBC calibration. But with full
depreciation and log utility, consumption and investment are **constant shares** of output, so all three series have exactly
the same volatility and are perfectly correlated — far from the data, where investment is about six times as volatile as
consumption. With \(\delta<1\), capital is a durable stock: households smooth consumption by letting investment absorb shocks,
which generates the observed volatility ranking. The simple model is useful as a check on numerical methods (Lecture 10),
not as a model of the cycle.

</details>

## Further reading
King, Plosser and Rebelo (1988); Heer and Maussner, *Dynamic General Equilibrium Modeling*.
