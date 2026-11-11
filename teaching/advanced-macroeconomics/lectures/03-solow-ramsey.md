# Lecture 03 — Solow and Ramsey Growth: Steady States and Transitional Dynamics

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive the Solow steady state, golden-rule capital, and Ramsey saddle path; distinguish exogenous saving from optimal saving.

## 1. Solow model
In per-worker terms, output is \(y=f(k)=k^\alpha\). Capital evolves as
\[
\dot k=s f(k)-(n+g+\delta)k,
\]
where \(s\) is the saving rate, \(n\) population growth, \(g\) labor-augmenting technology growth, and \(\delta\) depreciation.

The steady state satisfies \(sf(k^*)=(n+g+\delta)k^*\), giving
\[
k^*=\left(\frac{s}{n+g+\delta}\right)^{1/(1-\alpha)}.
\]
A higher saving rate raises steady-state capital and output per effective worker, but does not permanently raise the exogenous long-run growth rate.

## 2. Golden-rule capital
Steady-state consumption is \(c=f(k)-(n+g+\delta)k\). The golden-rule condition is
\[
f'(k_{GR})=n+g+\delta.
\]
It maximizes steady-state consumption, not necessarily welfare along the transition.

## 3. Ramsey household
A planner chooses consumption to maximize
\[
\int_0^\infty e^{-\rho t}u(c_t)dt
\]
subject to \(\dot k=f(k)-c-(n+g+\delta)k\). For CRRA utility, the Euler equation is
\[
\frac{\dot c}{c}=\frac{1}{\sigma}[f'(k)-\delta-\rho-\sigma g],
\]
under the standard effective-worker normalization and corresponding parameter convention. State the normalization explicitly when applying this expression.

## 4. Phase diagram and stability
The \(\dot k=0\) locus is \(c=f(k)-(n+g+\delta)k\). The \(\dot c=0\) locus is determined by the modified golden-rule condition. The steady state is typically saddle-path stable: one stable eigenvalue and one unstable eigenvalue. The initial capital stock is predetermined; consumption jumps onto the stable arm.

## 5. Worked example
For \(f(k)=k^{0.3}\), \(s=0.2\), and \(n+g+\delta=0.05\), the Solow steady state is \(k^*=(0.2/0.05)^{1/0.7}\). Evaluate numerically and calculate steady-state output and consumption.

## 6. Computational lab
Plot Solow transition paths for alternative saving rates. For Ramsey, linearize the two-dimensional system around steady state, compute eigenvalues, and identify the stable eigenvector.

## Problem set
1. Derive the Solow steady state.
2. Derive the golden-rule condition.
3. Explain why the Ramsey model has a jump variable.
4. Compare a saving-rate increase in Solow and Ramsey.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 3.1 — The Ramsey model: steady state, saddle path and transition

*Data:* Calibrated to standard annual U.S. values: \(\alpha=1/3\), \(\beta=0.96\), \(\delta=0.08\), \(\sigma=2\).

In discrete time, the resource constraint is \(k_{t+1}=k_t^{\alpha}+(1-\delta)k_t-c_t\) and the Euler equation is \(c_t^{-\sigma}=\beta c_{t+1}^{-\sigma}(\alpha k_{t+1}^{\alpha-1}+1-\delta)\).

1. Compute the steady state \((k^*,c^*)\) and the saving rate.
2. Find the saddle path by **shooting**: from \(k_0=0.5k^*\), search for the initial \(c_0\) such that the path converges to the steady state (bisection on \(c_0\)).
3. Report the transition: how many years until capital closes half of the gap? How does the saving rate evolve along the path?

```python
import numpy as np
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
def path(c0, k0, T=200):
    # TODO: iterate k' from the resource constraint, c' from the Euler equation; return arrays (stop if k<0 or c<0)
    ...
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
a, b, d, s = 1/3, 0.96, 0.08, 2.0
kss = (a / (1/b - 1 + d))**(1 / (1 - a)); css = kss**a - d * kss
print(f"steady state: k* = {kss:.4f}, c* = {css:.4f}, y* = {kss**a:.4f}, saving rate = {d*kss/kss**a:.3f}")
def path(c0, k0, T=150):
    k, c = [k0], [c0]
    for _ in range(T):
        k1 = k[-1]**a + (1 - d) * k[-1] - c[-1]
        if k1 <= 0: return np.array(k), np.array(c), "crash"
        c1 = c[-1] * (b * (a * k1**(a - 1) + 1 - d))**(1 / s)
        k.append(k1); c.append(c1)
        if c1 > 3 * css: return np.array(k), np.array(c), "explode"
    return np.array(k), np.array(c), "ok"
k0 = 0.5 * kss; lo, hi = 1e-6, k0**a + (1 - d) * k0
for _ in range(200):
    c0 = 0.5 * (lo + hi); k, c, status = path(c0, k0)
    if status == "crash" or (status == "ok" and k[-1] < kss): hi = c0      # consumed too much -> capital falls
    else: lo = c0                                                          # consumed too little -> overshoot
k, c, _ = path(c0, k0, 60)
print(f"saddle-path initial consumption c0 = {c0:.6f} ({c0/css:.1%} of c*)")
half = np.argmax(kss - k <= 0.5 * (kss - k0))
print(f"capital closes half of the gap after {half} years")
for t in (0, 5, 10, 20, 40):
    y = k[t]**a
    print(f"  t = {t:2d}: k/k* = {k[t]/kss:.3f}, c/c* = {c[t]/css:.3f}, saving rate = {(y - c[t])/y:.3f}")
```

Output:

```text
steady state: k* = 4.5348, c* = 1.2924, y* = 1.6552, saving rate = 0.219
saddle-path initial consumption c0 = 0.965075 (74.7% of c*)
capital closes half of the gap after 9 years
  t =  0: k/k* = 0.500, c/c* = 0.747, saving rate = 0.265
  t =  5: k/k* = 0.662, c/c* = 0.839, saving rate = 0.249
  t = 10: k/k* = 0.775, c/c* = 0.896, saving rate = 0.238
  t = 20: k/k* = 0.903, c/c* = 0.957, saving rate = 0.227
  t = 40: k/k* = 0.982, c/c* = 0.992, saving rate = 0.221
```

The steady-state saving rate is about 22%. Shooting finds the unique initial consumption (about 75% of its steady-state
level) that puts the economy on the saddle path; any other choice leads to capital exhaustion or oversaving. Capital closes
half the gap in about 9 years. Along the path, the saving rate **falls** from 26.5% toward 22%: with \(\sigma=2\), poor
households save more while the return to capital is high. Unlike in Solow, the saving rate is an outcome, not a parameter.

</details>

### Exercise 3.2 — Is the U.S. saving too much or too little? The golden rule

*Data:* `macrodata` — U.S. gross investment as a share of GDP; capital stock from the perpetual-inventory method of Lecture 2.

In the Solow model, steady-state consumption is maximized when the marginal product of capital net of depreciation equals the growth rate, \(\alpha Y/K-\delta=n+g\) — equivalently, when the saving rate equals the capital share \(\alpha\).

1. Compute the U.S. average investment share and the average capital–output ratio from the perpetual-inventory capital stock.
2. Compute the implied net marginal product of capital \(\alpha Y/K-\delta\) (annual) and compare it with average real GDP growth.
3. Is the U.S. dynamically efficient (below the golden rule)? What would happen to steady-state consumption if the saving rate rose toward \(\alpha\)?

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy()
# TODO: perpetual inventory (delta = 0.025 per quarter); s = mean(I/Y); K/Y; MPK
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl_q, a = 0.025, 1/3
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl_q)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl_q) * K[t] + I[t]
s = np.mean(I / Y); KY = np.mean(K[40:] / (4 * Y[40:]))          # annual K/Y, skipping the initialization decade
mpk = a / KY - 4 * dl_q
g = 4 * np.mean(np.diff(np.log(Y)))
print(f"investment share s = {s:.3f};  golden-rule saving rate = alpha = {a:.3f}")
print(f"capital-output ratio {KY:.2f} (annual);  net MPK = alpha*Y/K - delta = {100*mpk:.1f}%;  GDP growth = {100*g:.1f}%")
print(f"dynamically efficient (net MPK > growth): {mpk > g}")
n_g = g; ky_gold = a / (n_g + 4 * dl_q)
print(f"golden-rule K/Y would be {ky_gold:.2f} versus {KY:.2f} today")
```

Output:

```text
investment share s = 0.133;  golden-rule saving rate = alpha = 0.333
capital-output ratio 0.99 (annual);  net MPK = alpha*Y/K - delta = 23.7%;  GDP growth = 3.1%
dynamically efficient (net MPK > growth): True
golden-rule K/Y would be 2.54 versus 0.99 today
```

The U.S. investment share (13%) is far below the capital share (1/3), and the net return to capital comfortably exceeds the
growth rate, so the U.S. is **dynamically efficient**: it is below the golden rule, and higher saving would eventually raise
steady-state consumption, after a period of lower consumption. The levels here depend on the crude capital measure: with a
10% annual depreciation rate, the capital–output ratio is about 1, much lower than the official U.S. figure of about 3,
which includes structures with low depreciation. A larger \(K/Y\) lowers the net MPK, but it remains above the growth rate
(Abel et al., 1989).

</details>

## Further reading
Acemoglu, *Introduction to Modern Economic Growth*; Barro and Sala-i-Martin, *Economic Growth*.
