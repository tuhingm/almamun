# Lecture 13 — HANK Models: Distribution Dynamics and Aggregate Transmission

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Understand how household heterogeneity changes aggregate policy transmission; formulate continuous-time HJB and Kolmogorov forward equations; interpret sequence-space methods.

## 1. Heterogeneous-agent New Keynesian structure
HANK models combine nominal rigidities and monetary policy with incomplete-market households. Aggregate demand depends on the distribution of wealth and income because households have different marginal propensities to consume (MPCs). Transfers and interest-rate changes therefore have distribution-dependent effects.

## 2. Continuous-time household problem
For liquid assets \(a\), idiosyncratic state \(z\), and value function \(V(a,z)\), a stationary HJB equation takes the form
\[
\rho V(a,z)=\max_c\left\{u(c)+V_a(a,z)[wz+ra-c]\right\}+\mathcal L_zV,
\]
subject to the borrowing constraint. \(\mathcal L_z\) is the generator of the idiosyncratic income process.

## 3. Distribution dynamics
The Kolmogorov forward equation evolves the density \(g_t(a,z)\):
\[
\partial_t g_t=-\partial_a[\dot a(a,z)g_t]+\mathcal L_z^*g_t.
\]
The adjoint operator \(\mathcal L_z^*\) governs probability flows across idiosyncratic states. Stationarity requires zero net change in the distribution.

## 4. General equilibrium and policy
The household block determines consumption and asset supply; firms and nominal rigidities determine production and inflation; monetary and fiscal rules close the model. A transfer shock can have a large demand effect when directed toward high-MPC households, even if aggregate transfer expenditure is unchanged.

## 5. Sequence-space Jacobians
Linear response can be represented by Jacobians mapping aggregate paths (interest rates, wages, transfers) into paths of consumption, assets, and prices. Combining household and firm Jacobians with market-clearing conditions yields aggregate impulse responses. Explain the discretization and horizon underlying every computed Jacobian.

## 6. Computational lab
Solve a stationary heterogeneous-household block, compute MPCs by wealth percentile, and simulate a temporary transfer. Compare equal-per-capita and targeted transfers with the same aggregate budget. Report distributional and aggregate outcomes.

## Problem set
1. Interpret the HJB equation.
2. Explain the forward equation and its adjoint operator.
3. Why does the wealth distribution matter for aggregate demand?
4. Describe the sequence-space Jacobian workflow.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 13.1 — Who spends a transfer? MPCs across the wealth distribution

*Data:* The stationary Aiyagari distribution from Lecture 12's exercise (re-solved here so the code is self-contained).

1. Re-solve the Aiyagari household and its stationary distribution (same calibration).
2. Compute the average MPC by wealth quintile.
3. Compare the aggregate first-year spending response to a uniform \$1 transfer to everyone with a transfer **targeted** at the poorest 20% of households by wealth. By how much does targeting raise the spending response per dollar?

```python
# Reuse the EGM and distribution code of Lecture 12, Exercise 12.1, then:
# TODO: sort grid points by assets, assign quintiles by the cumulative distribution, average the MPC within each
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
b, s, r, rho, sig, n = 0.96, 2.0, 0.03, 0.9, 0.2, 7
p = (1 + rho) / 2; P = np.array([[p, 1 - p], [1 - p, p]])
for m_ in range(3, n + 1):
    Pn = np.zeros((m_, m_)); Pn[:-1, :-1] += p * P; Pn[:-1, 1:] += (1 - p) * P; Pn[1:, :-1] += (1 - p) * P; Pn[1:, 1:] += p * P
    Pn[1:-1] /= 2; P = Pn
psi = sig / np.sqrt(1 - rho**2) * np.sqrt(n - 1); z = np.exp(np.linspace(-psi, psi, n)); pi = np.ones(n) / n
for _ in range(2000): pi = pi @ P
z = z / (pi @ z); a = 50 * np.linspace(0, 1, 500)**2
C = (1 + r) * a[:, None] + z[None, :]
for it in range(5000):
    c_endo = (b * (1 + r) * (C**(-s) @ P.T))**(-1 / s); a_endo = (c_endo + a[:, None] - z[None, :]) / (1 + r)
    Cn = np.empty_like(C)
    for j in range(n):
        Cn[:, j] = np.interp(a, a_endo[:, j], c_endo[:, j]); bind = a < a_endo[0, j]; Cn[bind, j] = (1 + r) * a[bind] + z[j]
    if np.max(np.abs(Cn - C)) < 1e-9: break
    C = Cn
ap = (1 + r) * a[:, None] + z[None, :] - C
idx = np.clip(np.searchsorted(a, ap) - 1, 0, len(a) - 2); w = np.clip((a[idx + 1] - ap) / (a[idx + 1] - a[idx]), 0, 1)
D = np.ones((len(a), n)) / (len(a) * n)
for _ in range(5000):
    Dn = np.zeros_like(D)
    for j in range(n):
        np.add.at(Dn[:, j], idx[:, j], w[:, j] * D[:, j]); np.add.at(Dn[:, j], idx[:, j] + 1, (1 - w[:, j]) * D[:, j])
    Dn = Dn @ P
    if np.max(np.abs(Dn - D)) < 1e-12: break
    D = Dn
mpc = np.vstack([(C[1:] - C[:-1]) / ((1 + r) * (a[1:] - a[:-1]))[:, None], ((C[-1] - C[-2]) / ((1 + r) * (a[-1] - a[-2])))[None, :]])
wa = D.sum(1); cum = np.cumsum(wa)
q = np.minimum((cum / cum[-1] * 5).astype(int), 4)
q_mass = np.array([wa[q == k].sum() for k in range(5)])
for k in range(5):
    sel = q == k
    print(f"wealth quintile {k+1}: assets {a[sel].min():6.2f}-{a[sel].max():6.2f}, mass {q_mass[k]:.3f}, average MPC {np.sum(D[sel] * mpc[sel]) / D[sel].sum():.3f}")
uni = np.sum(D * mpc)
poor = q == 0
tgt = np.sum(D[poor] * mpc[poor]) / D[poor].sum()
print(f"spending per $1 transferred: uniform {uni:.3f}, targeted to the poorest-wealth group {tgt:.3f} ({tgt/uni:.1f}x)")
```

Output:

```text
wealth quintile 1: assets   0.00-  1.59, mass 0.198, average MPC 0.199
wealth quintile 2: assets   1.63-  4.11, mass 0.199, average MPC 0.059
wealth quintile 3: assets   4.16-  7.40, mass 0.199, average MPC 0.048
wealth quintile 4: assets   7.48- 12.55, mass 0.201, average MPC 0.042
wealth quintile 5: assets  12.65- 50.00, mass 0.203, average MPC 0.038
spending per $1 transferred: uniform 0.077, targeted to the poorest-wealth group 0.199 (2.6x)
```

MPCs fall steeply with wealth: about 0.20 in the poorest wealth quintile and about 0.04 at the top. A transfer targeted at the
poorest fifth generates about 2.6 times as much first-year spending per dollar as a uniform transfer. In a HANK model this
"distributional channel" matters for fiscal and monetary transmission. This one-asset model still understates MPCs — the
empirical average is about 0.2–0.3 per quarter for windfalls (Parker et al., 2013) — because it has too few constrained
households, as the previous lecture's exercise showed.

</details>

### Exercise 13.2 — A TANK multiplier with a data-based hand-to-mouth share

*Data:* `401ksubs` — the share of hand-to-mouth households from Lecture 12's exercise.

In a two-agent New Keynesian (TANK) model with a share \(\lambda\) of hand-to-mouth households who consume their income,
and Ricardian households who smooth consumption, a deficit-financed transfer \(T\) given equally to all households has a
first-round output effect of approximately \(\Delta Y=\frac{\lambda}{1-\lambda\,\mu}T\) under a fixed real rate, where
\(\mu\) is the share of income that HtM households receive relative to their population share (set \(\mu=1\) for simplicity) —
a simple "Keynesian cross" among HtM households.

1. Compute the HtM share \(\lambda\) from the data (half-month-of-income rule) and from the stricter "non-positive wealth" rule.
2. Compute the transfer multiplier \(\Delta Y/T\) for each, and for \(\lambda\in\{0,0.1,0.2,0.3,0.4\}\).
3. Compare with the Campbell–Mankiw \(\lambda\) estimated in Lecture 1's exercise. Why do HANK models with *wealthy* hand-to-mouth households give larger multipliers than this simple TANK calculation?

```python
import numpy as np, wooldridge as wd
d = wd.data("401ksubs")
lam1 = (d["nettfa"] < d["inc"] / 24).mean(); lam2 = (d["nettfa"] <= 0).mean()
mult = lambda lam: lam / (1 - lam)
# TODO: print multipliers
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
d = wd.data("401ksubs")
cases = {"half-month-of-income rule": (d["nettfa"] < d["inc"] / 24).mean(), "non-positive wealth rule": (d["nettfa"] <= 0).mean()}
mult = lambda lam: lam / (1 - lam)
for lab, lam in cases.items():
    print(f"{lab:26s}: lambda = {lam:.3f} -> transfer multiplier {mult(lam):.3f}")
for lam in (0.0, 0.1, 0.2, 0.3, 0.4):
    print(f"   lambda = {lam:.1f}: multiplier {mult(lam):.3f}")
```

Output:

```text
half-month-of-income rule : lambda = 0.464 -> transfer multiplier 0.865
non-positive wealth rule  : lambda = 0.350 -> transfer multiplier 0.538
   lambda = 0.0: multiplier 0.000
   lambda = 0.1: multiplier 0.111
   lambda = 0.2: multiplier 0.250
   lambda = 0.3: multiplier 0.429
   lambda = 0.4: multiplier 0.667
```

With the half-month-of-income rule (\(\lambda\approx0.46\)), the stylized transfer multiplier is about 0.87; with the stricter rule
(\(\lambda\approx0.35\)), about 0.54. The multiplier is convex in \(\lambda\) because hand-to-mouth households spend the income
created by each other's spending. Campbell and Mankiw's \(\lambda\) (Lecture 1) points to a similar range. HANK models produce
larger and more persistent effects than this calculation because many households are only **partly** constrained, MPCs are
high for several quarters, and the "intertemporal Keynesian cross" works through future income too (Auclert, Rognlie and
Straub, 2024).

</details>

## Further reading
Kaplan, Moll and Violante (2018); Auclert et al. (2021); Achdou et al. (2022).
