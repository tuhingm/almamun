# Lecture 07 — Experimental Design, Randomization, and Blocking

## Learning objectives
Define experimental units, treatments, outcomes, and estimands; explain randomization, blocking, replication, factorial and split-plot designs; connect assignment mechanisms to design-based inference.

## 1. Potential outcomes and estimands
For unit \(i\), define potential outcomes \(Y_i(1)\) and \(Y_i(0)\). The individual effect is \(\tau_i=Y_i(1)-Y_i(0)\), and the finite-population average effect is \(N^{-1}\sum_i\tau_i\). Only one potential outcome is observed per unit. The design specifies the unit of assignment, treatment, outcome, target population, and assignment mechanism.

## 2. Completely randomized design
Under complete random assignment, treatment status is independent of fixed potential outcomes over the assignment mechanism. The difference in treatment and control means estimates the finite-population average effect. Randomization supports unbiasedness over assignments but does not guarantee exact covariate balance in the realized sample.

## 3. Randomized blocks
When pre-treatment characteristics predict outcomes, randomize within blocks. A model representation is
\[
Y_{ij}=\mu+\tau_i+\beta_j+\varepsilon_{ij}.
\]
Blocking can reduce residual variance when blocks explain outcome variation. Define blocks before observing treatment outcomes.

## 4. Factorial designs
A two-factor design can be represented as
\[
Y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}.
\]
The interaction means the effect of one factor depends on the level of the other. Interpret main effects in light of the interaction and coding constraints.

## 5. Split-plot and nested designs
Difficult-to-randomize factors may be assigned at a whole-plot level, with other factors randomized within plots. Whole-plot and subplot errors differ; treating all observations as independently randomized gives incorrect standard errors. Nested designs likewise require covariance structures consistent with assignment and sampling.

## 6. Design-based inference
A randomization test reassigns treatment labels according to the original design. Under a sharp null of no effect for any unit, this can provide exact inference. For heterogeneous effects and average-effect targets, specify the estimand and variance estimator carefully.

## Worked example
If clinic strongly predicts the outcome, allocate treatment and control units within each clinic rather than randomizing without regard to clinic. The blocked contrast compares units within pre-specified strata and can improve precision.

## Problem set
1. Define the experimental unit and estimand in a cluster trial.
2. Explain why randomization does not ensure exact balance.
3. Derive a treatment contrast in a randomized block design.
4. Interpret a two-factor interaction.
5. Explain why split-plot designs require distinct error terms.

## Computational lab
Simulate complete and blocked randomization under strong block-level outcome heterogeneity. Compare estimator variance and conduct a randomization test under the sharp null.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 7.1 — Checking a randomized experiment: balance and randomization inference

*Data:* `jtrain2` — NSW experiment.

1. Build a balance table of pre-treatment covariates (`age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`, `re75`) by treatment arm, with standardized differences.
2. Run an omnibus test of balance: regress treatment on all covariates and test that all slopes are zero.
3. Compute a **randomization-inference** p-value for the effect on `re78` using the difference in means as the statistic (5,000 re-randomizations that keep 185 treated).

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("jtrain2")
X = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
# TODO: means by arm, standardized difference (m1 - m0) / sqrt((v1 + v0)/2); smf.ols("train ~ ...").f_test
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("jtrain2")
X = ["age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75"]
g = d.groupby("train")[X]
tab = pd.DataFrame({"control": g.mean().loc[0], "treated": g.mean().loc[1]})
tab["std diff"] = (tab["treated"] - tab["control"]) / np.sqrt((g.var().loc[0] + g.var().loc[1]) / 2)
print(tab.round(3).to_string())
r = smf.ols("train ~ " + " + ".join(X), data=d).fit(cov_type="HC1")
w = r.wald_test(", ".join(f"{x} = 0" for x in X), scalar=True)
print(f"\nomnibus balance test: chi2({len(X)}) = {w.statistic:.2f}, p = {w.pvalue:.3f}")
y, t = d["re78"].to_numpy(), d["train"].to_numpy().astype(bool)
obs = y[t].mean() - y[~t].mean(); rng = np.random.default_rng(11); n1 = t.sum(); hits = 0
for _ in range(5000):
    s = np.zeros(len(y), bool); s[rng.choice(len(y), n1, replace=False)] = True
    hits += abs(y[s].mean() - y[~s].mean()) >= abs(obs)
print(f"randomization-inference p-value for re78: {hits/5000:.4f}  (observed difference {obs:.3f})")
```

Output:

```text
control  treated  std diff
age        25.054   25.816     0.107
educ       10.088   10.346     0.141
black       0.827    0.843     0.044
hisp        0.108    0.059    -0.175
married     0.154    0.189     0.094
nodegree    0.835    0.708    -0.304
re74        2.107    2.096    -0.002
re75        1.267    1.532     0.084

omnibus balance test: chi2(8) = 17.99, p = 0.021
randomization-inference p-value for re78: 0.0042  (observed difference 1.794)
```

Most standardized differences are small, but `nodegree` differs by 0.30 SD and `hisp` by 0.18, and the omnibus test
rejects balance at 5% (p ≈ 0.02). In a genuinely randomized experiment, imbalance on some covariate happens by chance;
it is a reason to adjust for these covariates for precision, not evidence that randomization failed. The randomization
inference p-value (0.004) is valid by design, with no distributional assumptions.

</details>

### Exercise 7.2 — Does blocking help? A design simulation using real outcomes

*Data:* `jtrain2` controls (260 men) — their actual 1978 earnings serve as untreated potential outcomes.

Use the 260 control men as a "population" with known \(Y(0)\). Add a constant treatment effect of 1 (thousand dollars) to get \(Y(1)\).

1. Simulate 5,000 **completely randomized** experiments (130 treated) and record the difference-in-means estimate.
2. Simulate 5,000 **blocked** experiments, randomizing within blocks defined by 1975 earnings (zero vs positive) and `nodegree`, and estimate the effect as the block-size-weighted average of within-block differences.
3. Compare the bias and standard deviation of the two designs. Why does blocking help only if the blocking variables predict the outcome?

```python
import numpy as np, wooldridge as wd
c = wd.data("jtrain2").query("train == 0").reset_index(drop=True)
y0 = c["re78"].to_numpy(); y1 = y0 + 1.0
block = (c["re75"] > 0).astype(int) * 2 + c["nodegree"]
# TODO: simulate complete and blocked randomization
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
c = wd.data("jtrain2").query("train == 0").reset_index(drop=True)
y0 = c["re78"].to_numpy(); y1 = y0 + 1.0; n = len(y0)
block = ((c["re75"] > 0).astype(int) * 2 + c["nodegree"]).to_numpy()
rng = np.random.default_rng(5); cr, bl = [], []
for _ in range(5000):
    t = np.zeros(n, bool); t[rng.choice(n, n // 2, replace=False)] = True
    cr.append(y1[t].mean() - y0[~t].mean())
    est = 0.0
    for b in np.unique(block):
        idx = np.where(block == b)[0]; tb = np.zeros(len(idx), bool); tb[rng.choice(len(idx), len(idx) // 2, replace=False)] = True
        est += len(idx) / n * (y1[idx][tb].mean() - y0[idx][~tb].mean())
    bl.append(est)
for lab, v in (("complete randomization", cr), ("blocked randomization", bl)):
    print(f"{lab:23s}: mean estimate {np.mean(v):.3f} (true 1.000), SD {np.std(v):.3f}")
print(f"variance reduction from blocking: {1 - np.var(bl)/np.var(cr):.1%}")
print("block sizes:", np.bincount(block), " block means of Y(0):", [round(y0[block == b].mean(), 2) for b in range(4)])
```

Output:

```text
complete randomization : mean estimate 0.993 (true 1.000), SD 0.675
blocked randomization  : mean estimate 0.984 (true 1.000), SD 0.683
variance reduction from blocking: -2.2%
block sizes: [ 25 153  18  64]  block means of Y(0): [np.float64(4.62), np.float64(4.15), np.float64(5.19), np.float64(5.32)]
```

Both designs are unbiased. Here blocking does **not** reduce variance — in this simulation it is even slightly worse —
because the blocking variables barely predict 1978 earnings (block means range from 4.2 to 5.3, while the within-block SD
is about 5.5). Blocking pays off only when blocks are homogeneous in the outcome; blocking on weak predictors costs a
little precision through smaller within-block samples. In a real design, choose blocks by their predictive power for the
outcome, for example from pilot data.

</details>

## Reading
Montgomery, *Design and Analysis of Experiments*; Fisher, *The Design of Experiments*; Imbens and Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*.