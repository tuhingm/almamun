# Lecture 01 — The Dynamic Household: Intertemporal Choice and Euler Equations

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive household optimality conditions in discrete time; distinguish budget feasibility from optimality; interpret the Euler equation, borrowing constraints, and transversality.

## 1. Environment
A representative household chooses consumption \(c_t\) and next-period assets \(a_{t+1}\) to maximize
\[
\max_{\{c_t,a_{t+1}\}} E_0\sum_{t=0}^{\infty}\beta^t u(c_t),\quad 0<\beta<1,
\]
subject to
\[
c_t+a_{t+1}=w_t n_t+(1+r_t)a_t+T_t.
\]
Here \(\beta\) is the discount factor, \(r_t\) the real interest rate, \(w_t\) the real wage, and \(T_t\) net transfers.

## 2. Euler equation
Let \(\lambda_t\) be the multiplier on the period budget. The Lagrangian yields \(u'(c_t)=\lambda_t\) and \(\lambda_t=\beta E_t[(1+r_{t+1})\lambda_{t+1}]\). Hence
\[
u'(c_t)=\beta E_t[(1+r_{t+1})u'(c_{t+1})].
\]
With CRRA utility \(u(c)=c^{1-\sigma}/(1-\sigma)\), deterministic consumption growth satisfies
\[
\frac{c_{t+1}}{c_t}=\left[\beta(1+r_{t+1})\right]^{1/\sigma}.
\]
Under uncertainty, the expectation applies to marginal utility times the gross return; it cannot generally be separated into products of expectations.

## 3. Lifetime budget and transversality
Iterating the flow budget forward gives the present-value lifetime resource constraint, subject to a terminal condition. A standard transversality condition is
\[
\lim_{T\to\infty}E_0[\beta^T u'(c_T)a_{T+1}]=0.
\]
It rules out plans inconsistent with household optimality and the lifetime resource constraint.

## 4. Borrowing constraints
If \(a_{t+1}\geq\underline a\), the Euler condition becomes an inequality at a binding lower bound. With multiplier \(\mu_t\geq0\),
\[
u'(c_t)=\beta E_t[(1+r_{t+1})u'(c_{t+1})]+\mu_t,\quad
\mu_t(a_{t+1}-\underline a)=0.
\]
A constrained household may consume less today than the unconstrained Euler equation prescribes.

## 5. Worked example
For log utility, deterministic constant return \(r\), and no labor income after date zero, the Euler equation gives \(c_{t+1}/c_t=\beta(1+r)\). The lifetime budget and terminal condition determine the consumption level.

## 6. Computational lab
Implement a finite-horizon household problem. Discretize assets, solve by backward induction, and compare policy functions with and without a borrowing limit. Report grid convergence, Euler-equation errors, and the share of states at the constraint.

## Problem set
1. Derive the Euler equation from the Lagrangian.
2. Derive the CRRA consumption-growth equation.
3. State the transversality condition and explain its economic role.
4. Show how a binding borrowing constraint modifies the Euler condition.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 1.1 — Estimating the elasticity of intertemporal substitution from the Euler equation

*Data:* `macrodata` — U.S. quarterly real consumption `realcons` and ex-post real interest rate `realint`, 1959–2009.

The log-linearized Euler equation is \(\Delta\log c_{t+1}=\text{const}+\psi\,r_t+\varepsilon_{t+1}\), where \(\psi\) is the
elasticity of intertemporal substitution (EIS). The realized real rate is measured with error and correlated with the
expectational error, so use instrumental variables (Hall, 1988).

1. Estimate \(\psi\) by OLS.
2. Estimate it by 2SLS, instrumenting \(r_t\) with variables dated \(t-2\) and \(t-3\) (lagged real rate, inflation and consumption growth). Report the first-stage \(F\).
3. What does the estimate imply for the coefficient of relative risk aversion if preferences are CRRA (\(\gamma=1/\psi\))?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 400 * np.log(m["realcons"]).diff().shift(-1), "r": m["realint"], "infl": m["infl"]})
# TODO: lags dated t-2, t-3; IV2SLS.from_formula("dc ~ 1 + [r ~ r_l2 + r_l3 + infl_l2 + dc_l2]", d)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 400 * np.log(m["realcons"]).diff().shift(-1), "r": m["realint"], "infl": m["infl"]})
d["dc0"] = 400 * np.log(m["realcons"]).diff()
for v in ("r", "infl", "dc0"):
    for k in (2, 3):
        d[f"{v}_l{k}"] = d[v].shift(k)
d = d.dropna()
ols = smf.ols("dc ~ r", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
iv = IV2SLS.from_formula("dc ~ 1 + [r ~ r_l2 + r_l3 + infl_l2 + infl_l3 + dc0_l2 + dc0_l3]", data=d).fit(cov_type="kernel")
fs = smf.ols("r ~ r_l2 + r_l3 + infl_l2 + infl_l3 + dc0_l2 + dc0_l3", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test("r_l2 = 0, r_l3 = 0, infl_l2 = 0, infl_l3 = 0, dc0_l2 = 0, dc0_l3 = 0", scalar=True).statistic / 6
print(f"OLS  EIS = {ols.params['r']:.3f} (HAC se {ols.bse['r']:.3f})")
print(f"2SLS EIS = {iv.params['r']:.3f} (HAC se {iv.std_errors['r']:.3f});  first-stage F = {F:.1f}")
lo, hi = iv.params["r"] - 1.96 * iv.std_errors["r"], iv.params["r"] + 1.96 * iv.std_errors["r"]
print(f"95% CI for the EIS: [{lo:.3f}, {hi:.3f}]")
```

Output:

```text
OLS  EIS = 0.225 (HAC se 0.102)
2SLS EIS = 0.261 (HAC se 0.179);  first-stage F = 13.1
95% CI for the EIS: [-0.089, 0.611]
```

OLS and 2SLS both give an EIS of about 0.2–0.3, and the 2SLS confidence interval includes zero. This is Hall's (1988)
classic finding: aggregate consumption growth responds little to the real interest rate, so the EIS is small (and, under
CRRA, risk aversion \(1/\psi\) would be large, around 4 or more). The first-stage \(F\) of about 13 is only moderately
strong. Instruments dated \(t-2\) and earlier are used because time aggregation of quarterly consumption makes \(t-1\)
instruments invalid. Many macro models assume an EIS near 0.5–1, which is hard to reconcile with this evidence.

</details>

### Exercise 1.2 — Is consumption a random walk? The Campbell–Mankiw test

*Data:* `macrodata` — U.S. quarterly real consumption and real disposable income.

Hall's (1978) permanent-income hypothesis implies that consumption growth is unpredictable. Campbell and Mankiw (1989)
nest it in \(\Delta\log c_t=\mu+\lambda\,\Delta\log y_t+\varepsilon_t\), where \(\lambda\) is the share of income going to
"rule-of-thumb" (hand-to-mouth) consumers.

1. Estimate \(\lambda\) by OLS and by 2SLS, using lags 2–4 of income growth, consumption growth and the change in the T-bill rate as instruments.
2. Test \(\lambda=0\). What does the estimate say about the importance of hand-to-mouth households — a central ingredient of TANK and HANK models (Lectures 12–13)?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 100 * np.log(m["realcons"]).diff(), "dy": 100 * np.log(m["realdpi"]).diff(), "di": m["tbilrate"].diff()})
# TODO: lags 2-4 as instruments; IV2SLS
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
m = sm.datasets.macrodata.load_pandas().data
d = pd.DataFrame({"dc": 100 * np.log(m["realcons"]).diff(), "dy": 100 * np.log(m["realdpi"]).diff(), "di": m["tbilrate"].diff()})
z = []
for v in ("dy", "dc", "di"):
    for k in (2, 3, 4):
        d[f"{v}_l{k}"] = d[v].shift(k); z.append(f"{v}_l{k}")
d = d.dropna()
ols = smf.ols("dc ~ dy", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
iv = IV2SLS.from_formula(f"dc ~ 1 + [dy ~ {' + '.join(z)}]", data=d).fit(cov_type="kernel")
fs = smf.ols(f"dy ~ {' + '.join(z)}", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
F = fs.wald_test(", ".join(f"{v} = 0" for v in z), scalar=True).statistic / len(z)
print(f"OLS  lambda = {ols.params['dy']:.3f} (se {ols.bse['dy']:.3f})")
print(f"2SLS lambda = {iv.params['dy']:.3f} (se {iv.std_errors['dy']:.3f}), t = {iv.tstats['dy']:.2f};  first-stage F = {F:.1f}")
print(f"overidentification (Sargan) p-value: {iv.sargan.pval:.3f}")
```

Output:

```text
OLS  lambda = 0.342 (se 0.075)
2SLS lambda = 0.648 (se 0.154), t = 4.21;  first-stage F = 4.2
overidentification (Sargan) p-value: 0.004
```

The 2SLS estimate of \(\lambda\) is about 0.65 and strongly significant, so consumption growth is far from unpredictable:
predictable income growth passes into consumption. Taken at face value, around half or more of income goes to
rule-of-thumb consumers — Campbell and Mankiw (1989) found about 0.5. Two warnings: the first-stage \(F\) of about 4 signals
**weak instruments** (Lecture 5 of Econometrics), so the 2SLS point estimate is unreliable and weak-IV-robust inference is
needed; and the over-identification test rejects, so the instruments disagree. The qualitative conclusion — a sizable
hand-to-mouth share — is what motivates TANK and HANK models.

</details>

## Further reading
Ljungqvist and Sargent, *Recursive Macroeconomic Theory*; Deaton, *Understanding Consumption*.
