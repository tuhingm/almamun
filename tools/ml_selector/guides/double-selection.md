title: Double-Selection LASSO: Inference on One Variable with Many Controls
short: Double selection
when: You want the effect of one variable and have dozens of possible controls, perhaps more than observations, and need valid standard errors after selecting among them
data: 90 countries, growth rate of GDP per capita and 60 country characteristics (Barro and Lee 1994; R package hdm GrowthData)
oes: dsel
tools: Python · OES engine · scikit-learn · NumPy
lead: Use this guide when the question is the coefficient on one variable (here: do poorer countries grow faster?) and there are many candidate controls. Putting all of them in the regression is impossible or noisy, and choosing them by LASSO for the outcome alone gives the wrong answer and invalid confidence intervals, because a control that is weakly related to the outcome but strongly to the treatment is dropped although it confounds. Post-double-selection (Belloni, Chernozhukov and Hansen 2014) selects controls for both the outcome and the treatment, takes the union and runs OLS. The guide reproduces the published convergence estimate and shows how it differs from the alternatives.
---

## The question and the method

**What is the effect of \(d\) on \(y\) when the right set of controls is unknown?** The model is

\[
y_i = \alpha d_i + x_i'\beta + \varepsilon_i,\qquad d_i = x_i'\gamma + v_i ,
\]

with \(p\) controls, possibly more than \(n\), of which only a few matter (approximate sparsity). Double selection:

1. LASSO of \(y\) on the controls: keep the selected set \(S_y\).
2. LASSO of \(d\) on the controls: keep \(S_d\).
3. OLS of \(y\) on \(d\) and the controls in \(S_y\cup S_d\), with heteroskedasticity-robust standard errors.

Step 2 is what protects the estimate. A control with a small coefficient \(\beta_j\) can be missed in step 1, but if it predicts \(d\) strongly its omission biases \(\hat\alpha\) by about \(\beta_j\gamma_j\), and step 2 catches it. The penalty is the **plug-in** value of the rigorous LASSO, \(\lambda = 2c\sqrt n\,\Phi^{-1}(1-\gamma/(2p))\) with \(c=1.1\), \(\gamma=0.1/\log n\), and heteroskedasticity-robust loadings, which the theory covers. A cross-validated penalty is not covered by the theory and selects many more controls.

<div class="read" markdown="1">
**Three conditions for reading double selection**

1. **Sparsity:** a few controls (or good approximations to them) explain \(y\) and \(d\). If the effects of the controls are dense, no selection method is reliable.
2. **Selection on observables:** the controls must contain the confounders. Selection does not create variables you did not measure.
3. **A treatment that the controls do not explain completely:** if \(d\) is almost a function of the controls, there is no variation left to identify \(\alpha\).
</div>

<div class="write" markdown="1">
**How to write it: the model**

We test conditional convergence: do countries with lower initial GDP grow faster, holding other characteristics fixed? The outcome is the growth rate of GDP per capita and the target is the logarithm of initial GDP (gdpsh465); there are 60 potential controls (education, institutions, trade, demographics, health) for 90 countries. We use post-double-selection (Belloni, Chernozhukov and Hansen 2014): the plug-in rigorous LASSO of the outcome and of initial GDP on the controls, then OLS of growth on initial GDP and the union of the selected controls, with heteroskedasticity-robust standard errors (the degrees-of-freedom convention of the R package hdm).
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("growth.csv").read()
df = pd.read_csv("growth.csv")
xs = [c for c in df.columns if c not in ("growth", "gdpsh465")]
print(len(df), "countries;", len(xs), "potential controls")
print(df[["growth", "gdpsh465"]].describe().round(3).T)
```

<div class="read" markdown="1">
**How to read it**

There are 90 countries and 60 candidate controls, so a regression with all controls has 62 coefficients to estimate from 90 observations. Growth averages 4.5% a year with a standard deviation of 5.1 points, and initial log GDP has a mean of 7.7 and a standard deviation of 0.9. With so few observations per control the OLS with all controls is imprecise, which is the case for selection.
</div>

## Estimate and compare

```python
spec = dict(y="growth", treat="gdpsh465", x=xs)
r = oes.run("dsel", data, spec)
oes.summary(r)
oes.show(r, "Effect of")
oes.show(r, "Selected controls")
```

<div class="read" markdown="1">
**How to read it**

Double selection gives -0.0500 (robust standard error 0.0158, 95% interval -0.081 to -0.019): holding the selected controls fixed, a country with one log point higher initial GDP grew 5 percentage points a year more slowly, which is evidence of conditional convergence. This reproduces the estimate published with the R package hdm (-0.05001, standard error 0.01579). Without controls the coefficient is 0.001 (standard error 0.005), the textbook finding of no unconditional convergence. With all 60 controls it is -0.009 with a standard error of 0.026, so the 60 controls take the precision away. Selecting controls for the outcome alone keeps one control (the black-market premium, bmp1l) and gives -0.011 (standard error 0.005, which would reject zero), while selecting for the treatment alone gives -0.044 (0.017).
</div>

## Why both selections matter

```python
sel = r["tables"][1]["rows"]
print(pd.DataFrame(sel, columns=["control", "outcome", "treatment", "final"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

The outcome equation selects only the black-market premium (bmp1l). The treatment equation selects six other controls: a trade-policy variable (freetar), three education variables (hm65, sf65, humanf65), life expectancy (lifee065) and a population-share variable (pop6565). The growth equation selects none of them, but each predicts initial GDP. Dropping them, as the outcome-only selection does, leaves their confounding in the coefficient: the estimate moves from -0.050 to -0.011. This is the bias that double selection removes, and it is not visible in the standard error of the single-selection estimate, which is small (0.005) and would mislead the reader about precision.
</div>

## Sensitivity to the penalty and to the set of controls

```python
rows = []
for c in (0.8, 1.1, 1.5, 2.0):
    rr = oes.run("dsel", data, dict(spec, c=c))
    a = [x for x in rr["tables"][0]["rows"] if x[0].startswith("Double selection (plug")][0]
    rows.append(["penalty c = " + str(c), a[1], a[2], a[5]])
rng = np.random.default_rng(0)
for k in range(4):
    sub = list(rng.choice(xs, 40, replace=False))
    rr = oes.run("dsel", data, dict(spec, x=sub))
    a = [x for x in rr["tables"][0]["rows"] if x[0].startswith("Double selection (plug")][0]
    rows.append([f"random 40 of the 60 controls, draw {k + 1}", a[1], a[2], a[5]])
print(pd.DataFrame(rows, columns=["specification", "estimate", "robust s.e.", "controls kept"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

A larger penalty keeps fewer controls: 10 at c = 0.8, 7 at 1.1 and 5 at 1.5 and 2.0. The estimate is -0.044 at 0.8, -0.050 at 1.1, -0.043 at 1.5 and -0.031 at 2.0, and the 95% interval stays below zero except at the largest penalty. The sign and rough magnitude are stable; the third decimal is not. With random subsets of 40 of the 60 controls the estimate is between -0.050 and -0.020 and the standard error between 0.0158 and 0.0168, so the conclusion does not depend on which controls were offered, but a precise value does, and a reader should be told so.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Without controls, initial GDP has no relation with subsequent growth (0.001, standard error 0.005). Controlling for country characteristics selected by post-double-selection, the coefficient is -0.050 (robust standard error 0.016, 95% confidence interval -0.081 to -0.019), which is evidence of conditional convergence: seven of 60 potential controls are selected, one for the growth equation and six for initial GDP. A regression with all 60 controls gives -0.009 (0.026) and is too imprecise to be informative with 90 countries. Choosing controls for growth alone gives -0.011, because it omits controls that predict initial GDP and that confound the relation. The point estimate varies between -0.031 and -0.050 with the penalty constant (0.8 to 2.0) and between -0.020 and -0.050 with random subsets of the candidate controls, and the interval excludes zero in most specifications. The result relies on sparsity and on the controls containing the relevant confounders.
</div>

| Report | What | Where |
|---|---|---|
| Method | Two selections, union, OLS | Step 1 |
| Data | Many controls, few countries | Step 2 |
| Estimates | Comparison with other approaches | Step 3 |
| Selection | Controls selected in each equation | Step 4 |
| Robustness | Penalty constant, control subsets | Step 5 |

**References.** Belloni, A., Chernozhukov, V. and Hansen, C. (2014), "Inference on Treatment Effects after Selection among High-Dimensional Controls," *Review of Economic Studies* 81. Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012), "Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain," *Econometrica* 80. Chernozhukov, V., Hansen, C. and Spindler, M. (2016), "hdm: High-Dimensional Metrics," *R Journal* 8. Barro, R. and Lee, J.-W. (1994), "Sources of Economic Growth," *Carnegie-Rochester Conference Series on Public Policy* 40. Leeb, H. and Pötscher, B. (2005), "Model Selection and Inference: Facts and Fiction," *Econometric Theory* 21.
