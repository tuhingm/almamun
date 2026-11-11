# Lecture 03 — Functional Form, Partial Effects, and Omitted-Variable Bias

## Learning objectives
By the end of this lecture you should be able to:

1. interpret coefficients in level, log and mixed specifications exactly, not only approximately;
2. compute partial effects, turning points and their standard errors in models with polynomials and interactions;
3. derive the omitted-variable bias formula in finite samples and in the limit, and sign the bias;
4. derive attenuation bias from classical measurement error, and explain why "bad controls" bias regression coefficients;
5. use specification tests (RESET) and sensitivity analysis responsibly.

## Motivation
Lecture 1 showed that OLS estimates a linear projection. In applied work we choose *which* projection: which transformations of the variables, which interactions, which controls. Each choice changes what the coefficient means. This lecture is about two failure modes: reading a coefficient as something it is not (an interpretation error), and leaving out a variable that matters (an identification error). The omitted-variable bias formula is the most useful single equation in applied econometrics, because it turns a vague worry ("maybe ability matters") into a signed, quantitative argument.

## 1. Partial effects
For a model \(Y=g(X;\beta)+u\) with \(E[u\mid X]=0\), the **partial effect** of a continuous regressor \(X_j\) is
\[
\theta_j(x)=\frac{\partial E[Y\mid X=x]}{\partial x_j}.
\]
In the linear model it is constant and equal to \(\beta_j\). With nonlinear terms it depends on \(x\), and we report either the partial effect at a representative value (PEA) or the **average partial effect** \(APE_j=E[\theta_j(X)]\), estimated by \(n^{-1}\sum_i\theta_j(X_i;\widehat\beta)\).

## 2. Logarithms
| Specification | Interpretation of \(\beta_1\) |
|---|---|
| \(Y=\beta_0+\beta_1x\) | \(\Delta Y=\beta_1\Delta x\) |
| \(\log Y=\beta_0+\beta_1x\) | \(100\beta_1\) ≈ % change in \(Y\) per unit of \(x\) (semi-elasticity) |
| \(Y=\beta_0+\beta_1\log x\) | \(\beta_1/100\) ≈ change in \(Y\) per 1% change in \(x\) |
| \(\log Y=\beta_0+\beta_1\log x\) | \(\beta_1\) = elasticity of \(Y\) with respect to \(x\) |

**Exact percentage effects.** In the log-level model, a one-unit change in \(x\) changes \(Y\) by exactly \(100(e^{\beta_1}-1)\)%. The approximation \(100\beta_1\) is accurate for \(|\beta_1|<0.1\). For a dummy variable \(D\) in \(\log Y=\beta_0+\delta D+\dots\), the percentage difference is \(100(e^{\delta}-1)\). With \(\delta=0.4\), that is 49%, not 40%.

**Retransformation.** A model for \(E[\log Y\mid X]\) is not a model for \(E[Y\mid X]\). By Jensen's inequality, \(\exp(X'\widehat\beta)\) underpredicts \(E[Y\mid X]\). If \(u\) is independent of \(X\), \(E[Y\mid X]=\exp(X'\beta)E[e^u]\), and Duan's smearing estimator \(n^{-1}\sum_ie^{\widehat u_i}\) estimates the correction. If \(u\) is heteroskedastic, log-linearized OLS is inconsistent even for elasticities. Then estimate \(E[Y\mid X]=\exp(X'\beta)\) directly by Poisson pseudo-maximum likelihood (Santos Silva and Tenreyro, 2006), which also handles zeros in \(Y\).

## 3. Polynomials and interactions
**Quadratics.** In \(Y=\beta_0+\beta_1x+\beta_2x^2+u\), the partial effect is \(\beta_1+2\beta_2x\). If \(\beta_2<0\), it turns negative at \(x^*=-\beta_1/(2\beta_2)\). A turning point is only meaningful if it lies well inside the support of the data. Its standard error follows from the delta method with gradient
\[
\nabla x^*=\Big(-\frac1{2\beta_2},\;\frac{\beta_1}{2\beta_2^2}\Big).
\]

**Interactions.** In \(Y=\beta_0+\beta_1x+\beta_2z+\beta_3xz+u\), the effect of \(x\) is \(\beta_1+\beta_3z\). Hence:

- \(\beta_1\) is the effect of \(x\) **at \(z=0\)**, which may be outside the data. Centering \(z\) at its mean makes \(\beta_1\) the effect at the average \(z\).
- A test of \(\beta_1=0\) is not a test of "no effect of \(x\)". The hypothesis \(H_0:\beta_1=\beta_3=0\) is.
- The standard error of the effect at \(z_0\) is \(\sqrt{\operatorname{Var}(\widehat\beta_1)+z_0^2\operatorname{Var}(\widehat\beta_3)+2z_0\operatorname{Cov}(\widehat\beta_1,\widehat\beta_3)}\).

**Dummy variables.** With an intercept and \(G\) mutually exclusive categories, include \(G-1\) dummies; including all \(G\) creates perfect collinearity (the dummy-variable trap). Each coefficient is a difference relative to the omitted base category. A full interaction of a group dummy with all regressors is equivalent to estimating separate regressions by group. The Chow test is the \(F\) test of the interaction coefficients.

## 4. Omitted-variable bias
Suppose the **long** regression is \(Y=X_1'\beta_1+X_2'\beta_2+u\) with \(E[Xu]=0\), but we estimate the **short** regression of \(Y\) on \(X_1\) only.

<div class="theorem" markdown="1">
**Theorem 3.1 (Omitted-variable bias).** Let \(\widehat\beta_1^{S}=(X_1'X_1)^{-1}X_1'y\). Then
\[
\widehat\beta_1^{S}=\widehat\beta_1+\widehat\Gamma\widehat\beta_2,\qquad \widehat\Gamma=(X_1'X_1)^{-1}X_1'X_2,
\]
an exact algebraic identity, where \(\widehat\beta_1,\widehat\beta_2\) are the long-regression OLS estimates. In the limit,
\[
\widehat\beta_1^{S}\to_p\beta_1+\Gamma\beta_2,\qquad \Gamma=E[X_1X_1']^{-1}E[X_1X_2'] .
\]
</div>

<div class="proof" markdown="1">
*Proof.* Substitute the long-regression decomposition \(y=X_1\widehat\beta_1+X_2\widehat\beta_2+\widehat u\) into \(\widehat\beta^S_1\) and use \(X_1'\widehat u=0\):
\[
\widehat\beta_1^{S}=(X_1'X_1)^{-1}X_1'(X_1\widehat\beta_1+X_2\widehat\beta_2+\widehat u)=\widehat\beta_1+\widehat\Gamma\widehat\beta_2 .
\]
The probability limit follows from the WLLN, the CMT and consistency of the long-regression estimates. ∎
</div>

\(\Gamma\) collects the coefficients of the **auxiliary regressions** of each omitted variable on the included ones. With one included and one omitted regressor (plus intercepts), the bias is
\[
\operatorname{bias}=\beta_2\cdot\frac{\operatorname{Cov}(x_1,x_2)}{\operatorname{Var}(x_1)} .
\]

| | \(\operatorname{Cov}(x_1,x_2)>0\) | \(\operatorname{Cov}(x_1,x_2)<0\) |
|---|---|---|
| \(\beta_2>0\) | upward bias | downward bias |
| \(\beta_2<0\) | downward bias | upward bias |

**Two facts that are often missed.**

1. There is no bias if the omitted variable is irrelevant (\(\beta_2=0\)) **or** uncorrelated with the included regressors (\(\Gamma=0\)). Randomized experiments work through the second channel.
2. Adding a control changes the coefficient of interest by exactly \(\widehat\Gamma\widehat\beta_2\). Coefficient movement when controls are added is therefore informative about selection on observables. Oster (2019) formalizes how far one can extrapolate from it to selection on unobservables.

## 5. Measurement error
**Classical error in the regressor.** Let the true model be \(Y=\beta_0+\beta_1x^*+u\), but we observe \(x=x^*+v\), with \(v\) independent of \(x^*\) and \(u\), mean zero, and variance \(\sigma_v^2\). Then \(Y=\beta_0+\beta_1x+(u-\beta_1v)\), and the composite error is correlated with \(x\). OLS satisfies
\[
\widehat\beta_1\to_p\beta_1\cdot\frac{\sigma^2_{x^*}}{\sigma^2_{x^*}+\sigma^2_v}=\beta_1\lambda,\qquad 0<\lambda<1 .
\]
The factor \(\lambda\) is the **reliability ratio**: the estimate is attenuated toward zero. With other controls, the attenuation worsens because the controls absorb part of the true signal in \(x^*\) but none of the noise: replace \(\sigma^2_{x^*}\) by the residual variance of \(x^*\) after partialling out the controls. Fixed-effects and first-difference estimators can therefore amplify attenuation (Lecture 7).

**Error in the dependent variable.** If \(y=y^*+w\) with \(w\) uncorrelated with \(X\), OLS remains consistent; only precision falls. **Non-classical** error, for example misreporting correlated with the true value, can bias in either direction.

**Remedies.** Use an instrument such as a second, independent measurement (Lecture 5), a validation sample, or bounds based on assumed reliability ratios.

## 6. Bad controls
More controls are not always better. A **bad control** is a variable that is itself an outcome of the regressor of interest. Example: when estimating the effect of a college degree on earnings, controlling for occupation conditions on a channel through which the degree works. It also introduces selection bias, because people in the same occupation with and without degrees differ in unobserved ways. In the language of causal graphs, conditioning on a mediator removes part of the effect, and conditioning on a collider opens a spurious path. **Rule:** good controls are determined *before* the treatment (Angrist and Pischke, 2009, §3.2.3).

## 7. Specification tests
**RESET (Ramsey, 1969).** Add \(\widehat y^2,\widehat y^3\) to the regression and test their joint significance with a robust Wald test. Rejection indicates that the conditional mean is nonlinear in a way the specification misses. RESET is not a test for omitted variables in general: an omitted variable that is linearly related to the included ones leaves no trace in the fitted values.

**Model choice.** Between non-nested specifications (for example, level versus log of \(Y\)), \(R^2\) values are not comparable when the dependent variables differ. Compare on a common scale, use out-of-sample prediction error, or use economic theory. A specification search conducted on the same data used for inference invalidates nominal p-values. Report the search, or hold out data.

## Worked example
**Ability bias in returns to schooling.** Let the long model be \(\log w=\beta_0+\beta_1S+\beta_2A+u\), where \(A\) is standardized ability. Suppose \(\beta_2=0.10\) (one SD of ability raises wages by about 10%), and the auxiliary regression of \(A\) on \(S\) has slope \(\gamma=0.20\) (each year of schooling is associated with 0.2 SD more ability). The short regression then gives
\[
\operatorname{plim}\widehat\beta_1^{S}=\beta_1+\gamma\beta_2=\beta_1+0.02 .
\]
If the short regression returns 0.10, the ability-adjusted return is about 0.08: the omission inflates the return by two percentage points.

Now add classical measurement error in schooling with reliability \(\lambda=0.9\). Attenuation pulls the estimate down by about 10%, partly offsetting the upward ability bias. This tension is exactly why IV estimates of the return to schooling (Card, 1999) are often *larger* than OLS: they correct measurement error as well as ability bias. They may also estimate a local average effect for compliers (Lecture 5).

## Problem set
1. **Exact versus approximate.** In \(\log w=1.2+0.35\,union+\dots\), what is the exact union wage premium? When is the approximation acceptable?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(100(e^{0.35}-1)=41.9\%\), against the approximate 35%. The approximation is acceptable when \(|\beta|\lesssim0.1\), because \(e^{\beta}-1=\beta+\beta^2/2+\dots\).
    </details>

2. **Turning point inference.** With \(\widehat\beta_1=0.040\), \(\widehat\beta_2=-0.0008\) on experience and experience squared, compute \(x^*\). Derive its delta-method variance in terms of the \(2\times2\) covariance matrix of \((\widehat\beta_1,\widehat\beta_2)\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(x^*=0.040/(2\times0.0008)=25\) years. With \(g=-\beta_1/(2\beta_2)\), \(\nabla g=(-1/(2\beta_2),\,\beta_1/(2\beta_2^2))=(625,\,31{,}250)\), and \(\operatorname{Var}(\widehat x^*)\approx\nabla g\,\widehat V\,\nabla g'\).
    </details>

3. **OVB with multiple regressors.** Show that with two included regressors \((x_1,x_2)\) and one omitted \(x_3\), the bias in \(\widehat\beta_1\) is \(\beta_3\delta_1\), where \(\delta_1\) is the coefficient on \(x_1\) in the regression of \(x_3\) on \((1,x_1,x_2)\). Explain why the sign of the bias cannot be read from \(\operatorname{Cov}(x_1,x_3)\) alone.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Apply Theorem 3.1 with \(X_1=(1,x_1,x_2)\) and \(X_2=x_3\); \(\Gamma\) is the vector of auxiliary-regression coefficients, whose \(x_1\) entry is \(\delta_1\). By FWL, \(\delta_1\) is the partial association of \(x_3\) with \(x_1\) given \(x_2\), which can have the opposite sign from the simple covariance.
    </details>

4. **Attenuation with controls.** Prove that, with a correctly measured control \(w\), \(\operatorname{plim}\widehat\beta_1=\beta_1\,\sigma^2_{r}/(\sigma^2_{r}+\sigma^2_v)\), where \(r\) is the residual from projecting \(x^*\) on \((1,w)\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    By FWL, \(\widehat\beta_1\) is the slope from regressing \(y\) on the residual of \(x=x^*+v\) after projecting on \((1,w)\). Since \(v\) is independent of \(w\), that residual is \(r+v\) asymptotically. Applying the one-regressor attenuation result to \(r+v\) gives the formula. Because \(\sigma^2_r\le\sigma^2_{x^*}\), attenuation is weakly worse.
    </details>

5. **Bad control.** A study regresses health on a hospital-visit dummy, controlling for "number of prescriptions received". Explain why the control is problematic.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Prescriptions are an outcome of the visit, a mediator. Controlling for them removes part of the visit's effect and compares visitors and non-visitors with the same prescriptions, who differ systematically in unobserved health.
    </details>

## Computational lab
**Goal:** reproduce the OVB identity exactly and quantify attenuation by simulation.

```python
import numpy as np
rng = np.random.default_rng(3)
n = 5_000
A = rng.normal(size=n)                              # ability
S = 12 + 2 * (0.2 * A + rng.normal(size=n))         # schooling, correlated with ability
logw = 0.5 + 0.08 * S + 0.10 * A + rng.normal(0, 0.3, n)

def ols(y, *cols):
    X = np.column_stack([np.ones(len(y)), *cols])
    return np.linalg.lstsq(X, y, rcond=None)[0]

long_ = ols(logw, S, A)
short = ols(logw, S)
gamma = ols(A, S)
print(short[1], long_[1] + gamma[1] * long_[2])      # identical: the OVB identity

S_obs = S + rng.normal(0, 1, n)                     # add classical measurement error
lam = S.var() / S_obs.var()
print(ols(logw, S_obs, A)[1], "reliability ratio", round(lam, 3))
```

Tasks: (i) verify the identity to machine precision; (ii) compute the theoretical attenuation factor with ability controlled, using the residual variance of \(S\) after projecting on \(A\), and compare with the simulation; (iii) add a "bad control", such as occupation generated from \(S\) and \(A\), and document how \(\widehat\beta_S\) moves. In R, `lm()` with `margins` or `marginaleffects` computes APEs and delta-method standard errors.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 3.1 — Ability bias in the return to schooling, with IQ as a proxy

*Data:* `wage2` — 935 men, NLS 1980; `IQ` is an IQ test score.

1. Estimate the short regression of `log(wage)` on `educ`, `exper`, `tenure`, `married`, `south`, `urban`, `black`.
2. Add `IQ` (the long regression). How much does the return to schooling fall?
3. Verify the omitted-variable bias identity \(\widehat\beta^S_{educ}=\widehat\beta_{educ}+\widehat\gamma\,\widehat\beta_{IQ}\) exactly, where \(\widehat\gamma\) is the coefficient on `educ` in the auxiliary regression of `IQ` on all the short-regression regressors.
4. Express the bias as a percentage of the short-regression estimate.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage2")
ctrl = "exper + tenure + married + south + urban + black"
short = smf.ols(f"np.log(wage) ~ educ + {ctrl}", data=df).fit()
# TODO: long regression adds IQ; auxiliary regression IQ ~ educ + ctrl
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage2")
ctrl = "exper + tenure + married + south + urban + black"
short = smf.ols(f"np.log(wage) ~ educ + {ctrl}", data=df).fit()
long_ = smf.ols(f"np.log(wage) ~ educ + IQ + {ctrl}", data=df).fit()
aux = smf.ols(f"IQ ~ educ + {ctrl}", data=df).fit()
bS, bL, bIQ, gam = short.params["educ"], long_.params["educ"], long_.params["IQ"], aux.params["educ"]
print(f"short-regression return to educ: {bS:.5f}")
print(f"long-regression return to educ:  {bL:.5f}")
print(f"IQ coefficient {bIQ:.5f};  auxiliary slope of IQ on educ {gam:.4f}")
print(f"bL + gamma * bIQ = {bL + gam * bIQ:.5f}   (identity holds: {abs(bL + gam*bIQ - bS) < 1e-10})")
print(f"omitted-variable bias = {gam * bIQ:.5f}, i.e. {100 * gam * bIQ / bS:.1f}% of the short estimate")
```

Output:

```text
short-regression return to educ: 0.06543
long-regression return to educ:  0.05441
IQ coefficient 0.00356;  auxiliary slope of IQ on educ 3.0963
bL + gamma * bIQ = 0.06543   (identity holds: True)
omitted-variable bias = 0.01102, i.e. 16.8% of the short estimate
```

Each year of schooling is associated with a few more IQ points, and IQ raises wages, so omitting IQ biases the return
to schooling **upward**. The identity holds to machine precision, as Theorem 3.1 says it must. Controlling for IQ lowers
the return from about 6.5% to 5.4% per year — the bias is about one sixth of the short estimate. IQ is only a proxy for ability: if it measures ability with error, part of the bias
remains, and the true ability-adjusted return could be lower still. Remember the offsetting attenuation bias from
measurement error in schooling (Lecture 3, worked example).

</details>

### Exercise 3.2 — Exact percentage effects, interactions and RESET

*Data:* `wage1` — 526 workers, CPS 1976.

1. Estimate `log(wage)` on `female`, `educ`, `exper`, `exper²`, `tenure`. Report the **exact** percentage gender gap \(100(e^{\delta}-1)\) with a delta-method standard error.
2. Add `female × educ`. Does the return to education differ by gender? What is the gender gap at 12 and at 16 years of schooling?
3. Run a RESET test (add \(\widehat y^2\) and \(\widehat y^3\)) for the model of part 1, using robust inference.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
base = "np.log(wage) ~ female + educ + exper + I(exper**2) + tenure"
r1 = smf.ols(base, data=df).fit(cov_type="HC1")
# TODO: exact gap 100*(exp(d)-1); interaction model; RESET with fitted values squared and cubed
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("wage1")
base = "np.log(wage) ~ female + educ + exper + I(exper**2) + tenure"
r1 = smf.ols(base, data=df).fit(cov_type="HC1")
d, sd = r1.params["female"], r1.bse["female"]
print(f"female coefficient {d:.4f}: approx gap {100*d:.1f}%, exact gap {100*(np.exp(d)-1):.1f}% (se {100*np.exp(d)*sd:.1f} pp)")

r2 = smf.ols(base + " + female:educ", data=df).fit(cov_type="HC1")
print(f"female:educ = {r2.params['female:educ']:.4f} (robust se {r2.bse['female:educ']:.4f}, p = {r2.pvalues['female:educ']:.3f})")
for e in (12, 16):
    t = r2.t_test(f"female + {e}*female:educ = 0")
    print(f"  log gap at educ = {e}: {float(np.squeeze(t.effect)):.4f} (se {float(np.squeeze(t.sd)):.4f})")

df["yh"] = smf.ols(base, data=df).fit().fittedvalues
r3 = smf.ols(base + " + I(yh**2) + I(yh**3)", data=df).fit(cov_type="HC1")
w = r3.wald_test("I(yh ** 2) = 0, I(yh ** 3) = 0", scalar=True)
print(f"RESET (robust): chi2(2) = {w.statistic:.3f}, p = {w.pvalue:.3f}")
```

Output:

```text
female coefficient -0.2979: approx gap -29.8%, exact gap -25.8% (se 2.7 pp)
female:educ = -0.0040 (robust se 0.0148, p = 0.787)
  log gap at educ = 12: -0.2958 (se 0.0367)
  log gap at educ = 16: -0.3118 (se 0.0662)
RESET (robust): chi2(2) = 25.470, p = 0.000
```

The exact gap is a few points smaller in absolute value than the approximate one, because \(e^{\delta}-1>\delta\) for
negative \(\delta\). The interaction is small and statistically insignificant: in 1976 the return to education was
similar for women and men, so the gender gap is roughly constant across schooling levels. The \(t\)-test on `female`
alone in the interaction model would be the gap at **zero** years of education — not a meaningful quantity. RESET **rejects** strongly
(p < 0.001): the specification misses some nonlinearity — candidates include a quadratic in tenure or interactions of
experience with education. RESET signals that the conditional mean is misspecified; it does not say how.

</details>

## Reading
**Required:** Wooldridge, *Introductory Econometrics*, chapters 3.3, 6 and 7; Angrist and Pischke, *Mostly Harmless Econometrics*, chapter 3.2.
**Further:** Card (1999), "The causal effect of education on earnings", *Handbook of Labor Economics* vol. 3A; Santos Silva and Tenreyro (2006), "The log of gravity", *Review of Economics and Statistics* 88(4); Oster (2019), "Unobservable selection and coefficient stability", *Journal of Business & Economic Statistics* 37(2); Ramsey (1969), "Tests for specification errors in classical linear least-squares regression analysis", *JRSS-B* 31(2).
