# Lecture 02 — OLS Inference: Sampling Distributions, Hypothesis Tests, and Confidence Intervals

## Learning objectives
By the end of this lecture you should be able to:

1. state the convergence concepts and limit theorems used in econometrics (WLLN, CLT, continuous mapping, Slutsky, delta method);
2. prove consistency and asymptotic normality of OLS for the linear projection model;
3. derive the exact finite-sample distribution of \(t\) and \(F\) statistics in the normal regression model;
4. construct and interpret asymptotic \(t\) tests, Wald tests, and confidence intervals, including for nonlinear functions of \(\beta\);
5. explain size, power, and the duality between tests and confidence sets.

## Motivation
An estimate without a standard error is not a result. Lecture 1 characterized what OLS estimates; this lecture asks how precisely. There are two routes to inference. The *exact* route assumes normal errors and delivers \(t\) and \(F\) distributions that hold for every \(n\). The *asymptotic* route drops normality and homoskedasticity and relies on large-sample approximations. Modern applied work uses the asymptotic route almost exclusively, but the exact theory explains where the familiar formulas and degrees-of-freedom corrections come from.

## 1. Tools from asymptotic theory
Let \(Z_n\) be a sequence of random vectors.

<div class="definition" markdown="1">
**Definition 2.1.** \(Z_n\) **converges in probability** to \(c\), written \(Z_n\to_p c\), if \(P(\|Z_n-c\|>\varepsilon)\to0\) for every \(\varepsilon>0\). \(Z_n\) **converges in distribution** to \(Z\), written \(Z_n\to_d Z\), if \(P(Z_n\le z)\to P(Z\le z)\) at every continuity point \(z\) of the distribution of \(Z\).
</div>

The four workhorse results:

- **Weak law of large numbers (WLLN).** If \(W_i\) are i.i.d. with \(E\|W_i\|<\infty\), then \(\bar W_n\to_pE[W_i]\).
- **Central limit theorem (CLT).** If \(W_i\) are i.i.d. with \(E\|W_i\|^2<\infty\), then \(\sqrt n(\bar W_n-E[W_i])\to_dN(0,\operatorname{Var}(W_i))\).
- **Continuous mapping theorem (CMT).** If \(Z_n\to_pc\) (or \(\to_dZ\)) and \(g\) is continuous at \(c\) (or on the support of \(Z\)), then \(g(Z_n)\to_pg(c)\) (or \(\to_dg(Z)\)).
- **Slutsky's theorem.** If \(Z_n\to_dZ\) and \(A_n\to_pA\) with \(A\) constant, then \(A_nZ_n\to_dAZ\) and \(Z_n+A_n\to_dZ+A\).

<div class="theorem" markdown="1">
**Theorem 2.1 (Delta method).** If \(\sqrt n(\widehat\theta-\theta)\to_dN(0,V)\) and \(g:\mathbb R^k\to\mathbb R^q\) is continuously differentiable at \(\theta\) with Jacobian \(G=\partial g(\theta)/\partial\theta'\), then
\[
\sqrt n\big(g(\widehat\theta)-g(\theta)\big)\to_dN(0,\,GVG').
\]
</div>

<div class="proof" markdown="1">
*Proof sketch.* By the mean value theorem, row by row, \(g(\widehat\theta)-g(\theta)=G(\bar\theta)(\widehat\theta-\theta)\) for some \(\bar\theta\) between \(\widehat\theta\) and \(\theta\). Since \(\widehat\theta\to_p\theta\), \(G(\bar\theta)\to_pG\) by the CMT. Slutsky then gives the result. ∎
</div>

## 2. Consistency of OLS
Work in the projection model of Lecture 1: \(Y_i=X_i'\beta+u_i\) with \(E[X_iu_i]=0\), i.i.d. sampling, and \(Q_{XX}=E[X_iX_i']\) positive definite.

<div class="theorem" markdown="1">
**Theorem 2.2 (Consistency).** Under these assumptions and \(E\|X_i\|^2<\infty\), \(E[Y_i^2]<\infty\): \(\widehat\beta\to_p\beta\).
</div>

<div class="proof" markdown="1">
*Proof.* Write
\[
\widehat\beta-\beta=\Big(\tfrac1n\textstyle\sum_iX_iX_i'\Big)^{-1}\tfrac1n\textstyle\sum_iX_iu_i .
\]
By the WLLN, \(\tfrac1n\sum X_iX_i'\to_pQ_{XX}\) and \(\tfrac1n\sum X_iu_i\to_pE[X_iu_i]=0\). Matrix inversion is continuous at the positive definite matrix \(Q_{XX}\), so by the CMT \(\widehat\beta-\beta\to_pQ_{XX}^{-1}\cdot0=0\). ∎
</div>

Consistency requires only the projection condition. It does **not** require \(E[u\mid X]=0\), homoskedasticity, or normality. It fails when \(E[X_iu_i]\neq0\): omitted variables, simultaneity and measurement error (Lectures 3, 5, 6).

## 3. Asymptotic normality
Add finite fourth moments. Define \(\Omega=E[X_iX_i'u_i^2]\).

<div class="theorem" markdown="1">
**Theorem 2.3 (Asymptotic normality).** If in addition \(E\|X_i\|^4<\infty\) and \(E[Y_i^4]<\infty\), then
\[
\sqrt n(\widehat\beta-\beta)\to_dN(0,V_\beta),\qquad V_\beta=Q_{XX}^{-1}\,\Omega\,Q_{XX}^{-1}.
\]
</div>

<div class="proof" markdown="1">
*Proof.* \(\sqrt n(\widehat\beta-\beta)=\big(\tfrac1n\sum X_iX_i'\big)^{-1}\tfrac1{\sqrt n}\sum X_iu_i\). The vectors \(X_iu_i\) are i.i.d., mean zero, with variance \(\Omega\) (finite by Cauchy–Schwarz and the fourth-moment conditions). The CLT gives \(\tfrac1{\sqrt n}\sum X_iu_i\to_dN(0,\Omega)\). Combine with \(\big(\tfrac1n\sum X_iX_i'\big)^{-1}\to_pQ_{XX}^{-1}\) using Slutsky. ∎
</div>

The **sandwich** form \(Q^{-1}\Omega Q^{-1}\) is the general case. Under conditional homoskedasticity, \(E[u_i^2\mid X_i]=\sigma^2\), the LIE gives \(\Omega=\sigma^2Q_{XX}\), and the variance collapses to \(V^0_\beta=\sigma^2Q_{XX}^{-1}\). Homoskedasticity is a strong assumption with no special economic justification. The default in this course is the sandwich.

**Variance estimators.** Replace population moments by sample moments:
\[
\widehat V^0_\beta=s^2\widehat Q_{XX}^{-1},\qquad
\widehat V_\beta^{HC0}=\widehat Q_{XX}^{-1}\Big(\tfrac1n\textstyle\sum_iX_iX_i'\widehat u_i^2\Big)\widehat Q_{XX}^{-1}.
\]
Both are consistent under their respective assumptions. Standard errors for \(\widehat\beta_j\) are \(se(\widehat\beta_j)=\sqrt{[\widehat V_\beta]_{jj}/n}\). Finite-sample refinements (HC1–HC3) are the subject of Lecture 4.

## 4. Exact inference in the normal regression model
Now assume A1–A5 of Lecture 1 plus normality: \(u\mid X\sim N(0,\sigma^2I_n)\). Then, conditional on \(X\),
\[
\widehat\beta\mid X\sim N\big(\beta,\sigma^2(X'X)^{-1}\big),\qquad \frac{(n-k)s^2}{\sigma^2}=\frac{u'M_Xu}{\sigma^2}\sim\chi^2_{n-k},
\]
and \(\widehat\beta\) is independent of \(s^2\). The chi-square result holds because \(M_X\) is idempotent with rank \(n-k\). Independence holds because \(\widehat\beta-\beta=(X'X)^{-1}X'u\) and \(\widehat u=M_Xu\) are jointly normal with covariance \(\sigma^2(X'X)^{-1}X'M_X=0\).

<div class="theorem" markdown="1">
**Theorem 2.4 (Exact \(t\) and \(F\)).** Under the normal regression model:

1. \(T_j=\dfrac{\widehat\beta_j-\beta_j}{s\sqrt{[(X'X)^{-1}]_{jj}}}\sim t_{n-k}\);
2. for \(H_0:R\beta=r\) with \(R\) a \(q\times k\) matrix of rank \(q\),
\[
F=\frac{(R\widehat\beta-r)'\big[R(X'X)^{-1}R'\big]^{-1}(R\widehat\beta-r)}{q\,s^2}\sim F_{q,\,n-k}.
\]
</div>

<div class="proof" markdown="1">
*Proof.* (1) The numerator over \(\sigma\sqrt{[(X'X)^{-1}]_{jj}}\) is \(N(0,1)\); dividing by \(\sqrt{s^2/\sigma^2}=\sqrt{\chi^2_{n-k}/(n-k)}\), which is independent of it, gives a \(t_{n-k}\). (2) Under \(H_0\), \(R\widehat\beta-r\sim N(0,\sigma^2R(X'X)^{-1}R')\), so the quadratic form divided by \(\sigma^2\) is \(\chi^2_q\). Divide by \(q\) and by the independent \(\chi^2_{n-k}/(n-k)\). ∎
</div>

The \(F\) statistic has an equivalent and very useful **restricted-versus-unrestricted** form:
\[
F=\frac{(SSR_r-SSR_u)/q}{SSR_u/(n-k)},
\]
where \(SSR_r\) is the residual sum of squares from least squares imposing \(R\beta=r\).

## 5. Asymptotic tests
Without normality, the exact distributions are unavailable, but the asymptotic ones are simple.

**\(t\) test.** For \(H_0:\beta_j=\beta_j^0\), \(T_j=(\widehat\beta_j-\beta_j^0)/se(\widehat\beta_j)\to_dN(0,1)\) under \(H_0\), by Theorem 2.3, consistency of \(\widehat V_\beta\), and Slutsky. Reject at level \(\alpha\) when \(|T_j|>z_{1-\alpha/2}\).

**Wald test.** For \(H_0:R\beta=r\),
\[
W=n\,(R\widehat\beta-r)'\big(R\widehat V_\beta R'\big)^{-1}(R\widehat\beta-r)\to_d\chi^2_q .
\]
Under homoskedasticity with \(\widehat V^0_\beta\), \(W=qF\), so for large \(n\) the \(F\) test and the Wald test agree. Using \(F_{q,n-k}\) critical values with the robust \(W/q\) is a common, mildly conservative small-sample convention.

**Nonlinear restrictions.** For \(H_0:g(\beta)=0\), apply the delta method:
\[
W=n\,g(\widehat\beta)'\big(\widehat G\widehat V_\beta\widehat G'\big)^{-1}g(\widehat\beta)\to_d\chi^2_q,\qquad \widehat G=\frac{\partial g(\widehat\beta)}{\partial\beta'} .
\]
Wald tests of nonlinear hypotheses are **not invariant** to how the restriction is written: \(\beta_1/\beta_2=1\) and \(\beta_1-\beta_2=0\) define the same null but can give different statistics in finite samples. Prefer the algebraically simplest form.

**Likelihood ratio and score (LM) tests.** Under normality the LR statistic is \(n\log(SSR_r/SSR_u)\). The LM statistic is computed from the restricted model only, as \(nR^2\) from regressing restricted residuals on all regressors. Wald, LR and LM are asymptotically equivalent under the null and local alternatives; they differ in finite samples and in what must be estimated.

## 6. Confidence intervals and the duality with tests
A \(1-\alpha\) confidence interval for \(\beta_j\) is the set of values not rejected by a level-\(\alpha\) test:
\[
CI_{1-\alpha}=\big[\widehat\beta_j-z_{1-\alpha/2}\,se(\widehat\beta_j),\;\widehat\beta_j+z_{1-\alpha/2}\,se(\widehat\beta_j)\big].
\]
**Interpretation.** Coverage is a property of the *procedure*: in repeated samples, intervals constructed this way contain the fixed true \(\beta_j\) with probability approaching \(1-\alpha\). A computed interval either does or does not contain \(\beta_j\); it is not a probability statement about \(\beta_j\).

**Joint confidence regions** for \(q\) coefficients are ellipsoids \(\{b: n(R\widehat\beta-b)'(R\widehat V_\beta R')^{-1}(R\widehat\beta-b)\le\chi^2_{q,1-\alpha}\}\). Two individual 95% intervals do not form a 95% joint region.

## 7. Size, power, and what a p-value is
- The **size** of a test is its rejection probability under \(H_0\). An asymptotic test has correct size in the limit; in finite samples the actual size can differ, sometimes badly (heavy tails, high leverage, few clusters, weak instruments).
- **Power** is the rejection probability under an alternative. For a fixed alternative \(\beta_j\neq\beta_j^0\), \(|T_j|\to_p\infty\), so a consistent test has power approaching one. To study power meaningfully, consider **local alternatives** \(\beta_j=\beta_j^0+h/\sqrt n\): then \(T_j\to_dN(h/\sigma_j,1)\), and power depends on the signal-to-noise ratio \(h/\sigma_j\).
- A **p-value** is the probability, computed under \(H_0\), of a test statistic at least as extreme as the one observed. It is not the probability that \(H_0\) is true, and a small p-value does not imply an economically important effect.

## Worked example
**Testing constant returns to scale.** Estimate a Cobb–Douglas production function \(\log Q_i=\beta_0+\beta_1\log K_i+\beta_2\log L_i+u_i\) with \(n=500\) plants. Suppose \(\widehat\beta_1=0.31\), \(\widehat\beta_2=0.74\), robust variances \(\widehat{\operatorname{Var}}(\widehat\beta_1)=0.0016\), \(\widehat{\operatorname{Var}}(\widehat\beta_2)=0.0025\), and covariance \(-0.0010\) (all already divided by \(n\)).

Test \(H_0:\beta_1+\beta_2=1\). With \(R=(0,1,1)\) and \(r=1\):
\[
R\widehat\beta-r=0.05,\qquad \widehat{\operatorname{Var}}(\widehat\beta_1+\widehat\beta_2)=0.0016+0.0025-2(0.0010)=0.0021 .
\]
Then \(T=0.05/\sqrt{0.0021}=0.05/0.0458=1.09\), so constant returns are not rejected at 5%. The 95% interval for returns to scale is \(1.05\pm1.96\times0.0458=[0.96,\,1.14]\). The negative covariance matters: ignoring it would give \(se=\sqrt{0.0041}=0.064\) and a much wider, incorrect interval.

**Reparameterization trick.** Defining \(\theta=\beta_1+\beta_2\) and estimating \(\log Q=\beta_0+\theta\log K+\beta_2(\log L-\log K)+u\) delivers \(\widehat\theta\) and its standard error directly from one regression.

## Problem set
1. **Consistency without unbiasedness.** In the model of Lecture 1, Problem 1, show that OLS is consistent for the BLP coefficients even though it is conditionally biased.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Theorem 2.2 only uses \(E[X_iu_i]=0\), which holds for the projection error by construction, together with moment conditions that hold because \(X\) is bounded and \(\varepsilon\) has finite variance. Conditional bias is irrelevant for the probability limit.
    </details>

2. **Homoskedastic simplification.** Show that \(E[u_i^2\mid X_i]=\sigma^2\) implies \(\Omega=\sigma^2Q_{XX}\). Give an economic example in which this assumption is implausible.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\Omega=E[X_iX_i'E(u_i^2\mid X_i)]=\sigma^2E[X_iX_i']\). Example: in a regression of household consumption on income, the dispersion of consumption is much larger among high-income households, so \(E[u^2\mid income]\) increases with income.
    </details>

3. **Delta method.** Let \(\theta=\beta_1/\beta_2\) (for instance, a willingness-to-pay ratio). Derive the asymptotic variance of \(\widehat\theta\) and explain what happens when \(\beta_2\) is close to zero.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(G=(1/\beta_2,\,-\beta_1/\beta_2^2)\), so \(\operatorname{avar}(\widehat\theta)=V_{11}/\beta_2^2-2\beta_1V_{12}/\beta_2^3+\beta_1^2V_{22}/\beta_2^4\). As \(\beta_2\to0\), the gradient explodes and the normal approximation becomes unreliable: the ratio has heavy tails. Use Fieller intervals, obtained by inverting the test of \(\beta_1-\theta_0\beta_2=0\), instead.
    </details>

4. **\(F\) from sums of squares.** Prove that the Wald form and the \(SSR_r-SSR_u\) form of the \(F\) statistic are identical under homoskedastic OLS.

    <details class="solution" markdown="1"><summary>Solution</summary>

    The restricted estimator is \(\tilde\beta=\widehat\beta-(X'X)^{-1}R'[R(X'X)^{-1}R']^{-1}(R\widehat\beta-r)\). Then \(y-X\tilde\beta=\widehat u+X(\widehat\beta-\tilde\beta)\), and the cross term vanishes because \(X'\widehat u=0\). Hence \(SSR_r-SSR_u=(\widehat\beta-\tilde\beta)'X'X(\widehat\beta-\tilde\beta)=(R\widehat\beta-r)'[R(X'X)^{-1}R']^{-1}(R\widehat\beta-r)\).
    </details>

5. **Interpretation.** A coauthor writes: "The 95% CI is [0.2, 0.6], so there is a 95% probability that the effect lies between 0.2 and 0.6." Correct the statement.

    <details class="solution" markdown="1"><summary>Solution</summary>

    In repeated sampling, 95% of intervals built by this procedure contain the true effect. The realized interval either contains it or does not. A probability statement about the parameter requires a Bayesian posterior (Statistics, Lecture 6).
    </details>

## Computational lab
**Goal:** see the CLT at work and compare classical and robust inference by Monte Carlo.

```python
import numpy as np
rng = np.random.default_rng(2)

def one_draw(n, hetero=True):
    x = rng.exponential(1.0, n)
    u = rng.normal(0, 1, n) * (x if hetero else 1.0)   # Var(u|x) = x^2 when hetero
    y = 1.0 + 0.5 * x + u
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    V0 = XtX_inv * (e @ e) / (n - 2)                     # classical
    meat = (X * e[:, None]**2).T @ X
    V1 = XtX_inv @ meat @ XtX_inv * n / (n - 2)          # HC1
    t0 = (b[1] - 0.5) / np.sqrt(V0[1, 1])
    t1 = (b[1] - 0.5) / np.sqrt(V1[1, 1])
    return abs(t0) > 1.96, abs(t1) > 1.96

for n in (50, 200, 1000):
    rej = np.array([one_draw(n) for _ in range(5000)]).mean(axis=0)
    print(f"n={n:5d}  size classical={rej[0]:.3f}  size HC1={rej[1]:.3f}")
```

Tasks: (i) explain why the classical test over-rejects under heteroskedasticity; (ii) set `hetero=False` and confirm that both tests have size close to 5%; (iii) plot a histogram of the robust \(t\) statistic for \(n=50\) and compare its tails with \(N(0,1)\). In R, use `lm()` with `sandwich::vcovHC(type = "HC1")` and `lmtest::coeftest()`. In Stata, use `regress y x, vce(robust)`.

## Reading
**Required:** Hansen, *Econometrics*, chapters 5–7 and 9; Hayashi, *Econometrics*, sections 1.4 and 2.1–2.6.
**Further:** White (1980), "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity", *Econometrica* 48(4); Engle (1984), "Wald, likelihood ratio, and Lagrange multiplier tests in econometrics", *Handbook of Econometrics* vol. 2; Wasserstein and Lazar (2016), "The ASA statement on p-values", *The American Statistician* 70(2).
