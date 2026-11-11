# Lecture 01 — From Probability to the Linear Regression Model

## Learning objectives
By the end of this lecture you should be able to:

1. define the conditional expectation function (CEF) and prove that it is the best predictor of \(Y\) given \(X\) in mean square;
2. define the best linear predictor (linear projection), derive its coefficients, and explain how it relates to the CEF;
3. derive the OLS estimator in matrix form and interpret it as the sample analogue of the projection coefficient;
4. state and prove the algebraic properties of the projection matrices \(P_X\) and \(M_X\), including the Frisch–Waugh–Lovell theorem;
5. prove unbiasedness of OLS and the Gauss–Markov theorem, and state exactly which assumptions each result uses.

## Motivation
Almost every empirical paper in economics reports a regression. Before asking whether a regression coefficient is *causal*, we need to know what it *is*: what population object the estimator converges to, under what conditions, and with what precision. This lecture builds the regression model from probability alone. The key message is that a linear regression always estimates *something* well defined — the linear projection — whether or not the conditional mean is linear and whether or not the model has a causal interpretation. Assumptions are added only when we want more: unbiasedness, efficiency, or a structural interpretation.

## 1. The conditional expectation function
Let \((Y, X')'\) be a random vector with \(E[Y^2]<\infty\), where \(Y\) is scalar and \(X\) is \(k\times 1\). The **conditional expectation function** is \(m(x)=E[Y\mid X=x]\). Define the CEF error \(e=Y-m(X)\), so that
\[
Y = m(X) + e .
\]
This is a *decomposition*, not a model: it holds for any joint distribution with a finite first moment.

<div class="theorem" markdown="1">
**Theorem 1.1 (Properties of the CEF error).** Let \(e = Y - m(X)\). Then

1. \(E[e\mid X]=0\);
2. \(E[e]=0\);
3. \(E[h(X)\,e]=0\) for any function \(h\) with \(E|h(X)e|<\infty\).
</div>

<div class="proof" markdown="1">
*Proof.* (1) \(E[e\mid X]=E[Y\mid X]-m(X)=0\). (2) By the law of iterated expectations (LIE), \(E[e]=E\{E[e\mid X]\}=0\). (3) \(E[h(X)e]=E\{h(X)E[e\mid X]\}=0\), again by the LIE, because \(h(X)\) is known once we condition on \(X\). ∎
</div>

Property 3 says the CEF error is uncorrelated with *every* function of \(X\). This is much stronger than being uncorrelated with \(X\) itself.

<div class="theorem" markdown="1">
**Theorem 1.2 (The CEF is the best predictor).** If \(E[Y^2]<\infty\), then for every function \(g\) with \(E[g(X)^2]<\infty\),
\[
E\big[(Y-g(X))^2\big] \;\ge\; E\big[(Y-m(X))^2\big],
\]
with equality if and only if \(g(X)=m(X)\) almost surely.
</div>

<div class="proof" markdown="1">
*Proof.* Write \(Y-g(X) = e + (m(X)-g(X))\). Expanding the square,
\[
E[(Y-g(X))^2]=E[e^2]+2E\big[e\,(m(X)-g(X))\big]+E\big[(m(X)-g(X))^2\big].
\]
The cross term is zero by Theorem 1.1(3) with \(h=m-g\). The last term is non-negative and equals zero only when \(g(X)=m(X)\) a.s. ∎
</div>

**Variance decomposition.** Since \(m(X)\) and \(e\) are uncorrelated, \(\operatorname{Var}(Y)=\operatorname{Var}(m(X))+E[e^2]\), and \(E[e^2]=E[\operatorname{Var}(Y\mid X)]\). This is the law of total variance: total variation equals explained variation plus average conditional variance.

## 2. The best linear predictor
The CEF can be any function. In practice we often approximate it with a linear function \(X'\beta\). Define the **best linear predictor** (BLP) or **linear projection** as the minimizer of mean squared prediction error over linear functions:
\[
\beta = \arg\min_{b\in\mathbb R^k} S(b),\qquad S(b)=E\big[(Y-X'b)^2\big].
\]

<div class="theorem" markdown="1">
**Theorem 1.3 (Projection coefficient).** Assume \(E[Y^2]<\infty\), \(E\|X\|^2<\infty\), and \(Q_{XX}=E[XX']\) is positive definite. Then the unique minimizer is
\[
\beta = \big(E[XX']\big)^{-1}E[XY],
\]
and the projection error \(u=Y-X'\beta\) satisfies \(E[Xu]=0\).
</div>

<div class="proof" markdown="1">
*Proof.* \(S(b)=E[Y^2]-2b'E[XY]+b'E[XX']b\). This is a strictly convex quadratic in \(b\) because \(E[XX']\) is positive definite. The first-order condition \(-2E[XY]+2E[XX']b=0\) gives the formula. Substituting, \(E[Xu]=E[XY]-E[XX']\beta=0\). ∎
</div>

Three facts follow, and they are worth memorizing.

- **The BLP always exists** under moment conditions alone. No linearity of the CEF, no homoskedasticity, no normality and no causal story is required.
- **If \(X\) includes a constant**, then \(E[u]=0\) and \(\operatorname{Cov}(X_j,u)=0\) for every regressor \(j\).
- **\(E[Xu]=0\) is weaker than \(E[u\mid X]=0\).** The projection error is uncorrelated with \(X\) but may be correlated with nonlinear functions of \(X\). The two conditions coincide exactly when the CEF is linear: \(m(X)=X'\beta\).

**The BLP also approximates the CEF.** Because \(Y=m(X)+e\) with \(E[Xe]=0\),
\[
\beta=\arg\min_b E\big[(m(X)-X'b)^2\big],
\]
so the projection coefficient is the best linear approximation to the CEF itself. This is why regression remains informative when the CEF is nonlinear.

**Simple regression.** With \(X=(1,x)'\), solving the two normal equations gives the familiar formulas
\[
\beta_1=\frac{\operatorname{Cov}(x,Y)}{\operatorname{Var}(x)},\qquad \beta_0=E[Y]-\beta_1E[x].
\]

## 3. Linear regression models and their assumptions
The literature uses several different "linear regression models". They are nested, and it pays to keep them separate.

| Model | Defining assumption | What it buys |
|---|---|---|
| Linear projection | \(E[Xu]=0\) (by construction) | OLS is consistent for \(\beta\) |
| Linear CEF | \(E[u\mid X]=0\) | OLS is unbiased; \(X'\beta\) is the best predictor |
| Homoskedastic linear CEF | adds \(E[u^2\mid X]=\sigma^2\) | Gauss–Markov efficiency; simple variance formula |
| Normal regression | adds \(u\mid X\sim N(0,\sigma^2)\) | exact finite-sample \(t\) and \(F\) distributions |
| Structural (causal) model | \(\beta\) measures the effect of an intervention on \(X\) | economic interpretation; requires identification arguments (Lectures 5–6, 13) |

A common error is to treat a statement in the bottom row as if it followed from the rows above it. It does not: no amount of statistical structure turns a projection coefficient into a causal effect.

## 4. Ordinary least squares
Observe a random sample \(\{(Y_i,X_i)\}_{i=1}^n\). Stack the data as \(y=(Y_1,\dots,Y_n)'\) (\(n\times1\)) and \(X=(X_1,\dots,X_n)'\) (\(n\times k\)). OLS replaces population moments with sample moments:
\[
\widehat\beta=\Big(\frac1n\sum_{i=1}^n X_iX_i'\Big)^{-1}\frac1n\sum_{i=1}^n X_iY_i=(X'X)^{-1}X'y .
\]
Equivalently, \(\widehat\beta\) minimizes the sum of squared residuals \(SSR(b)=(y-Xb)'(y-Xb)\). The first-order conditions are the **normal equations**
\[
X'(y-X\widehat\beta)=0 \quad\Longleftrightarrow\quad X'\widehat u=0,
\]
the sample analogue of \(E[Xu]=0\). The estimator exists and is unique if and only if \(X\) has full column rank, \(\operatorname{rank}(X)=k\). Perfect multicollinearity, such as including a full set of dummies together with an intercept, violates this.

**Method-of-moments view.** OLS is the method-of-moments estimator for the moment condition \(E[X(Y-X'\beta)]=0\). This view generalizes directly to instrumental variables (Lecture 5) and GMM (Lecture 8).

## 5. Projection geometry
Define the \(n\times n\) matrices
\[
P_X = X(X'X)^{-1}X',\qquad M_X = I_n-P_X .
\]
Then fitted values are \(\widehat y=X\widehat\beta=P_Xy\) and residuals are \(\widehat u=y-\widehat y=M_Xy\).

<div class="theorem" markdown="1">
**Theorem 1.4 (Projection algebra).** \(P_X\) and \(M_X\) are symmetric and idempotent; \(P_XX=X\); \(M_XX=0\); \(P_XM_X=0\); \(\operatorname{tr}(P_X)=k\); and \(\operatorname{tr}(M_X)=n-k\).
</div>

<div class="proof" markdown="1">
*Proof.* Symmetry is immediate because \((X'X)^{-1}\) is symmetric. Idempotence: \(P_XP_X=X(X'X)^{-1}X'X(X'X)^{-1}X'=P_X\). Then \(M_XM_X=I-2P_X+P_X=M_X\), \(P_XX=X\) and \(M_XX=X-X=0\). For the traces, \(\operatorname{tr}(P_X)=\operatorname{tr}\big((X'X)^{-1}X'X\big)=\operatorname{tr}(I_k)=k\), using \(\operatorname{tr}(AB)=\operatorname{tr}(BA)\). ∎
</div>

Geometrically, \(P_X\) projects \(y\) orthogonally onto the column space of \(X\), and \(M_X\) projects it onto the orthogonal complement. The decomposition \(y=P_Xy+M_Xy\) is orthogonal, so
\[
y'y=\widehat y'\widehat y+\widehat u'\widehat u .
\]
When \(X\) contains a constant, the same holds after demeaning, giving \(SST=SSE+SSR\) and the coefficient of determination \(R^2=1-SSR/SST\in[0,1]\). \(R^2\) measures in-sample linear predictive fit. It is not a test of the model, and adding regressors can never lower it.

**Leverage.** The diagonal elements \(h_{ii}=X_i'(X'X)^{-1}X_i\) of \(P_X\) satisfy \(0\le h_{ii}\le1\) and \(\sum_ih_{ii}=k\). An observation with large leverage has unusual regressor values and pulls the fitted line toward itself. Leverage will matter again for robust variance estimators (Lecture 4).

## 6. Partitioned regression: the Frisch–Waugh–Lovell theorem
Partition \(X=[X_1\;X_2]\) and \(\widehat\beta=(\widehat\beta_1',\widehat\beta_2')'\). Let \(M_1=I-X_1(X_1'X_1)^{-1}X_1'\).

<div class="theorem" markdown="1">
**Theorem 1.5 (Frisch–Waugh–Lovell).** The OLS coefficient on \(X_2\) in the long regression of \(y\) on \([X_1\;X_2]\) equals
\[
\widehat\beta_2=(X_2'M_1X_2)^{-1}X_2'M_1y,
\]
the coefficient from regressing the residuals \(M_1y\) on the residuals \(M_1X_2\). The residuals of the two regressions are identical.
</div>

<div class="proof" markdown="1">
*Proof.* Start from \(y=X_1\widehat\beta_1+X_2\widehat\beta_2+\widehat u\) and premultiply by \(M_1\). Since \(M_1X_1=0\), and \(M_1\widehat u=\widehat u\) because \(X_1'\widehat u=0\),
\[
M_1y=M_1X_2\widehat\beta_2+\widehat u .
\]
Premultiply by \(X_2'\). Since \(X_2'\widehat u=0\), we get \(X_2'M_1y=X_2'M_1X_2\widehat\beta_2\). Solving gives the result. The same equation shows that the residuals of the short regression of \(M_1y\) on \(M_1X_2\) equal \(\widehat u\). ∎
</div>

FWL makes "holding other variables constant" precise: a multiple-regression coefficient measures the association between \(y\) and the part of \(X_2\) that is not linearly explained by \(X_1\). Applications appear throughout the course: demeaning with an intercept, detrending, and the within estimator for panel data (Lecture 7), which is FWL with individual dummies in \(X_1\).

## 7. Finite-sample properties
Substituting \(y=X\beta+u\) gives the **sampling-error representation**
\[
\widehat\beta-\beta=(X'X)^{-1}X'u .
\]
Every property of OLS is a property of this expression.

**Assumptions.** (A1) \(y=X\beta+u\). (A2) Random sampling. (A3) \(\operatorname{rank}(X)=k\). (A4) \(E[u\mid X]=0\). (A5) \(E[uu'\mid X]=\sigma^2I_n\).

<div class="theorem" markdown="1">
**Theorem 1.6 (Unbiasedness and variance).** Under A1–A4, \(E[\widehat\beta\mid X]=\beta\). Under A1–A5, \(\operatorname{Var}(\widehat\beta\mid X)=\sigma^2(X'X)^{-1}\).
</div>

<div class="proof" markdown="1">
*Proof.* \(E[\widehat\beta-\beta\mid X]=(X'X)^{-1}X'E[u\mid X]=0\). For the variance,
\[
\operatorname{Var}(\widehat\beta\mid X)=(X'X)^{-1}X'E[uu'\mid X]X(X'X)^{-1}=\sigma^2(X'X)^{-1}. \qquad ∎
\]
</div>

Note what A4 rules out: with only the projection condition \(E[Xu]=0\), OLS is generally **biased** in finite samples (but still consistent, Lecture 2).

<div class="theorem" markdown="1">
**Theorem 1.7 (Gauss–Markov).** Under A1–A5, OLS is the best linear unbiased estimator (BLUE): for any estimator \(\tilde\beta=Ay\) with \(A\) a function of \(X\) and \(E[\tilde\beta\mid X]=\beta\), the matrix \(\operatorname{Var}(\tilde\beta\mid X)-\operatorname{Var}(\widehat\beta\mid X)\) is positive semidefinite.
</div>

<div class="proof" markdown="1">
*Proof.* Unbiasedness for every \(\beta\) requires \(AX=I_k\). Write \(A=(X'X)^{-1}X'+D\). Then \(AX=I_k+DX\), so \(DX=0\). Hence
\[
\operatorname{Var}(\tilde\beta\mid X)=\sigma^2AA'=\sigma^2\big[(X'X)^{-1}+DD'\big],
\]
because the cross terms \((X'X)^{-1}X'D'\) vanish when \(DX=0\). The difference \(\sigma^2DD'\) is positive semidefinite. ∎
</div>

Gauss–Markov does **not** require normality. It does require homoskedasticity and no serial correlation; without A5, generalized least squares is more efficient and the OLS variance formula is wrong (Lecture 4). The theorem also restricts the comparison to *linear unbiased* estimators; biased or nonlinear estimators can have lower mean squared error.

**Estimating \(\sigma^2\).** Using \(\widehat u=M_Xu\) and the trace result,
\[
E[\widehat u'\widehat u\mid X]=E[u'M_Xu\mid X]=\sigma^2\operatorname{tr}(M_X)=\sigma^2(n-k),
\]
so \(s^2=\widehat u'\widehat u/(n-k)\) is unbiased for \(\sigma^2\). The divisor \(n-k\) is a degrees-of-freedom correction: \(k\) dimensions of \(u\) are "used up" in fitting \(\widehat\beta\).

## Worked example
**Returns to schooling, two ways.** Suppose \(\log w = \beta_0+\beta_1\,educ+\beta_2\,exper+u\). By FWL, \(\widehat\beta_1\) can be computed in two steps:

1. regress \(educ\) on \((1,exper)\) and keep the residual \(\tilde r\), the part of schooling not linearly predicted by experience;
2. regress \(\log w\) on \(\tilde r\): the slope is \(\widehat\beta_1=\sum_i\tilde r_i\log w_i/\sum_i\tilde r_i^2\).

Numerically, both routes give identical coefficients and identical residuals, which you should verify in the lab. Interpretation: \(\widehat\beta_1\) estimates the projection coefficient — the approximate percentage wage difference associated with one more year of schooling among workers with the same experience. Whether it is the *return* to schooling depends on whether \(E[u\mid educ,exper]=0\) has a causal justification; unobserved ability is the classic reason to doubt it (Lecture 3 derives the resulting bias).

## Problem set
1. **CEF and BLP.** Let \(X\sim U[0,1]\) and \(Y=X^2+\varepsilon\), with \(\varepsilon\) independent of \(X\), \(E[\varepsilon]=0\). (a) Find \(m(x)\). (b) Find the BLP of \(Y\) on \((1,X)\). (c) Show that the projection error is uncorrelated with \(X\) but not mean-independent of \(X\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    (a) \(m(x)=x^2\). (b) \(\beta_1=\operatorname{Cov}(X,X^2)/\operatorname{Var}(X)\). With \(E[X]=1/2\), \(E[X^2]=1/3\) and \(E[X^3]=1/4\): \(\operatorname{Cov}(X,X^2)=1/4-1/6=1/12\) and \(\operatorname{Var}(X)=1/12\), so \(\beta_1=1\) and \(\beta_0=1/3-1/2=-1/6\). The BLP is \(X-1/6\). (c) \(u=X^2-X+1/6+\varepsilon\). By construction \(E[u]=0\) and \(E[Xu]=0\), but \(E[u\mid X]=X^2-X+1/6\neq0\).
    </details>

2. **Projection matrices.** Prove that \(h_{ii}\in[0,1]\). *Hint:* \(h_{ii}=e_i'P_Xe_i=\|P_Xe_i\|^2\le\|e_i\|^2\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(h_{ii}=e_i'P_Xe_i=e_i'P_X'P_Xe_i=\|P_Xe_i\|^2\ge0\) by symmetry and idempotence. Since \(e_i=P_Xe_i+M_Xe_i\) is an orthogonal decomposition, \(\|P_Xe_i\|^2\le\|e_i\|^2=1\).
    </details>

3. **Demeaning.** Use FWL with \(X_1=\iota\) (a column of ones) to show that the slope coefficients in a regression with an intercept equal the coefficients from regressing demeaned \(y\) on demeaned regressors.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(M_\iota=I-\iota(\iota'\iota)^{-1}\iota'=I-\tfrac1n\iota\iota'\), so \(M_\iota z=z-\bar z\iota\) for any vector \(z\). FWL then gives \(\widehat\beta_2=(\tilde X_2'\tilde X_2)^{-1}\tilde X_2'\tilde y\), where tildes denote demeaned data.
    </details>

4. **Biased under projection only.** Give an example in which \(E[Xu]=0\) holds but \(E[\widehat\beta\mid X]\neq\beta\). Explain why OLS can still be consistent.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Use problem 1. \(E[\widehat\beta\mid X]=\beta+(X'X)^{-1}X'E[u\mid X]\), and \(E[u_i\mid X_i]=X_i^2-X_i+1/6\neq0\), so the conditional bias is generally nonzero for a given sample of \(X\). Consistency only needs \(n^{-1}X'u\to_pE[X_iu_i]=0\), which holds by the law of large numbers (Lecture 2).
    </details>

5. **Gauss–Markov without homoskedasticity.** Let \(E[uu'\mid X]=\Omega\neq\sigma^2I\). Show that OLS is still unbiased, derive \(\operatorname{Var}(\widehat\beta\mid X)\), and explain why OLS is no longer BLUE.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Unbiasedness uses only A4. \(\operatorname{Var}(\widehat\beta\mid X)=(X'X)^{-1}X'\Omega X(X'X)^{-1}\), the "sandwich". The GLS estimator \((X'\Omega^{-1}X)^{-1}X'\Omega^{-1}y\) is OLS on the transformed model \(\Omega^{-1/2}y=\Omega^{-1/2}X\beta+\Omega^{-1/2}u\), which satisfies A5. By Gauss–Markov applied to the transformed model, GLS has variance \((X'\Omega^{-1}X)^{-1}\le\) the OLS sandwich.
    </details>

## Computational lab
**Goal:** verify the algebra of this lecture numerically.

```python
import numpy as np
rng = np.random.default_rng(14)
n = 1_000
exper = rng.uniform(0, 30, n)
educ = 10 + 0.1 * exper + rng.normal(0, 2, n)
logw = 1.0 + 0.08 * educ + 0.03 * exper - 0.0005 * exper**2 + rng.normal(0, 0.4, n)

X = np.column_stack([np.ones(n), educ, exper])
b = np.linalg.solve(X.T @ X, X.T @ logw)            # never invert X'X explicitly

# Frisch–Waugh–Lovell: partial out (1, exper)
X1 = np.column_stack([np.ones(n), exper])
M1 = lambda z: z - X1 @ np.linalg.lstsq(X1, z, rcond=None)[0]
b_fwl = (M1(educ) @ M1(logw)) / (M1(educ) @ M1(educ))
print(b[1], b_fwl)                                   # identical to machine precision

H = X @ np.linalg.solve(X.T @ X, X.T)                # projection matrix (small n only)
print(np.trace(H), np.allclose(H @ H, H))            # k = 3, idempotent
```

Tasks: (i) confirm that the residuals from the long regression and the FWL regression coincide; (ii) note that the true CEF is quadratic in experience, so the regression above estimates a projection — compare \(\widehat\beta\) with the BLP coefficients computed from a very large simulated sample; (iii) repeat in R with `lm()` and verify with `model.matrix()` and `hatvalues()`.

## Reading
**Required:** Hansen, *Econometrics* (2022), chapters 2–4; Hayashi, *Econometrics*, chapter 1.
**Further:** Angrist and Pischke, *Mostly Harmless Econometrics*, chapter 3.1 (regression and the CEF); Davidson and MacKinnon, *Econometric Theory and Methods*, chapter 2 (geometry of least squares); Frisch and Waugh (1933), "Partial time regressions as compared with individual trends", *Econometrica* 1(4); Lovell (1963), "Seasonal adjustment of economic time series and multiple regression analysis", *JASA* 58(304).
