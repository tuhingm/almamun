# Lecture 04 — Heteroskedasticity, Dependence, and Robust Inference

## Learning objectives
By the end of this lecture you should be able to:

1. derive the heteroskedasticity-robust (HC) covariance estimators and explain the finite-sample corrections HC1–HC3;
2. test for heteroskedasticity and decide when weighted least squares is worth using;
3. derive the cluster-robust variance estimator and the Moulton factor, and explain what goes wrong with few clusters;
4. derive the long-run variance and the Newey–West HAC estimator for time-series regressions;
5. choose the level of clustering from the sampling and assignment design.

## Motivation
Point estimates come from the moment condition \(E[X_iu_i]=0\). Standard errors come from the variance of the score \(\sum_iX_iu_i\), and that variance depends on how errors are related *across* observations and how their variance changes *with* the regressors. Getting this wrong does not bias \(\widehat\beta\), but it can make a 5% test reject 30% or 50% of the time. Much of the replication debate in applied economics has turned on standard errors rather than coefficients. The organizing principle of this lecture is simple: **the variance estimator must match the dependence structure of the scores.**

## 1. Heteroskedasticity-consistent covariance matrices
From Lecture 2, \(\operatorname{avar}(\widehat\beta)=Q^{-1}\Omega Q^{-1}/n\) with \(\Omega=E[X_iX_i'u_i^2]\). Conditional on \(X\), with independent observations,
\[
\operatorname{Var}(\widehat\beta\mid X)=(X'X)^{-1}\Big(\sum_iX_iX_i'\sigma_i^2\Big)(X'X)^{-1},\qquad \sigma_i^2=E[u_i^2\mid X_i] .
\]
The White (1980) insight is that we do not need to estimate each \(\sigma_i^2\) consistently. We only need the \(k\times k\) average \(n^{-1}\sum_iX_iX_i'\sigma_i^2\), and replacing \(\sigma_i^2\) by \(\widehat u_i^2\) does this.

| Estimator | Weight on \(\widehat u_i^2\) | Rationale |
|---|---|---|
| HC0 | \(1\) | White (1980); consistent, biased downward in small samples |
| HC1 | \(n/(n-k)\) | degrees-of-freedom scaling; Stata's `robust` |
| HC2 | \(1/(1-h_{ii})\) | unbiased under homoskedasticity |
| HC3 | \(1/(1-h_{ii})^2\) | approximates the jackknife; best small-sample size in most simulations |

**Why leverage matters.** Under homoskedasticity, \(E[\widehat u_i^2]=\sigma^2(1-h_{ii})\). Residuals at high-leverage points are shrunk because the fit chases those points. HC2 undoes this exactly; HC3 over-corrects, which protects against the remaining downward bias under heteroskedasticity. With \(n\) in the thousands and no extreme leverage, the four estimators agree. With small samples or a few influential observations, they can differ substantially. Report HC3, or check that conclusions survive it (Long and Ervin, 2000; MacKinnon and White, 1985).

## 2. Testing for heteroskedasticity, and WLS
**Breusch–Pagan / White tests.** Regress \(\widehat u_i^2\) on a set of variables \(Z_i\) (the regressors for BP, regressors plus squares and cross-products for White) and use \(nR^2\to_d\chi^2_{\dim Z-1}\) under homoskedasticity. A rejection says the classical variance formula is wrong. Because robust standard errors are valid either way, the test matters mainly for deciding whether **efficiency gains** from reweighting are worthwhile.

**Weighted least squares.** If \(\operatorname{Var}(u_i\mid X_i)=\sigma^2h(X_i)\) with \(h\) known, OLS on the transformed model \(Y_i/\sqrt{h_i}=(X_i/\sqrt{h_i})'\beta+u_i/\sqrt{h_i}\) is BLUE (GLS). With \(h(X_i;\gamma)\) estimated in a first step, **feasible GLS** is asymptotically efficient if the variance model is correct. Practical advice (Romano and Wolf, 2017): use WLS for efficiency **and** robust standard errors for validity, because the variance model is almost never exactly right. Note that WLS and OLS estimate the same \(\beta\) only when the CEF is linear. Under misspecification, they converge to different weighted projections.

## 3. Clustered dependence
Suppose observations come in \(G\) groups (villages, schools, firms, states), indexed by \(g\), with \(n_g\) members. Errors are independent *across* clusters but arbitrarily correlated *within* them. Writing \(X_g\) and \(u_g\) for the cluster's stacked data,
\[
\operatorname{Var}(\widehat\beta\mid X)=(X'X)^{-1}\Big(\sum_{g=1}^GX_g'E[u_gu_g'\mid X]X_g\Big)(X'X)^{-1}.
\]
The **cluster-robust** (CR) estimator (Liang and Zeger, 1986; Arellano, 1987) replaces \(E[u_gu_g']\) by \(\widehat u_g\widehat u_g'\):
\[
\widehat V_{CR}=c\,(X'X)^{-1}\Big(\sum_gX_g'\widehat u_g\widehat u_g'X_g\Big)(X'X)^{-1},\qquad c=\frac{G}{G-1}\cdot\frac{n-1}{n-k}.
\]
It is consistent as \(G\to\infty\) with cluster sizes bounded. **The effective sample size for inference is \(G\), not \(n\).**

**The Moulton factor.** Why does clustering matter so much? Take equal cluster sizes \(m\), a regressor \(x\) with within-cluster correlation \(\rho_x\), and an error with within-cluster correlation \(\rho_u\). Then
\[
\frac{\operatorname{Var}_{\text{true}}(\widehat\beta_1)}{\operatorname{Var}_{\text{OLS}}(\widehat\beta_1)}\approx1+(m-1)\rho_x\rho_u .
\]
When the regressor is constant within clusters (\(\rho_x=1\)), as with a state-level policy, and \(m=100\) individuals per state with \(\rho_u=0.05\), the variance ratio is about \(5.95\). Conventional standard errors are then too small by a factor of about \(\sqrt{5.95}=2.4\) (Moulton, 1990).

**Few clusters.** With \(G\) below roughly 30–50, or with very unbalanced clusters, CR \(t\)-tests over-reject even with the \(c\) correction. Remedies:

- use \(t_{G-1}\) critical values rather than normal ones;
- use the CR2 estimator (Bell–McCaffrey) with Imbens–Kolesár degrees of freedom;
- use the **wild cluster bootstrap** with the null imposed (Cameron, Gelbach and Miller, 2008; Roodman et al., 2019). With very few *treated* clusters, every method struggles; randomization inference may be the honest option (MacKinnon and Webb, 2017).

**Where to cluster.** Abadie, Athey, Imbens and Wooldridge (2023) show that clustering is justified by the *design*: sampling clusters from a population (sampling-based clustering) or assigning treatment at the cluster level (design-based clustering). Clustering "because the errors might be correlated" at the coarsest available level is not always conservative, and it can be very imprecise. **Rule of thumb:** cluster at the level at which treatment is assigned. In panels, cluster at least at the unit level to allow arbitrary serial correlation (Bertrand, Duflo and Mullainathan, 2004).

## 4. Serial correlation: HAC estimation
In a time-series regression \(y_t=x_t'\beta+u_t\), the score \(s_t=x_tu_t\) may be serially correlated. Then
\[
\operatorname{Var}\Big(\frac1{\sqrt T}\sum_{t=1}^Ts_t\Big)\to\Omega_{LR}=\sum_{j=-\infty}^{\infty}\Gamma_j,\qquad \Gamma_j=E[s_ts_{t-j}'],
\]
the **long-run variance**: the spectral density of \(s_t\) at frequency zero, times \(2\pi\). The sample autocovariances \(\widehat\Gamma_j=T^{-1}\sum_t\widehat s_t\widehat s_{t-j}'\) cannot simply be summed over all lags: the result is not consistent and need not be positive semidefinite. Newey and West (1987) down-weight with a Bartlett kernel:
\[
\widehat\Omega_{NW}=\widehat\Gamma_0+\sum_{j=1}^{L}\Big(1-\frac{j}{L+1}\Big)\big(\widehat\Gamma_j+\widehat\Gamma_j'\big),
\]
which is positive semidefinite by construction and consistent if \(L\to\infty\) with \(L/T\to0\).

**Bandwidth.** A common rule is \(L=\lfloor0.75\,T^{1/3}\rfloor\) (Stock and Watson). Andrews (1991) gives data-dependent choices. Small \(L\) with persistent scores gives size distortions; large \(L\) gives noisy estimates. Recent work recommends larger bandwidths combined with fixed-\(b\) critical values, or equal-weighted cosine (EWC) estimators with \(t\) critical values (Lazarus, Lewis, Stock and Watson, 2018).

**HAC does not fix dynamic misspecification.** If \(y_t\) depends on \(y_{t-1}\) and the lag is omitted, the problem is in the conditional mean, not just the variance. In particular, a regression with a lagged dependent variable and serially correlated errors is inconsistent (Lecture 9).

## 5. A unified view
All of the estimators above have the same sandwich form \(\widehat Q^{-1}\widehat\Omega\widehat Q^{-1}\). They differ only in the "meat":

| Dependence assumption | Meat \(\widehat\Omega\) | Asymptotics in |
|---|---|---|
| i.i.d., homoskedastic | \(s^2\,n^{-1}X'X\) | \(n\) |
| independent, heteroskedastic | \(n^{-1}\sum_iX_iX_i'\widehat u_i^2\) | \(n\) |
| clustered | \(n^{-1}\sum_gX_g'\widehat u_g\widehat u_g'X_g\) | \(G\) |
| serially correlated | kernel-weighted autocovariances | \(T\) |
| two-way (firm and year) | \(\widehat\Omega_{firm}+\widehat\Omega_{year}-\widehat\Omega_{firm\cap year}\) | \(\min(G_1,G_2)\) |
| spatial | distance-kernel weights (Conley, 1999) | spatial mixing |

## Worked example
**State minimum wage policy.** A researcher regresses teen employment for 200,000 individuals on a state minimum-wage indicator, using 50 states over 10 years. The robust (HC1) standard error is 0.004, and the coefficient is 0.012 (\(t=3.0\)). The policy varies only at the state-year level, and employment shocks are correlated within states over time. Clustering by state gives a standard error of 0.011 (\(t=1.09\)).

What happened? The regressor is perfectly correlated within state-years and highly persistent within states. The Moulton factor is therefore large, and the HC1 standard error treats 200,000 correlated observations as independent. The correct inference uses \(G=50\) clusters, at the level of policy assignment, with \(t_{49}\) critical values. A wild cluster bootstrap p-value is a useful check.

## Problem set
1. **HC2 is unbiased under homoskedasticity.** Show that \(E[\widehat u_i^2\mid X]=\sigma^2(1-h_{ii})\) when \(\operatorname{Var}(u\mid X)=\sigma^2I\). Conclude that HC2 is conditionally unbiased in this case.

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(\widehat u=M_Xu\), so \(E[\widehat u\widehat u'\mid X]=\sigma^2M_X\), whose \(i\)-th diagonal element is \(\sigma^2(1-h_{ii})\). HC2 weights \(\widehat u_i^2\) by \(1/(1-h_{ii})\), so the meat has expectation \(\sum_iX_iX_i'\sigma^2\), the true meat.
    </details>

2. **Moulton factor.** Derive \(1+(m-1)\rho_x\rho_u\) for a single regressor with equal cluster sizes, equicorrelated errors (\(\operatorname{Corr}(u_{ig},u_{jg})=\rho_u\)), and a regressor with within-cluster correlation \(\rho_x\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    With demeaned \(x\), \(\operatorname{Var}(\sum_{g,i}x_{ig}u_{ig})=\sigma_u^2\sum_g\big[\sum_ix_{ig}^2+\rho_u\sum_{i\neq j}x_{ig}x_{jg}\big]\). The second sum is approximately \(\rho_x(m-1)\sum_ix_{ig}^2\) on average. Dividing by the OLS variance \(\sigma^2_u\sum x_{ig}^2\) gives the factor.
    </details>

3. **Positive semidefiniteness.** Show by example (\(T=3\), \(L=1\), unweighted sum) that \(\widehat\Gamma_0+\sum_{j=1}^{L}(\widehat\Gamma_j+\widehat\Gamma_j')\) can be negative for a scalar score. Then explain why Bartlett weights guarantee non-negativity.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Take \(\widehat s=(1,-1,1)\). Then \(\widehat\Gamma_0=1\) and \(\widehat\Gamma_1=-2/3\), so the truncated unweighted sum is \(1-4/3=-1/3<0\): a negative "variance". (Summing *all* \(T-1\) lags gives \((\sum_t\widehat s_t)^2/T\ge0\), but that estimator is inconsistent.) The Bartlett estimator equals a scaled average of squared partial sums of length \(L+1\), which is a variance and so non-negative.
    </details>

4. **Choosing the cluster level.** A randomized trial assigns a training program at the classroom level (200 classrooms in 20 schools). Students are the unit of observation. At which level should you cluster, and why not at the school level by default?

    <details class="solution" markdown="1"><summary>Solution</summary>

    Cluster at the classroom level, the unit of assignment (design-based justification). Clustering at the school level is not required by the design, leaves only 20 clusters with poor finite-sample behavior, and inflates variance without reducing bias in the variance estimate (Abadie et al., 2023).
    </details>

5. **WLS under misspecification.** Show with a simple example that when the CEF is nonlinear, OLS and WLS converge to different coefficients, so a Hausman-type comparison of OLS and WLS is a specification test.

    <details class="solution" markdown="1"><summary>Solution</summary>

    WLS converges to \(E[w XX']^{-1}E[wXY]\), the projection under weights \(w(X)\). With \(Y=X^2+e\), \(X\sim U[0,1]\) and \(w=X\), the weighted slope is \(\operatorname{Cov}_w(X,X^2)/\operatorname{Var}_w(X)\), which differs from the unweighted slope of 1. If the CEF is linear, both equal \(\beta\), so a significant difference indicates misspecification.
    </details>

## Computational lab
**Goal:** see the Moulton problem and its fixes.

```python
import numpy as np
rng = np.random.default_rng(4)

def sim(G=40, m=50, rho_u=0.1):
    g = np.repeat(np.arange(G), m)
    d = (rng.random(G) < 0.5).astype(float)[g]               # treatment assigned by cluster
    u = np.sqrt(rho_u) * rng.normal(size=G)[g] + np.sqrt(1 - rho_u) * rng.normal(size=G * m)
    y = 0.0 * d + u                                         # true effect = 0
    X = np.column_stack([np.ones_like(d), d])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V_hc = XtXi @ (X * e[:, None]**2).T @ X @ XtXi
    S = np.zeros((2, 2))
    for c in range(G):
        sc = X[g == c].T @ e[g == c]
        S += np.outer(sc, sc)
    n, k = len(y), 2
    V_cr = XtXi @ S @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return abs(b[1] / np.sqrt(V_hc[1, 1])) > 1.96, abs(b[1] / np.sqrt(V_cr[1, 1])) > 2.02  # t_39 critical value

rej = np.array([sim() for _ in range(2000)]).mean(axis=0)
print("rejection rate under H0: HC =", rej[0], " cluster =", rej[1])
```

Tasks: (i) compare the HC rejection rate with the Moulton prediction \(1+(m-1)\rho_u\); (ii) reduce \(G\) to 10 and observe the cluster-robust size; (iii) implement a wild cluster bootstrap with Rademacher weights and the null imposed. In R, use `fixest::feols(y ~ d, cluster = ~g)` and `fwildclusterboot::boottest()`. In Stata, use `reg y d, vce(cluster g)` and `boottest`.

## Reading
**Required:** Cameron and Miller (2015), "A practitioner's guide to cluster-robust inference", *Journal of Human Resources* 50(2); Hansen, *Econometrics*, chapters 4.14–4.23 and 7.
**Further:** White (1980), *Econometrica* 48(4); Long and Ervin (2000), "Using heteroscedasticity consistent standard errors in the linear regression model", *The American Statistician* 54(3); Moulton (1990), "An illustration of a pitfall in estimating the effects of aggregate variables on micro units", *REStat* 72(2); Bertrand, Duflo and Mullainathan (2004), "How much should we trust differences-in-differences estimates?", *QJE* 119(1); Newey and West (1987), *Econometrica* 55(3); Lazarus, Lewis, Stock and Watson (2018), "HAR inference: recommendations for practice", *JBES* 36(4); Abadie, Athey, Imbens and Wooldridge (2023), "When should you adjust standard errors for clustering?", *QJE* 138(1).
