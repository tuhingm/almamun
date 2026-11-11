# Lecture 07 — Panel Data: Fixed Effects, Random Effects, and Correlated Effects

## Learning objectives
By the end of this lecture you should be able to:

1. write the unobserved-effects model and explain what panel data can and cannot identify;
2. derive the fixed-effects (within) and first-difference estimators and state the strict-exogeneity assumption each needs;
3. derive the random-effects GLS estimator as quasi-demeaning and explain when it is consistent;
4. use the Mundlak (correlated random effects) device and a robust Hausman test to choose between FE and RE;
5. conduct inference with clustered standard errors, and recognize the limits of two-way fixed effects.

## Motivation
Cross-sectional regressions compare different people, firms or countries. Panel data follow the same units over time, and so can compare each unit *with itself*. If the confounders that worry us are stable over time — management quality, ability, geography, institutions — differencing them out removes omitted-variable bias without ever measuring them. That is the promise. The price is that we learn only from within-unit variation, which can be small, noisy and dominated by measurement error, and that we need a strong exogeneity assumption about the *whole* history of the regressors.

## 1. The unobserved-effects model
For units \(i=1,\dots,N\) and periods \(t=1,\dots,T\),
\[
y_{it}=x_{it}'\beta+c_i+u_{it},
\]
where \(c_i\) is a time-invariant unobserved effect and \(u_{it}\) is the idiosyncratic error. Asymptotics are for \(N\to\infty\) with \(T\) fixed, the typical micro panel. Two questions organize everything that follows:

1. Is \(c_i\) correlated with \(x_{it}\)? If so, pooled OLS and random effects are inconsistent; fixed effects and first differences are not.
2. Is \(u_{it}\) uncorrelated with \(x_{is}\) for **all** \(s,t\) (strict exogeneity), or only for \(s\le t\) (sequential exogeneity)? FE and FD need strict exogeneity; dynamic panels (Lecture 8) need only sequential exogeneity, with GMM.

<div class="definition" markdown="1">
**Strict exogeneity.** \(E[u_{it}\mid x_{i1},\dots,x_{iT},c_i]=0\) for all \(t\). It rules out feedback from past shocks to future regressors (for example, a firm raising investment after a good productivity shock) and lagged dependent variables among the regressors.
</div>

## 2. Pooled OLS
Pooled OLS regresses \(y_{it}\) on \(x_{it}\) with composite error \(v_{it}=c_i+u_{it}\). It is consistent only if \(E[x_{it}c_i]=0\) and \(E[x_{it}u_{it}]=0\). Even then, \(v_{it}\) is serially correlated within units (through \(c_i\)), so standard errors must be clustered by unit.

## 3. Fixed effects (the within estimator)
Average the model over time for each unit, \(\bar y_i=\bar x_i'\beta+c_i+\bar u_i\), and subtract:
\[
\ddot y_{it}=\ddot x_{it}'\beta+\ddot u_{it},\qquad \ddot z_{it}=z_{it}-\bar z_i .
\]
The unobserved effect is gone. The FE estimator is pooled OLS on demeaned data:
\[
\widehat\beta_{FE}=\Big(\sum_i\sum_t\ddot x_{it}\ddot x_{it}'\Big)^{-1}\sum_i\sum_t\ddot x_{it}\ddot y_{it}.
\]

<div class="theorem" markdown="1">
**Theorem 7.1 (FE consistency).** Under strict exogeneity and the rank condition \(\operatorname{rank}\big(\sum_tE[\ddot x_{it}\ddot x_{it}']\big)=k\), \(\widehat\beta_{FE}\to_p\beta\) as \(N\to\infty\) with \(T\) fixed, **whatever the correlation between \(c_i\) and \(x_{it}\)**.
</div>

<div class="proof" markdown="1">
*Proof.* \(\widehat\beta_{FE}-\beta=\big(N^{-1}\sum_i\ddot X_i'\ddot X_i\big)^{-1}N^{-1}\sum_i\ddot X_i'u_i\), using \(\ddot X_i'\ddot u_i=\ddot X_i'u_i\) because demeaning is idempotent. By the WLLN across \(i\), the second factor converges to \(E[\ddot X_i'u_i]=\sum_tE[\ddot x_{it}u_{it}]\). Since \(\ddot x_{it}=x_{it}-\bar x_i\) involves regressors from *every* period, strict exogeneity makes each term zero. ∎
</div>

**Three equivalent views.** (i) OLS on demeaned data. (ii) **LSDV**: OLS with a dummy for each unit, which by Frisch–Waugh–Lovell (Lecture 1) gives identical slopes, because partialling out unit dummies *is* demeaning. (iii) A GMM estimator using the moments \(E[\ddot x_{it}u_{it}]=0\).

**What FE cannot do.** Coefficients on time-invariant regressors (gender, distance, initial conditions) are not identified, because they are absorbed into \(c_i\). Their interactions with time-varying regressors are identified.

**Degrees of freedom.** When computing the error variance by hand, divide by \(N(T-1)-k\), not \(NT-k\): demeaning uses up \(N\) degrees of freedom. Software that estimates LSDV does this automatically; manual demeaning followed by OLS does not.

**Inference.** Use standard errors clustered by unit. They are robust to arbitrary serial correlation and heteroskedasticity in \(u_{it}\) and valid as \(N\to\infty\) (Arellano, 1987).

## 4. First differences
Difference adjacent periods: \(\Delta y_{it}=\Delta x_{it}'\beta+\Delta u_{it}\), for \(t=2,\dots,T\). FD also needs strict exogeneity (\(\Delta x_{it}\) involves \(x_{i,t-1}\), and \(\Delta u_{it}\) involves \(u_{it}\)). With \(T=2\), FD and FE are numerically identical. With \(T>2\), they differ in efficiency:

- if \(u_{it}\) is serially uncorrelated, FE is more efficient;
- if \(u_{it}\) is a random walk, \(\Delta u_{it}\) is serially uncorrelated and FD is more efficient.

A large discrepancy between FE and FD is a symptom of violated strict exogeneity or of measurement error, since the two estimators are affected differently.

**Measurement error is amplified.** With classical error in \(x_{it}\), within-unit variation in the true regressor is often small while the noise is not, so the reliability ratio of \(\ddot x_{it}\) or \(\Delta x_{it}\) can be much lower than that of \(x_{it}\). This often explains why FE estimates are smaller than OLS (Griliches and Hausman, 1986).

## 5. Random effects
Assume additionally \(E[c_i\mid x_{i1},\dots,x_{iT}]=0\), so \(c_i\) is part of the error, plus homoskedasticity: \(\operatorname{Var}(c_i)=\sigma_c^2\) and \(\operatorname{Var}(u_{it})=\sigma_u^2\), with no serial correlation in \(u_{it}\). Then \(v_i=c_i\iota_T+u_i\) has the equicorrelated covariance matrix \(\Omega=\sigma_u^2I_T+\sigma_c^2\iota_T\iota_T'\). GLS is efficient, and it can be computed as pooled OLS on **quasi-demeaned** data:
\[
y_{it}-\theta\bar y_i=(x_{it}-\theta\bar x_i)'\beta+\text{error},\qquad \theta=1-\sqrt{\frac{\sigma_u^2}{\sigma_u^2+T\sigma_c^2}} .
\]
The cases \(\theta=0\) (pooled OLS, when \(\sigma_c^2=0\)) and \(\theta\to1\) (FE, as \(T\sigma_c^2/\sigma_u^2\to\infty\)) bracket RE. Feasible RE estimates \(\sigma_u^2\) from FE residuals and \(\sigma_c^2\) from pooled or between residuals.

**When to use RE.** RE is more efficient than FE and identifies coefficients on time-invariant regressors. But it is **inconsistent if \(c_i\) is correlated with any regressor**, which is precisely the concern that motivates panel data in observational economics. RE is natural when units are sampled at random and the regressor of interest is randomly assigned (for example, in an experiment with repeated measurements).

## 6. Correlated random effects and the Hausman test
**Mundlak (1978).** Model the correlation explicitly: \(c_i=\bar x_i'\xi+a_i\) with \(E[a_i\mid x_i]=0\). Substituting,
\[
y_{it}=x_{it}'\beta+\bar x_i'\xi+a_i+u_{it}.
\]
Estimating this equation by pooled OLS or RE gives \(\widehat\beta\) **numerically identical to FE** (with a balanced panel). It also allows time-invariant regressors and makes the FE-versus-RE question a test of \(H_0:\xi=0\).

**Hausman test.** The classic statistic is \(H=(\widehat\beta_{FE}-\widehat\beta_{RE})'[\widehat V_{FE}-\widehat V_{RE}]^{-1}(\widehat\beta_{FE}-\widehat\beta_{RE})\). It relies on RE being fully efficient under \(H_0\), which fails with heteroskedasticity or serial correlation. The variance difference can then be non-positive-definite. **Use the cluster-robust Wald test of \(\xi=0\) in the Mundlak regression instead.** It is valid under arbitrary within-unit dependence.

**Interpretation.** Not rejecting \(\xi=0\) does not prove that RE is valid. The test has low power when within-unit variation is small. And FE remains consistent under both hypotheses.

## 7. Two-way fixed effects and their limits
Adding time effects, \(y_{it}=x_{it}'\beta+c_i+\lambda_t+u_{it}\), removes common shocks. With balanced panels, this is OLS on double-demeaned data \(z_{it}-\bar z_i-\bar z_t+\bar z\). Two warnings:

1. FE removes **time-invariant** confounding only. Unit-specific trends, time-varying shocks correlated with \(x_{it}\), and reverse causality remain.
2. When \(x_{it}\) is a binary treatment adopted at different times and effects are heterogeneous across units or over time, the TWFE coefficient is a weighted average of unit-time effects in which some weights can be **negative** (de Chaisemartin and D'Haultfœuille, 2020; Goodman-Bacon, 2021). Lecture 13 develops the alternatives.

## Worked example
**Union wage premium.** In a panel of workers, pooled OLS of \(\log w_{it}\) on union status and controls gives about 0.18; FE gives about 0.08 (orders of magnitude similar to Freeman, 1984). How should we read the drop?

- **Selection on stable traits:** if more productive workers sort into union jobs, \(c_i\) is positively correlated with union status, and pooled OLS is biased upward. FE removes this.
- **Measurement error:** union status is misreported in a few percent of observations. Most within-worker changes in union status are real switches, but some are reporting errors, so the reliability of \(\Delta union\) is much lower than that of the level. Some of the drop is attenuation.
- **Strict exogeneity:** if workers move into union jobs after a negative wage shock (for example, after a layoff), union status depends on past \(u_{is}\), and FE is biased.

The honest conclusion is a range, supported by robustness checks: an FE–FD comparison, event-study plots around switches, and validation data on misreporting.

## Problem set
1. **FE equals LSDV.** Using Frisch–Waugh–Lovell, prove that the slope estimates from regressing \(y\) on \(x\) and \(N\) unit dummies equal \(\widehat\beta_{FE}\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    Let \(D\) be the \(NT\times N\) matrix of unit dummies. \(M_D=I-D(D'D)^{-1}D'\), and \(D(D'D)^{-1}D'z\) replaces each observation with its unit mean, so \(M_Dz=\ddot z\). By FWL, the slope on \(x\) is \((\ddot x'\ddot x)^{-1}\ddot x'\ddot y=\widehat\beta_{FE}\).
    </details>

2. **FD versus FE with \(T=2\).** Show that the two estimators are identical when \(T=2\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    With \(T=2\), \(\ddot z_{i1}=-(z_{i2}-z_{i1})/2\) and \(\ddot z_{i2}=(z_{i2}-z_{i1})/2\). Then \(\sum_t\ddot x_{it}\ddot x_{it}'=\Delta x_i\Delta x_i'/2\) and \(\sum_t\ddot x_{it}\ddot y_{it}=\Delta x_i\Delta y_i/2\). The factors of \(1/2\) cancel in the ratio.
    </details>

3. **Feedback.** Suppose \(x_{it}=\rho x_{i,t-1}+\gamma u_{i,t-1}+e_{it}\), with \(\gamma\neq0\). Show that strict exogeneity fails and derive the sign of the FE bias for small \(T\) when \(\gamma>0\) and \(\beta=0\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    For \(s>t\), \(E[x_{is}u_{it}]=\gamma\rho^{s-t-1}\sigma_u^2\neq0\): today's shock moves future regressors, so strict exogeneity fails (sequential exogeneity, \(E[x_{is}u_{it}]=0\) for \(s\le t\), still holds). With \(\beta=0\), \(\operatorname{plim}\widehat\beta_{FE}=E[\sum_t\ddot x_{it}u_{it}]/E[\sum_t\ddot x_{it}^2]\), and \(E[\sum_t\ddot x_{it}u_{it}]=\sum_tE[x_{it}u_{it}]-T\,E[\bar x_i\bar u_i]=-\tfrac1T\sum_{s>t}\gamma\rho^{s-t-1}\sigma_u^2<0\) for \(\gamma>0,\rho\ge0\). The bias is negative and of order \(1/T\), vanishing as \(T\to\infty\).
    </details>

4. **Mundlak equals FE.** Show that in a balanced panel the pooled OLS coefficient on \(x_{it}\) in the regression of \(y_{it}\) on \((1,x_{it},\bar x_i)\) equals \(\widehat\beta_{FE}\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    By FWL, partial out \((1,\bar x_i)\) from \(x_{it}\). The residual is \(\ddot x_{it}\), because \(x_{it}=\bar x_i+\ddot x_{it}\) and \(\ddot x_{it}\) is orthogonal to any function of \(i\) in a balanced panel. Regressing \(y_{it}\) on \(\ddot x_{it}\) gives \(\widehat\beta_{FE}\).
    </details>

5. **Clustering in panels.** A researcher with \(N=40\) states and \(T=30\) years clusters standard errors by year "because there are more years than states would allow". Comment.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Clustering by year allows correlation across states within a year but assumes independence over time within a state. That is usually the wrong assumption, since state-level shocks and policies are persistent. Cluster by state (40 clusters, with small-\(G\) corrections or a wild bootstrap) or two-way by state and year. The number of clusters is not a reason to choose the wrong dimension.
    </details>

## Computational lab
**Goal:** compare pooled OLS, FE, FD, RE and Mundlak when the unit effect is correlated with the regressor.

```python
import numpy as np
rng = np.random.default_rng(7)
N, T, beta = 1_000, 5, 1.0
c = rng.normal(size=N)
x = 0.8 * c[:, None] + rng.normal(size=(N, T))     # regressor correlated with c_i
y = beta * x + c[:, None] + rng.normal(size=(N, T))

def slope(yv, xv):
    xv, yv = xv - xv.mean(), yv - yv.mean()
    return (xv * yv).sum() / (xv * xv).sum()

b_pool = slope(y.ravel(), x.ravel())
xd, yd = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
b_fe = (xd * yd).sum() / (xd * xd).sum()
b_fd = slope(np.diff(y, axis=1).ravel(), np.diff(x, axis=1).ravel())
# Mundlak: regress y on x and x-bar
Xm = np.column_stack([np.ones(N * T), x.ravel(), np.repeat(x.mean(1), T)])
b_mundlak = np.linalg.lstsq(Xm, y.ravel(), rcond=None)[0][1]
print(f"pooled {b_pool:.3f}  FE {b_fe:.3f}  FD {b_fd:.3f}  Mundlak {b_mundlak:.3f}")
```

Tasks: (i) explain the pooled-OLS bias using the OVB formula; (ii) add classical measurement error to \(x\) and compare the attenuation of pooled OLS and FE; (iii) estimate RE with `linearmodels.panel.RandomEffects` and test \(\xi=0\) in the Mundlak regression with standard errors clustered by unit. In R, use `fixest::feols(y ~ x | id, cluster = ~id)` and `plm(model = "random")`. In Stata, use `xtreg, fe vce(cluster id)` and `xtreg, re` followed by the Mundlak regression.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 7.1 — The union wage premium: pooled OLS, random effects, fixed effects and first differences

*Data:* `wagepan` — 545 men, 1980–1987.

1. Estimate the union premium in `lwage` by pooled OLS, RE, FE and FD, with year effects and the controls `married`, `exper²` (and `educ`, `black`, `hisp` where identified). Use standard errors clustered by worker.
2. Run the Mundlak (correlated random effects) test of RE against FE, adding worker means of the time-varying regressors.
3. Interpret the ranking of the estimates.

```python
import wooldridge as wd
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS
df = wd.data("wagepan").set_index(["nr", "year"])
fe = PanelOLS.from_formula("lwage ~ union + married + expersq + TimeEffects + EntityEffects", data=df).fit(cov_type="clustered", cluster_entity=True)
# TODO: pooled, RE, FD; Mundlak: add group means of union, married, expersq to a RE/pooled regression
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS
raw = wd.data("wagepan")
df = raw.set_index(["nr", "year"])
yd = " + ".join(f"d8{i}" for i in range(1, 8))
cl = dict(cov_type="clustered", cluster_entity=True)
po = PooledOLS.from_formula(f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {yd}", data=df).fit(**cl)
re = RandomEffects.from_formula(f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {yd}", data=df).fit(**cl)
fe = PanelOLS.from_formula("lwage ~ union + married + expersq + TimeEffects + EntityEffects", data=df).fit(**cl)
fd = FirstDifferenceOLS.from_formula(f"lwage ~ union + married + expersq + {yd}", data=df).fit(**cl)
tab = pd.DataFrame({m: [r.params["union"], r.std_errors["union"]] for m, r in
                    [("pooled", po), ("RE", re), ("FE", fe), ("FD", fd)]}, index=["union", "cluster se"]).round(4)
print(tab)

for v in ("union", "married", "expersq"):
    raw[v + "_bar"] = raw.groupby("nr")[v].transform("mean")
mk = smf.ols(f"lwage ~ union + married + expersq + union_bar + married_bar + expersq_bar + educ + black + hisp + {yd}",
             data=raw).fit(cov_type="cluster", cov_kwds={"groups": raw["nr"]})
print(f"\nMundlak: union coefficient {mk.params['union']:.4f} (= FE)")
w = mk.wald_test("union_bar = 0, married_bar = 0, expersq_bar = 0", scalar=True)
print(f"robust test of RE vs FE (xi = 0): chi2(3) = {w.statistic:.2f}, p = {w.pvalue:.4f}")
```

Output:

```text
pooled      RE      FE      FD
union       0.1892  0.1103  0.0800  0.0411
cluster se  0.0275  0.0210  0.0243  0.0219

Mundlak: union coefficient 0.0800 (= FE)
robust test of RE vs FE (xi = 0): chi2(3) = 68.17, p = 0.0000
```

Pooled OLS gives the largest premium, RE an intermediate one, and FE and FD the smallest. The Mundlak regression
reproduces the FE coefficient exactly, and its cluster-robust test rejects \(\xi=0\) decisively, so the worker effect is
correlated with the regressors and RE is inconsistent. The drop from pooled OLS to FE is consistent with positive
selection of higher-wage workers into union jobs. Part of it may also be attenuation, because misreported switches in
union status loom large in within-worker variation (Lecture 7, worked example). The difference between FE and FD hints at
remaining measurement error or a failure of strict exogeneity.

</details>

### Exercise 7.2 — Crime and deterrence across North Carolina counties

*Data:* `crime4` — 90 North Carolina counties, 1981–1987 (Cornwell and Trumbull, 1994).

1. Estimate `lcrmrte` (log crimes per person) on log probability of arrest `lprbarr`, conviction `lprbconv`, prison `lprbpris`, log average sentence `lavgsen` and log police per capita `lpolpc`, with year effects, by pooled OLS and by county fixed effects. Cluster by county.
2. Compare the deterrence elasticities. Why could FE still be biased for `lpolpc`?

```python
import wooldridge as wd
from linearmodels.panel import PanelOLS, PooledOLS
df = wd.data("crime4").set_index(["county", "year"])
x = "lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"
# TODO: PooledOLS with year dummies vs PanelOLS with EntityEffects + TimeEffects
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import pandas as pd, wooldridge as wd
from linearmodels.panel import PanelOLS, PooledOLS
df = wd.data("crime4").set_index(["county", "year"])
x = "lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"
yd = " + ".join(f"d8{i}" for i in range(2, 8))
cl = dict(cov_type="clustered", cluster_entity=True)
po = PooledOLS.from_formula(f"lcrmrte ~ 1 + {x} + {yd}", data=df).fit(**cl)
fe = PanelOLS.from_formula(f"lcrmrte ~ {x} + EntityEffects + TimeEffects", data=df).fit(**cl)
names = x.split(" + ")
print(pd.DataFrame({"pooled": po.params[names], "pooled se": po.std_errors[names],
                    "FE": fe.params[names], "FE se": fe.std_errors[names]}).round(3))
print(f"\nwithin R2 (FE): {fe.rsquared_within:.3f};  counties: {df.index.get_level_values(0).nunique()}")
```

Output:

```text
pooled  pooled se     FE  FE se
lprbarr   -0.720      0.109 -0.360  0.064
lprbconv  -0.546      0.070 -0.286  0.055
lprbpris   0.248      0.108 -0.183  0.049
lavgsen   -0.087      0.112 -0.004  0.036
lpolpc     0.366      0.120  0.424  0.091

within R2 (FE): 0.354;  counties: 90
```

Fixed effects shrink the deterrence elasticities for arrest and conviction probabilities substantially compared with
pooled OLS: counties with persistently low crime also have high arrest rates for reasons that FE removes. The police
coefficient is **positive** in both models. That is the classic sign of reverse causality — counties hire police when
crime rises — which county fixed effects cannot fix because the feedback is time-varying. This is exactly the failure of
strict exogeneity discussed in Lecture 7, and it calls for an instrument or a dynamic model.

</details>

## Reading
**Required:** Wooldridge, *Econometric Analysis of Cross Section and Panel Data*, chapter 10; Angrist and Pischke, *Mostly Harmless Econometrics*, chapter 5.1.
**Further:** Mundlak (1978), "On the pooling of time series and cross section data", *Econometrica* 46(1); Hausman and Taylor (1981), *Econometrica* 49(6); Griliches and Hausman (1986), "Errors in variables in panel data", *Journal of Econometrics* 31(1); Arellano (1987), "Computing robust standard errors for within-groups estimators", *Oxford Bulletin* 49(4); Freeman (1984), "Longitudinal analyses of the effects of trade unions", *Journal of Labor Economics* 2(1).
