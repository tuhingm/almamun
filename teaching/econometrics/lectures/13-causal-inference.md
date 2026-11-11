# Lecture 13 — Causal Inference: Potential Outcomes, DiD, and Regression Discontinuity

## Learning objectives
By the end of this lecture you should be able to:

1. define causal estimands (ATE, ATT, LATE, CATE) in the potential-outcomes framework, and decompose a naive comparison into effect and selection bias;
2. estimate effects under unconfoundedness by regression adjustment, inverse probability weighting and doubly robust methods, and check overlap;
3. set up difference-in-differences, test its assumptions with event studies, and explain why two-way fixed effects fails with staggered adoption and heterogeneous effects;
4. describe the synthetic control method and its inference;
5. estimate sharp and fuzzy regression discontinuity designs with local polynomials, optimal bandwidths and robust bias-corrected inference, and validate them.

## Motivation
Since the "credibility revolution" (Angrist and Pischke, 2010), the most influential empirical work in economics has been organized around **research designs**: features of the world that make treated and untreated units comparable. This lecture collects the main designs that do not rely on instruments (Lecture 5 covered IV). The common thread is that every design replaces an untestable assumption about *everything* with a more transparent, partially testable assumption about *something specific*: random assignment, selection on observables, parallel trends, or continuity at a threshold.

## 1. Potential outcomes
Each unit \(i\) has potential outcomes \(Y_i(1)\) (treated) and \(Y_i(0)\) (untreated). We observe \(Y_i=D_iY_i(1)+(1-D_i)Y_i(0)\). The individual effect \(Y_i(1)-Y_i(0)\) is never observed: the **fundamental problem of causal inference**. The estimands are averages:
\[
ATE=E[Y(1)-Y(0)],\qquad ATT=E[Y(1)-Y(0)\mid D=1],\qquad CATE(x)=E[Y(1)-Y(0)\mid X=x].
\]
The notation embeds **SUTVA**: one unit's treatment does not affect another's outcomes (no spillovers), and there is a single version of treatment.

**Selection bias.** The naive difference in means decomposes as
\[
E[Y\mid D=1]-E[Y\mid D=0]=\underbrace{ATT}_{\text{effect}}+\underbrace{E[Y(0)\mid D=1]-E[Y(0)\mid D=0]}_{\text{selection bias}} .
\]
**Randomization** makes \(D\) independent of \((Y(1),Y(0))\), so the selection bias is zero and ATT = ATE. In a randomized experiment, regressing \(Y\) on \(D\) and pre-treatment covariates improves precision without introducing bias asymptotically. The fully interacted specification (Lin, 2013) is robust to heterogeneous effects.

## 2. Selection on observables
<div class="definition" markdown="1">
**Unconfoundedness and overlap.** (i) \((Y(1),Y(0))\perp D\mid X\): conditional on observed pre-treatment covariates, treatment is as good as random. (ii) \(0<e(x)<1\), where \(e(x)=P(D=1\mid X=x)\) is the **propensity score**.
</div>

Under these assumptions, \(ATE=E\big[\mu_1(X)-\mu_0(X)\big]\), with \(\mu_d(x)=E[Y\mid D=d,X=x]\). Three estimation approaches follow.

- **Regression adjustment:** fit \(\widehat\mu_1\) and \(\widehat\mu_0\) separately, then average \(\widehat\mu_1(X_i)-\widehat\mu_0(X_i)\). A single regression of \(Y\) on \(D\) and \(X\) instead estimates a variance-weighted average of conditional effects, which differs from the ATE when effects are heterogeneous (Angrist, 1998; Słoczyński, 2022).
- **Inverse probability weighting (IPW):** \(\widehat{ATE}=n^{-1}\sum_i\big[D_iY_i/\widehat e(X_i)-(1-D_i)Y_i/(1-\widehat e(X_i))\big]\). This is unstable when \(\widehat e\) is near 0 or 1; use normalized (Hájek) weights.
- **Doubly robust (AIPW):**
\[
\widehat{ATE}_{DR}=\frac1n\sum_i\Big[\widehat\mu_1(X_i)-\widehat\mu_0(X_i)+\frac{D_i(Y_i-\widehat\mu_1(X_i))}{\widehat e(X_i)}-\frac{(1-D_i)(Y_i-\widehat\mu_0(X_i))}{1-\widehat e(X_i)}\Big].
\]
It is consistent if **either** the outcome model **or** the propensity model is correct. It is efficient if both are. Combined with cross-fitting, it allows flexible machine-learning estimates of the nuisance functions while preserving \(\sqrt n\) inference (double/debiased machine learning; Chernozhukov et al., 2018).

**Matching** on covariates or on the propensity score is an alternative. Use bias-corrected matching estimators (Abadie and Imbens, 2011); bootstrap standard errors are invalid for nearest-neighbor matching.

**Overlap in practice.** Plot \(\widehat e(X)\) by treatment group. Trim units with extreme scores (Crump et al., 2009), and report how the estimand changes. **Sensitivity:** unconfoundedness is untestable. Report how strong unobserved confounding would have to be to overturn the result (Imbens, 2003; Cinelli and Hazlett, 2020).

## 3. Difference-in-differences
**The 2×2 design.** Two groups (treated \(G=1\), control \(G=0\)) and two periods (pre, post). The DiD estimand is
\[
\tau_{DiD}=\big(E[Y_{post}\mid G=1]-E[Y_{pre}\mid G=1]\big)-\big(E[Y_{post}\mid G=0]-E[Y_{pre}\mid G=0]\big).
\]

<div class="theorem" markdown="1">
**Theorem 13.1.** Under **parallel trends**, \(E[Y_{post}(0)-Y_{pre}(0)\mid G=1]=E[Y_{post}(0)-Y_{pre}(0)\mid G=0]\), and **no anticipation** (\(Y_{pre}(1)=Y_{pre}(0)\) for the treated), we have \(\tau_{DiD}=ATT\) in the post period.
</div>

<div class="proof" markdown="1">
*Proof.* For the treated, \(E[Y_{post}-Y_{pre}\mid G=1]=E[Y_{post}(1)-Y_{pre}(0)\mid G=1]\) by no anticipation. Subtract and add \(E[Y_{post}(0)\mid G=1]\) to get \(ATT+E[Y_{post}(0)-Y_{pre}(0)\mid G=1]\). By parallel trends, the second term equals the control group's observed change. ∎
</div>

The regression \(Y_{it}=\alpha+\beta G_i+\gamma Post_t+\tau(G_i\times Post_t)+\varepsilon_{it}\) gives \(\widehat\tau=\widehat\tau_{DiD}\). Parallel trends is an assumption about **untreated potential outcomes**, so it is untestable in the post period and **scale-dependent**: it cannot hold in both levels and logs unless trends are zero (Roth and Sant'Anna, 2023). Choose the scale on economic grounds. Cluster at the level of treatment assignment (Lecture 4).

**Event studies.** With multiple periods, estimate dynamic effects relative to the period before treatment:
\[
Y_{it}=\alpha_i+\lambda_t+\sum_{k\neq-1}\tau_k\,1\{t-E_i=k\}+\varepsilon_{it},
\]
where \(E_i\) is the treatment date. Pre-period coefficients near zero are *consistent with* parallel trends, but pre-trend tests have low power, and conditioning on passing them can distort inference (Roth, 2022). The approach of Rambachan and Roth (2023) gives inference that is robust to bounded violations of parallel trends.

**Staggered adoption and the TWFE problem.** When units adopt treatment at different times, the static TWFE regression \(Y_{it}=\alpha_i+\lambda_t+\tau D_{it}+\varepsilon_{it}\) is a weighted average of all 2×2 DiDs in the data. These include comparisons that use **already-treated units as controls** (Goodman-Bacon, 2021). If effects grow over time, those "forbidden comparisons" subtract treatment effects, and some implicit weights on group-time effects are **negative**. \(\widehat\tau\) can then have the wrong sign even if every unit's effect is positive (de Chaisemartin and D'Haultfœuille, 2020). Dynamic TWFE event studies suffer from the same contamination (Sun and Abraham, 2021). Heterogeneity-robust alternatives:

- **Callaway and Sant'Anna (2021):** estimate group-time ATTs, \(ATT(g,t)\), using never-treated or not-yet-treated units as controls, then aggregate with explicit, non-negative weights;
- **Sun and Abraham (2021):** an interaction-weighted event-study estimator;
- **Borusyak, Jaravel and Spiess (2024):** imputation — fit unit and time effects on untreated observations only, impute \(Y(0)\) for the treated, and average the differences.

## 4. Synthetic control
With a single (or few) treated aggregate unit(s) — a country, a state — and a long pre-period, Abadie, Diamond and Hainmueller (2010) construct a **synthetic control**: a convex combination of donor units, with weights \(w_j\ge0\) and \(\sum w_j=1\), chosen to match the treated unit's pre-treatment outcomes and predictors. The effect estimate is the post-treatment gap between the treated unit and its synthetic counterpart. The convexity constraint prevents extrapolation and makes weights interpretable. **Inference** uses placebo (permutation) tests: apply the method to each donor and compare the treated unit's post/pre RMSPE ratio with the placebo distribution. Extensions include synthetic DiD (Arkhangelsky et al., 2021), which combines unit and time weights, and conformal inference (Chernozhukov, Wüthrich and Zhu, 2021).

## 5. Regression discontinuity
Treatment is determined, fully or partly, by whether a **running variable** \(X\) crosses a cutoff \(c\).

**Sharp RD:** \(D=1\{X\ge c\}\). If \(E[Y(0)\mid X=x]\) and \(E[Y(1)\mid X=x]\) are continuous at \(c\), then
\[
\tau_{SRD}=\lim_{x\downarrow c}E[Y\mid X=x]-\lim_{x\uparrow c}E[Y\mid X=x]=E[Y(1)-Y(0)\mid X=c].
\]
**Fuzzy RD:** the probability of treatment jumps at \(c\) but by less than one. The estimand is the ratio of the outcome jump to the treatment jump, a LATE for compliers at the cutoff. It is IV with \(1\{X\ge c\}\) as the instrument.

**Estimation.** Use **local linear** regression on each side within a bandwidth \(h\), with a triangular kernel. Avoid global high-order polynomials, which give noisy estimates and misleading inference (Gelman and Imbens, 2019). Choose \(h\) to minimize MSE (Imbens and Kalyanaraman, 2012). Because the MSE-optimal bandwidth leaves a bias of the same order as the standard error, conventional confidence intervals undercover. Use **robust bias-corrected** inference (Calonico, Cattaneo and Titiunik, 2014), implemented in `rdrobust`.

**Validation.**

- **Manipulation:** if agents can precisely sort around the cutoff, continuity fails. Test for a discontinuity in the density of \(X\) at \(c\) (McCrary, 2008; Cattaneo, Jansson and Ma, 2020).
- **Covariate balance:** pre-determined covariates should not jump at \(c\). Run the RD with each covariate as the outcome.
- **Placebo cutoffs and bandwidth sensitivity:** no effects at fake cutoffs, and stable estimates across a range of bandwidths.
- **External validity:** the estimate is local to \(X=c\). Extrapolation away from the cutoff requires further assumptions.

## Worked example
**Minimum wages and employment (Card and Krueger, 1994).** New Jersey raised its minimum wage in April 1992; neighboring eastern Pennsylvania did not. Fast-food full-time-equivalent (FTE) employment per store, approximately:

| | NJ (treated) | PA (control) | Difference |
|---|---|---|---|
| Before | 20.4 | 23.3 | −2.9 |
| After | 21.0 | 21.2 | −0.2 |
| Change | +0.6 | −2.1 | **DiD = +2.7** |

The DiD estimate is about +2.7 FTE (standard error about 1.4): no evidence of the employment loss predicted by a competitive model. The identifying assumption is that NJ and PA stores would have followed parallel paths without the increase. The debate that followed (Neumark and Wascher, 2000, using payroll data; Card and Krueger, 2000) shows how measurement and the choice of comparison group matter. Modern versions use many state-level increases with staggered timing and heterogeneity-robust estimators (Cengiz, Dube, Lindner and Zipperer, 2019).

## Problem set
1. **Selection bias.** In observational data, people who enroll in job training earn less afterward than non-participants. Use the decomposition in Section 1 to explain how training could still have a positive ATT.

    <details class="solution" markdown="1"><summary>Solution</summary>

    The naive difference equals ATT plus \(E[Y(0)\mid D=1]-E[Y(0)\mid D=0]\). If participants have much lower earnings potential without training (for example, they enrolled after job loss), the selection term is large and negative, and it can outweigh a positive ATT. This is "Ashenfelter's dip" (Ashenfelter, 1978).
    </details>

2. **Double robustness.** Show that the AIPW estimator is consistent for the ATE if \(\widehat\mu_d\) is correct but \(\widehat e\) is misspecified.

    <details class="solution" markdown="1"><summary>Solution</summary>

    If \(\mu_d\) is correct, \(E[D(Y-\mu_1(X))/e^*(X)]=E\{D\,E[Y-\mu_1(X)\mid D=1,X]/e^*(X)\}=0\) for any \(e^*\), so the correction terms have mean zero and the estimator converges to \(E[\mu_1(X)-\mu_0(X)]=ATE\). The symmetric argument handles a correct propensity score with a wrong outcome model.
    </details>

3. **Negative weights.** Two groups adopt treatment: group A in period 2, group B in period 3; \(T=3\), no never-treated group. Effects are 1 in the first treated period and 3 in the second. Describe the 2×2 comparisons in TWFE and explain how a "forbidden" comparison biases \(\widehat\tau\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    The comparisons are: A vs not-yet-treated B for periods 1→2 (clean, effect 1); B vs already-treated A for periods 2→3. In the second, A's outcome changes by its effect growth \(3-1=2\), so the comparison yields \(1-2=-1\) for B's true effect of 1. The forbidden comparison enters with positive weight and pulls \(\widehat\tau\) down — it can even make it negative while all true effects are positive.
    </details>

4. **Fuzzy RD as IV.** Show that the fuzzy RD estimand equals the Wald estimand with \(Z=1\{X\ge c\}\) in a shrinking window around \(c\), and state the monotonicity assumption.

    <details class="solution" markdown="1"><summary>Solution</summary>

    In a window \((c-h,c+h)\) with \(h\to0\), \(Z\) is as good as randomly assigned by continuity. The ratio of the jumps in \(E[Y\mid X]\) and \(E[D\mid X]\) is the Wald ratio. Monotonicity: crossing the cutoff never *reduces* any unit's treatment take-up, so there are no defiers at \(c\).
    </details>

5. **Manipulation.** A scholarship is awarded to students scoring at least 80 on an exam graded by their own teachers. The density of scores shows excess mass at 80–81. What does this imply for the RD, and what can you do?

    <details class="solution" markdown="1"><summary>Solution</summary>

    Teachers likely push marginal students over the threshold, so students just above differ systematically from those just below: continuity fails. Options: use a running variable that cannot be manipulated (an externally graded test), use a "donut" RD excluding observations near the cutoff (with caution), or bound the effect under assumptions about the manipulators (Gerard, Rokkanen and Rothe, 2020).
    </details>

## Computational lab
**Goal:** see the TWFE bias under staggered adoption, and fix it with a clean-comparison estimator.

```python
import numpy as np
rng = np.random.default_rng(13)
N, T = 300, 10
cohort = rng.choice([4, 7, 99], size=N)                  # adoption period (99 = never treated)
alpha = rng.normal(size=N)[:, None]
lam = np.linspace(0, 1, T)[None, :]
t = np.arange(T)[None, :]
D = (t >= cohort[:, None]).astype(float)
effect = D * 0.5 * (t - cohort[:, None] + 1)              # effects grow with time since adoption
Y = alpha + lam + effect + rng.normal(0, 0.5, (N, T))

def twfe(Y, D):
    dm = lambda M: M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    y, d = dm(Y), dm(D)
    return (y * d).sum() / (d * d).sum()

true_att = effect[D == 1].mean()
print("true ATT (average over treated cells):", round(true_att, 3))
print("static TWFE:", round(twfe(Y, D), 3))

# imputation estimator: fit unit + time effects on untreated cells only, impute Y(0)
mask = D == 0
a, l = np.zeros(N), np.zeros(T)
for _ in range(200):                                      # alternating least squares
    a = np.nansum(np.where(mask, Y - l, np.nan), 1) / mask.sum(1)
    l = np.nansum(np.where(mask, Y - a[:, None], np.nan), 0) / mask.sum(0)
tau_hat = (Y - a[:, None] - l[None, :])[D == 1].mean()
print("imputation estimator:", round(tau_hat, 3))
```

Tasks: (i) explain why the static TWFE estimate is well below the true ATT; (ii) make the effects constant over time and show that TWFE becomes approximately unbiased; (iii) estimate group-time ATTs with the `csdid` package (Python), `did` (R) or `csdid` (Stata), and plot the event study; (iv) simulate a sharp RD and estimate it with `rdrobust`, comparing conventional and robust bias-corrected intervals.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 13.1 — LaLonde's job training data: can observational methods recover an experimental benchmark?

*Data:* `jtrain2` — the NSW randomized experiment (185 treated, 260 controls); `jtrain3` — the same treated men with 2,490 non-experimental controls from the PSID. Outcome: 1978 earnings `re78` (thousands of dollars).

1. Estimate the experimental effect of training on `re78` (difference in means and with covariates).
2. Using `jtrain3`, estimate the effect by (a) a naive difference, (b) regression adjustment, and (c) a doubly robust AIPW estimator with a logit propensity score. Covariates: `age`, `agesq`, `educ`, `black`, `hisp`, `married`, `re74`, `re75`, `unem74`, `unem75`.
3. Check overlap: what share of PSID controls have propensity scores below 0.01? Trim to \(0.01\le\widehat e\le0.99\) and re-estimate.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
exp_, obs = wd.data("jtrain2"), wd.data("jtrain3")
X = "age + agesq + educ + black + hisp + married + re74 + re75 + unem74 + unem75"
# TODO: experimental ATE; then on obs: ps = smf.logit(f"train ~ {X}", obs).fit().predict()
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
exp_, obs = wd.data("jtrain2"), wd.data("jtrain3")
exp_["agesq"] = exp_["age"]**2
X = "age + agesq + educ + black + hisp + married + re74 + re75 + unem74 + unem75"
e1 = smf.ols("re78 ~ train", data=exp_).fit(cov_type="HC1"); e2 = smf.ols(f"re78 ~ train + {X}", data=exp_).fit(cov_type="HC1")
print(f"EXPERIMENT   diff in means {e1.params['train']:.3f} (se {e1.bse['train']:.3f});  with covariates {e2.params['train']:.3f}")

def aipw(d):
    ps = smf.logit(f"train ~ {X}", data=d).fit(disp=0).predict(d)
    m1 = smf.ols(f"re78 ~ {X}", data=d[d.train == 1]).fit().predict(d)
    m0 = smf.ols(f"re78 ~ {X}", data=d[d.train == 0]).fit().predict(d)
    D, Y = d["train"], d["re78"]
    ra_att = (Y - m0)[D == 1].mean()                                     # regression adjustment, ATT
    w = ps / (1 - ps)                                                    # doubly robust ATT
    dr_att = ((D * (Y - m0)).sum() - ((1 - D) * w * (Y - m0)).sum()) / D.sum()
    return ps, ra_att, dr_att

naive = obs.loc[obs.train == 1, "re78"].mean() - obs.loc[obs.train == 0, "re78"].mean()
ps, ra, dr = aipw(obs)
print(f"PSID CONTROLS naive {naive:.3f};  regression adjustment ATT {ra:.3f};  doubly robust ATT {dr:.3f}")
print(f"share of PSID controls with propensity < 0.01: {(ps[obs.train == 0] < 0.01).mean():.3f}")
trim = obs[(ps >= 0.01) & (ps <= 0.99)].copy()
_, ra_t, dr_t = aipw(trim)
print(f"TRIMMED (n = {len(trim)}, treated {int(trim.train.sum())}): regression adjustment {ra_t:.3f}; doubly robust {dr_t:.3f}")
```

Output:

```text
EXPERIMENT   diff in means 1.794 (se 0.671);  with covariates 1.670
PSID CONTROLS naive -15.205;  regression adjustment ATT 0.894;  doubly robust ATT 3.654
share of PSID controls with propensity < 0.01: 0.814
TRIMMED (n = 646, treated 182): regression adjustment 1.785; doubly robust 2.497
```

The experiment gives an effect of about \$1,800 on 1978 earnings. With PSID controls, the naive comparison is hugely
**negative**: trainees were far poorer than the average PSID household — classic selection bias. Regression adjustment and
the doubly robust estimator move a long way toward the benchmark, but the answer depends on modeling choices, and most
PSID controls have propensity scores near zero: there is almost no overlap. Trimming to the region of common support
changes the estimates noticeably. This is LaLonde's (1986) lesson, refined by Dehejia and Wahba (1999): observational
methods can work only when comparable controls exist, and overlap diagnostics must come first.

</details>

### Exercise 13.2 — Difference-in-differences: an incinerator and house prices

*Data:* `kielmc` — 321 house sales in North Andover, Massachusetts, in 1978 (before rumors of an incinerator) and 1981 (during construction); `nearinc` = within 3 miles of the site.

1. Compute the 2×2 DiD estimate of the incinerator's effect on real house prices `rprice` from the four group means.
2. Estimate it by regression, `rprice ~ nearinc * y81`, with robust standard errors.
3. Add house characteristics (`age`, `agesq`, `intst`, `land`, `area`, `rooms`, `baths`). How does the estimate change, and why?
4. Use `log(rprice)` to express the effect in percent. Which scale makes parallel trends more plausible?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("kielmc")
means = df.groupby(["nearinc", "y81"])["rprice"].mean().unstack()
# TODO: DiD from means; regression with interaction; add controls
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
df = wd.data("kielmc")
mu = df.groupby(["nearinc", "y81"])["rprice"].mean().unstack()
print(mu.round(0).rename(index={0: "far", 1: "near"}, columns={0: "1978", 1: "1981"}))
did = (mu.loc[1, 1] - mu.loc[1, 0]) - (mu.loc[0, 1] - mu.loc[0, 0])
print(f"\nDiD from means: {did:,.0f} dollars")
r1 = smf.ols("rprice ~ nearinc * y81", data=df).fit(cov_type="HC1")
print(f"regression:       {r1.params['nearinc:y81']:,.0f} (robust se {r1.bse['nearinc:y81']:,.0f})")
ctrl = "age + I(age**2) + intst + land + area + rooms + baths"
r2 = smf.ols(f"rprice ~ nearinc * y81 + {ctrl}", data=df).fit(cov_type="HC1")
print(f"with controls:    {r2.params['nearinc:y81']:,.0f} (robust se {r2.bse['nearinc:y81']:,.0f})")
r3 = smf.ols(f"np.log(rprice) ~ nearinc * y81 + {ctrl}", data=df).fit(cov_type="HC1")
b = r3.params["nearinc:y81"]
print(f"log specification with controls: {b:.3f} -> {100*(np.exp(b)-1):.1f}% (robust se {r3.bse['nearinc:y81']:.3f})")
```

Output:

```text
y81         1978      1981
nearinc                   
far      82517.0  101308.0
near     63693.0   70619.0

DiD from means: -11,864 dollars
regression:       -11,864 (robust se 8,636)
with controls:    -14,178 (robust se 6,512)
log specification with controls: -0.093 -> -8.8% (robust se 0.061)
```

Houses near the site were already cheaper in 1978. The DiD removes that level difference and attributes the change in
the gap — a fall of roughly \$12,000 in real (1978) dollars — to the incinerator. Without controls the estimate is
imprecise. Adding house characteristics makes it larger and more precise, because the mix of houses sold near the site
changed between the two years. In logs with controls, the effect is about \(-9\%\), but with a standard error of 0.06 it is not statistically
significant: the evidence is suggestive rather than conclusive with 321 sales.
Parallel trends is an assumption about counterfactual price paths; the log scale is often more plausible for prices,
because it allows near and far houses to share common *percentage* appreciation. With only two periods, pre-trends cannot
be checked — a key limitation (Kiel and McClain, 1995; Wooldridge, Example 13.3).

</details>

## Reading
**Required:** Angrist and Pischke, *Mostly Harmless Econometrics*, chapters 2, 3.2–3.3, 5 and 6; Cunningham, *Causal Inference: The Mixtape* (2021), chapters 4–6 and 9–10.
**Further:** Imbens and Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences* (2015); Card and Krueger (1994), *AER* 84(4); Goodman-Bacon (2021), *Journal of Econometrics* 225(2); Callaway and Sant'Anna (2021), *Journal of Econometrics* 225(2); de Chaisemartin and D'Haultfœuille (2020), *AER* 110(9); Roth, Sant'Anna, Bilinski and Poe (2023), "What's trending in difference-in-differences?", *Journal of Econometrics* 235(2); Abadie, Diamond and Hainmueller (2010), *JASA* 105(490); Calonico, Cattaneo and Titiunik (2014), *Econometrica* 82(6); Lee and Lemieux (2010), "Regression discontinuity designs in economics", *Journal of Economic Literature* 48(2); Chernozhukov et al. (2018), "Double/debiased machine learning", *Econometrics Journal* 21(1).
