# Econometrics — Assignments

## Overview
There are seven problem sets, one for every two lecture sessions, plus a replication project. Each problem set mixes **theory** (derivations and proofs), **interpretation** (reading and critiquing estimates), and **computation** (simulation or estimation on public data). The exercises at the end of each set of lecture notes are practice problems with full solutions. The problems below are the graded assignments; solution guidance is released after the due date.

**Policies.** Problem sets are due one week after the second lecture they cover. You may discuss problems with classmates, but write up your own solutions and code, and list your collaborators. Submit code as a single script that runs from start to finish with relative paths and a fixed seed (see Lecture 14). Numerical answers without an econometric argument receive no credit.

**Grading rubric for every problem set.** Correct estimand and notation 20%; derivation and mathematical validity 30%; identification and assumptions 20%; interpretation and diagnostics 20%; reproducibility and presentation 10%.

| Problem set | Lectures | Topic |
|---|---|---|
| PS 1 | 1–2 | Projection, OLS algebra, asymptotic inference |
| PS 2 | 3–4 | Functional form, omitted-variable bias, robust and clustered inference |
| PS 3 | 5–6 | Instrumental variables, weak instruments, simultaneous equations |
| PS 4 | 7–8 | Panel data, fixed effects, dynamic panels and GMM |
| PS 5 | 9–10 | ARMA models, forecasting, unit roots and breaks |
| PS 6 | 11–12 | VARs, structural identification, cointegration |
| PS 7 | 13–14 | Causal designs, Monte Carlo, multiple testing |
| Project | all | Replication and extension of a published paper |

## Problem Set 1 — Projection and inference (Lectures 1–2)
**1.1 (Projection with a nonlinear CEF).** Let \(X\sim\text{Exponential}(1)\) and \(E[Y\mid X]=\log(1+X)\). (a) Derive the BLP coefficients of \(Y\) on \((1,X)\) as integrals and evaluate them numerically. (b) Show that the projection error is uncorrelated with \(X\) but that \(E[u\mid X]\neq0\). (c) Explain what OLS estimates in this model and why the result is still useful.

**1.2 (FWL and leverage).** Prove that adding a dummy variable for observation \(i\) alone to a regression sets \(\widehat u_i=0\) and gives the remaining coefficients as if observation \(i\) were deleted. Use this to derive the leave-one-out residual \(\widehat u_{(i)}=\widehat u_i/(1-h_{ii})\).

**1.3 (Delta method, computational).** Using the Current Population Survey extract of your choice (or Wooldridge's `wage2` data), estimate \(\log w=\beta_0+\beta_1exper+\beta_2exper^2+\beta_3educ+u\). Report the turning point of the experience profile with a delta-method 95% interval and a pairs-bootstrap percentile-\(t\) interval (999 draws). Comment on any differences.

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

1.1 \(\beta_1=\operatorname{Cov}(X,\log(1+X))/\operatorname{Var}(X)\), where \(\operatorname{Var}(X)=1\) and \(E[\log(1+X)]=e\,E_1(1)\approx0.596\) (\(E_1\) the exponential integral). Evaluate \(E[X\log(1+X)]\) numerically, giving \(\beta_1\approx0.40\). (c) OLS estimates the best linear approximation to a concave CEF. It is useful as a summary of the average slope, weighted by the design, but not as a structural marginal effect at a particular \(x\).

1.2 By FWL, partialling out the dummy \(e_i\) deletes row \(i\) from every other regressor. The dummy's coefficient absorbs \(y_i-X_i'\widehat\beta_{(i)}\), so \(\widehat u_i=0\). The leave-one-out formula follows from the Sherman–Morrison identity applied to \((X'X-X_iX_i')^{-1}\).

1.3 With the delta method, the interval is symmetric by construction. The percentile-\(t\) interval is typically asymmetric and wider when \(\widehat\beta_2\) is imprecise, reflecting the ratio form of the turning point.
</details>

## Problem Set 2 — Specification and robust inference (Lectures 3–4)
**2.1 (Signing bias).** In a regression of country growth on initial income and investment, explain the likely direction of bias in the investment coefficient from (a) omitting human capital and (b) classical measurement error in investment. Combine the two effects in a single formula and state the condition under which the OLS estimate is unbiased by accident.

**2.2 (Moulton in practice).** Using Project STAR data (student-level test scores with classroom assignment), estimate the effect of small classes with (a) HC1, (b) classroom-clustered, and (c) school-clustered standard errors. Report the intra-class correlation of the residual and compare the ratio of standard errors with the Moulton formula. Which level of clustering is justified by the design, and why?

**2.3 (Leverage and HC3).** Simulate \(n=40\) observations with one high-leverage point. Show by Monte Carlo (\(R=5{,}000\)) that HC0 under-covers and HC3 nearly fixes it. Report Monte Carlo standard errors.

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

2.1 Human capital is positively related to both growth and investment, giving upward bias. Measurement error attenuates. With reliability \(\lambda\) of investment after partialling out initial income, \(\operatorname{plim}\widehat\beta=\lambda(\beta+\gamma\delta)\), where \(\gamma\delta\) is the OVB term. It is accidentally unbiased when \(\lambda(\beta+\gamma\delta)=\beta\).

2.2 Treatment was randomized **within schools, across classrooms**, so the assignment unit is the classroom: cluster there and include school fixed effects. School-level clustering is not required by the design and leaves about 79 clusters with much noisier variance estimates (Abadie et al., 2023).

2.3 With one point at \(h_{ii}\approx0.6\), HC0 coverage is typically about 0.85–0.88, while HC3 is about 0.93–0.95.
</details>

## Problem Set 3 — Instrumental variables (Lectures 5–6)
**3.1 (Proximity to college).** Using Card's (1995) NLS Young Men data, estimate the return to schooling by OLS and by IV, with an indicator for growing up near a four-year college as the instrument, controlling for experience, experience squared, race, region and urban residence. Report (a) the first stage with an effective \(F\), (b) 2SLS with robust standard errors, (c) the Anderson–Rubin 95% confidence set, and (d) a discussion of who the compliers are and of plausible violations of the exclusion restriction.

**3.2 (Many instruments).** Add interactions of the instrument with region and parental education to obtain about 20 instruments. Compare 2SLS, LIML and Fuller(1). Explain the differences using the theory of Lectures 5–6.

**3.3 (Order versus rank).** Construct a two-equation system in which the order condition holds for the first equation but the rank condition fails. Simulate it, and show what happens to the 2SLS estimate and the first-stage \(F\).

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

3.1 The IV estimate exceeds OLS (roughly 0.13 versus 0.07–0.08 in Card's specification), with a much larger standard error. The first stage is significant but not overwhelming. Compliers are men whose schooling depends on living near a college, likely from lower-income families with higher marginal returns. That is one explanation for IV > OLS, alongside measurement error. Exclusion concerns: college proximity correlates with urban labor markets and family background.

3.2 With many instruments, 2SLS moves toward OLS; LIML and Fuller stay closer to the just-identified estimate, with wider dispersion.

3.3 Example: the excluded variable enters the reduced form of the endogenous regressor with coefficient zero. The 2SLS estimate is then centered near the OLS limit, with an erratic distribution, and the first-stage \(F\) is approximately \(\chi^2_\ell/\ell\).
</details>

## Problem Set 4 — Panel data (Lectures 7–8)
**4.1 (Within versus between).** Prove that the RE estimator is a matrix-weighted average of the within and between estimators. Give the weights.

**4.2 (Union premium).** Using the NLSY or PSID worker panel (e.g., Wooldridge's `wagepan`), estimate the union wage premium by pooled OLS, RE, FE and FD with unit-clustered standard errors. Run the Mundlak test. Interpret the pattern of estimates in light of selection and measurement error.

**4.3 (Dynamic panel).** Using the Arellano–Bond employment data (`abdata`), estimate the labor demand equation by FE, difference GMM and system GMM. For the GMM estimators, report results with all lags and with collapsed instruments limited to lags 2–4, along with instrument counts, AR(1)/AR(2) tests, Hansen tests and Windmeijer-corrected standard errors. Which estimates do you believe, and why?

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

4.1 \(\widehat\beta_{RE}=W\widehat\beta_B+(I-W)\widehat\beta_W\), with \(W=[X_W'X_W+\psi X_B'X_B]^{-1}\psi X_B'X_B\), \(\psi=\sigma_u^2/(\sigma_u^2+T\sigma_c^2)\), where \(X_W\) and \(X_B\) are the within and between design matrices.

4.2 Expect pooled > RE > FE ≈ FD (FD smallest if measurement error dominates). A Mundlak rejection indicates correlation between the unit effect and union status.

4.3 Results should bracket the lagged-employment coefficient between FE and pooled OLS. Estimates that move substantially when instruments are collapsed signal proliferation.
</details>

## Problem Set 5 — Univariate time series (Lectures 9–10)
**5.1 (ARMA identification).** Derive the ACF of an ARMA(1,1). Show that the ACF decays geometrically after lag 1, with the lag-1 value depending on both parameters.

**5.2 (Inflation forecasting).** Using quarterly U.S. CPI inflation from FRED (1960–present), compare the pseudo out-of-sample forecasts of an AR(\(p\)) chosen by BIC at each origin, a random walk (Atkeson–Ohanian), and the IMA(1,1) of Stock and Watson (2007), at horizons 1 and 4, using rolling 20-year windows. Report relative RMSFEs and Diebold–Mariano tests with HAC standard errors.

**5.3 (Unit roots and breaks).** For the log real exchange rate of a country of your choice against the U.S. dollar, run ADF, DF-GLS and KPSS tests, then a Zivot–Andrews test. Report a half-life of deviations with a bias-corrected estimate of the AR coefficient. Relate your results to the "PPP puzzle" (Rogoff, 1996).

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

5.1 \(\rho_1=(1+\phi\theta)(\phi+\theta)/(1+2\phi\theta+\theta^2)\) and \(\rho_j=\phi\rho_{j-1}\) for \(j\ge2\).

5.2 Since the mid-1980s, a random walk or IMA(1,1) is very hard to beat at a 4-quarter horizon. AR models do relatively better in high-volatility periods.

5.3 Typical half-lives are 3–5 years, with wide intervals: unit-root tests rarely reject in post-Bretton Woods samples, while panel or long-span data more often do.
</details>

## Problem Set 6 — Multivariate time series (Lectures 11–12)
**6.1 (Long-run restrictions).** In a bivariate VAR in output growth and unemployment, derive the Blanchard–Quah impact matrix in terms of \(\Sigma_u\) and \(\Psi(1)\). Explain what happens to the identified shocks if the VAR is estimated with too few lags.

**6.2 (Monetary policy).** Estimate a quarterly VAR in log real GDP, log GDP deflator, a commodity price index and the federal funds rate (1965–2007). Compare recursive identification with and without commodity prices, and document the price puzzle. Then estimate impulse responses by local projections using the Romer–Romer (2004) shock series, with HAC bands. Discuss the differences.

**6.3 (Term structure).** Test whether 3-month and 10-year Treasury yields are cointegrated with cointegrating vector \((1,-1)\) using both Engle–Granger and Johansen. Estimate the VECM and determine which rate adjusts. Interpret the result in light of the expectations hypothesis.

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

6.1 \(\Theta(1)=\operatorname{chol}(\Psi(1)\Sigma_u\Psi(1)')\) and \(B_0^{-1}=\Psi(1)^{-1}\Theta(1)\). Too few lags bias \(\Psi(1)\), and long-run identification is very sensitive to it (Faust and Leeper, 1997).

6.2 Without commodity prices, the price level typically rises for several quarters after a tightening. LP responses are noisier but have similar output troughs.

6.3 The spread is usually found to be stationary. The short rate tends to adjust more, but results depend on the sample and on regime shifts in monetary policy.
</details>

## Problem Set 7 — Causal designs and research practice (Lectures 13–14)
**7.1 (Staggered DiD).** Using county-level data on the staggered adoption of a policy (for example, the Callaway–Sant'Anna minimum wage data `mpdta`), estimate the effect on teen employment by static TWFE, a dynamic TWFE event study, Callaway–Sant'Anna group-time ATTs, and the imputation estimator. Produce a Goodman-Bacon decomposition of the TWFE estimate, and explain any differences.

**7.2 (RD).** Using the U.S. Senate incumbency data (Lee, 2008; available in `rdrobust`), estimate the incumbency advantage with local linear regression, MSE-optimal bandwidth and robust bias-corrected inference. Run a density test, covariate balance tests and placebo cutoffs.

**7.3 (Multiple outcomes).** For an experiment with 8 outcomes and the p-values you obtain from a provided dataset, report unadjusted, Holm, Romano–Wolf and Benjamini–Hochberg results, and a Kling–Liebman–Katz summary index. Explain which correction fits which question.

<details class="solution" markdown="1"><summary>Solution guidance (released after the due date)</summary>

7.1 With effects that grow after adoption, TWFE is attenuated relative to heterogeneity-robust estimators. The Bacon decomposition shows the weight on "later versus earlier treated" comparisons.

7.2 The estimated incumbency effect on the next vote share is about 7–8 percentage points, with no evidence of manipulation in the density of the margin.

7.3 FWER control (Holm, Romano–Wolf) suits confirmatory claims about each outcome; FDR suits exploratory screening; the index answers "did the program improve this domain overall?"
</details>

## Replication project
Choose a published empirical paper (from a list provided in Week 3, or your own with approval) that comes with a public replication package.

1. **Proposal (Week 5, 1 page):** the paper, its estimand, the identification strategy, and the data.
2. **Replication (Week 10):** reproduce the main table and one figure from raw data, writing your own code. Document every discrepancy.
3. **Extension (Week 14):** one substantive extension — a robustness check motivated by the theory of this course (for example, heterogeneity-robust DiD, weak-IV-robust inference, wild cluster bootstrap, or an alternative identification scheme).
4. **Deliverables:** a 12–15 page paper and a replication package meeting the standards of Lecture 14, which the instructor will run on a clean machine.

**Assessment:** Problem sets 30%; replication project 25%; midterm 20%; final 25%.
