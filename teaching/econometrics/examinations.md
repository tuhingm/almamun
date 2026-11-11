# Econometrics — Examinations

## Format and policies
| | Midterm | Final |
|---|---|---|
| Duration | 120 minutes | 180 minutes |
| Coverage | Lectures 1–6 | Entire course, emphasis on Lectures 7–14 |
| Weight | 20% | 25% |
| Materials | one double-sided formula sheet | two double-sided formula sheets |
| Statistical tables | provided (normal, \(t\), \(\chi^2\), \(F\), Dickey–Fuller, Johansen) | provided |

**Examination standards.** Define symbols and dimensions. Show derivations rather than stating formulas. Distinguish assumptions from testable implications. Interpret economic magnitudes and uncertainty. Do not claim causal identification from statistical significance alone. Credit is awarded for correct reasoning even when an arithmetic error occurs earlier.

## Midterm structure (100 marks)
- **Section A — Concepts (20).** Short questions on conditional mean independence versus orthogonality, projection matrices, Gauss–Markov assumptions, exact versus asymptotic inference, heteroskedasticity and clustering, and instrument validity.
- **Section B — Derivations (40).** Four derivations from: OLS and its conditional variance; the Wald/\(F\) statistic for linear restrictions; omitted-variable bias; attenuation bias; 2SLS and its asymptotic variance; the LATE theorem; the order and rank conditions.
- **Section C — Applied interpretation (20).** Interpret a regression table containing OLS and IV estimates, classical and robust standard errors, and first-stage diagnostics. Identify which claims are supported and which require more assumptions.
- **Section D — Research design (20).** Evaluate a proposed instrument or identification strategy: relevance, exclusion, weak identification, and the population the estimand describes.

## Final structure (100 marks)
- **Section A — Theory (30).** Three of five problems on panel transformations, GMM moment conditions, stationarity and unit roots, cointegration and VECMs, VAR identification, or causal estimands.
- **Section B — Interpretation (25).** An output package for a panel model and a time-series model: coefficients, standard errors, diagnostics, and identification.
- **Section C — Research design (25).** Design an empirical strategy for a policy question: estimand, data structure, identifying variation, estimator, variance estimator, diagnostics, and threats.
- **Section D — Reproducibility (20).** Audit a replication archive: data documentation, dependencies, seeds, transformations, and reporting.

## Sample midterm with solutions
**Question 1 (10 marks).** True, false or uncertain? Explain in at most three sentences each.
(a) "If \(E[Xu]=0\), then OLS is unbiased."
(b) "Robust standard errors correct for omitted-variable bias."
(c) "A first-stage \(F\) statistic of 12 guarantees that 2SLS \(t\)-tests have correct size."

<details class="solution" markdown="1"><summary>Solution</summary>

(a) **False.** Unbiasedness needs \(E[u\mid X]=0\). Orthogonality only delivers consistency (Lectures 1–2).
(b) **False.** They correct the variance estimator for heteroskedasticity. They do nothing about the probability limit of \(\widehat\beta\).
(c) **False.** \(F>10\) controls the *bias* of 2SLS relative to OLS (Stock–Yogo). For a 5% \(t\)-test to have correct size with \(|t|>1.96\), \(F\) must be around 104.7 (Lee et al., 2022). Otherwise use the tF adjustment or Anderson–Rubin.
</details>

**Question 2 (15 marks).** In \(y=X\beta+u\) with \(E[u\mid X]=0\) and \(\operatorname{Var}(u\mid X)=\sigma^2I\), let \(\tilde\beta=(X'X+\lambda I)^{-1}X'y\) for \(\lambda>0\) (ridge regression). (a) Is \(\tilde\beta\) unbiased? (b) Derive \(\operatorname{Var}(\tilde\beta\mid X)\). (c) Does this contradict Gauss–Markov?

<details class="solution" markdown="1"><summary>Solution</summary>

(a) \(E[\tilde\beta\mid X]=(X'X+\lambda I)^{-1}X'X\beta\neq\beta\) unless \(\beta=0\): biased toward zero.
(b) \(\sigma^2(X'X+\lambda I)^{-1}X'X(X'X+\lambda I)^{-1}\), which is smaller than \(\sigma^2(X'X)^{-1}\) in the positive semidefinite sense.
(c) No. Gauss–Markov compares *linear unbiased* estimators only. Ridge is linear but biased, and it can have lower MSE.
</details>

**Question 3 (15 marks).** True model: \(\log w=\beta_0+\beta_1S+\beta_2A+u\). You omit \(A\) (ability). The auxiliary regression of \(A\) on \(S\) has slope 0.15, and \(\beta_2=0.08\). Schooling is also measured with error, with reliability ratio 0.85. If the true \(\beta_1=0.07\), what is the probability limit of the short-regression OLS estimate? Which bias dominates?

<details class="solution" markdown="1"><summary>Solution</summary>

Treat the errors-in-variables problem as acting on the short regression. The OVB term is \(0.15\times0.08=0.012\), so with true schooling the short regression gives \(0.082\). Attenuation scales this by 0.85: \(0.082\times0.85\approx0.070\). The two biases almost exactly offset. (A full answer notes that the precise formula depends on how the measurement error interacts with the auxiliary regression, since the auxiliary slope of \(A\) on *observed* \(S\) is also attenuated. Under classical error independent of \(A\), both terms scale by the reliability ratio, giving \(0.85\times(0.07+0.012)\).)
</details>

**Question 4 (20 marks).** A study uses rainfall in the farmer's district as an instrument for agricultural income, to estimate the effect of income on children's schooling. The first-stage effective \(F\) is 31, the OLS estimate is 0.05 and the 2SLS estimate is 0.21 (s.e. 0.07).
(a) State the two instrument conditions in this context and assess each.
(b) Give two distinct reasons why 2SLS could exceed OLS.
(c) Who are the compliers, and what does that imply for policy relevance?

<details class="solution" markdown="1"><summary>Solution</summary>

(a) Relevance: rainfall shifts harvests and incomes. With \(F=31\), relevance is adequate for bias, though a tF adjustment of the \(t\)-test is advisable. Exclusion: rainfall could affect schooling directly (flooded roads, disease, child labor demand on the farm, school closures), so the exclusion restriction is doubtful.
(b) Measurement error in income (attenuating OLS); heterogeneous effects with a LATE for households whose income is sensitive to rainfall (poorer, credit-constrained households may have larger effects); a violated exclusion restriction (for example, through child labor demand).
(c) Compliers are households whose income responds to rainfall: rain-fed, non-diversified farmers. The estimate is informative for transfers to such households, less so for urban or diversified households.
</details>

**Question 5 (20 marks).** Consider the system \(q=\alpha p+\gamma y+u_d\) (demand), \(q=\beta p+\delta w+u_s\) (supply), with \(y,w\) exogenous.
(a) Derive the reduced form for \(p\).
(b) Check the order and rank conditions for each equation.
(c) If \(\delta=0\), what can be estimated consistently?

<details class="solution" markdown="1"><summary>Solution</summary>

(a) \(p=(\delta w-\gamma y+u_s-u_d)/(\alpha-\beta)\).
(b) Each equation excludes one exogenous variable and has one RHS endogenous variable: just identified by the order condition. Rank for demand requires that \(w\) moves \(p\): \(\delta\neq0\). Rank for supply requires \(\gamma\neq0\).
(c) If \(\delta=0\), demand is not identified (no supply shifter), but supply remains identified by \(y\) if \(\gamma\neq0\).
</details>

**Question 6 (20 marks).** You estimate the effect of a state-level policy on individual earnings with 1.2 million workers in 48 states over 15 years. Policy timing varies by state. Describe (a) the variance estimator you would use and why, (b) how many effective observations you have for inference, (c) one finite-sample check, and (d) one concern about the point estimate if you use static TWFE.

<details class="solution" markdown="1"><summary>Solution</summary>

(a) Cluster by state, the level of policy assignment, allowing arbitrary serial correlation within states. (b) Effectively 48 clusters, not 1.2 million or 720 state-years. (c) Wild cluster bootstrap p-values with the null imposed, or CR2 with Bell–McCaffrey degrees of freedom; check the number of treated clusters. (d) With staggered adoption and heterogeneous or dynamic effects, TWFE uses already-treated states as controls and can put negative weights on some group-time effects. Use Callaway–Sant'Anna, Sun–Abraham or imputation estimators (Lecture 13).
</details>

## Final examination question bank (selection)
1. Derive Nickell bias for \(T=2\), and show that Anderson–Hsiao IV is consistent. When is it weak?
2. Show that FE equals the Mundlak coefficient in a balanced panel, and describe a cluster-robust test of RE against FE.
3. Derive the ACF of an AR(2) via the Yule–Walker equations, and determine stationarity for given coefficients.
4. Explain why the Engle–Granger residual-based test needs non-standard critical values, and why OLS standard errors in the cointegrating regression are invalid.
5. In a 3-variable VAR, count the restrictions for exact identification. Compare Cholesky and proxy-SVAR identification of a monetary shock.
6. Prove that DiD identifies the ATT under parallel trends and no anticipation. Construct an example in which TWFE has the wrong sign.
7. Design a Monte Carlo study to compare RD inference with conventional and robust bias-corrected intervals. Specify the DGP, the metrics, \(R\), and Monte Carlo standard errors.
