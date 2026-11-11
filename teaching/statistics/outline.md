# Topic Outline

A condensed map of the core definitions and results in Statistics. Each topic is developed in full in the [lecture notes](lecture-notes.md).


**Level:** Advanced undergraduate and graduate foundation. **Coverage:** Probability, sampling distributions, estimation, testing, Bayesian inference, experimental design, ANOVA, regression, nonparametric methods, and power.

## 1. Probability model
A probability space is (Ω,F,P). For a random variable X, its CDF is F(x)=P(X≤x). Expectation E[g(X)] is a sum for discrete variables or an integral for continuous variables. Define variance, covariance, correlation, conditional density, independence, and conditional independence. Use the tower property E[Y]=E[E(Y|X)] and total variance Var(Y)=E[Var(Y|X)]+Var(E[Y|X]).

## 2. Common distributions
Bernoulli, Binomial, Poisson, Uniform, Exponential, Gamma, Beta, Normal, Lognormal, Chi-square, t, F, Multinomial, multivariate Normal, and Wishart. For each derive support, normalizing constant, mean, variance, and a data-generating interpretation. Example: if X~Binomial(n,p), E(X)=np and Var(X)=np(1-p), obtainable by writing X as a sum of independent Bernoulli indicators.

## 3. Multivariate probability
Joint and marginal densities, conditional distributions, transformations and Jacobians, covariance matrices, correlation matrices, conditional Gaussian distributions, and order statistics. For random vector X, Σ=E[(X-μ)(X-μ)'] is positive semidefinite because a'Σa=Var(a'X)≥0.

## 4. Sampling theory and asymptotics
Random sampling, sampling distributions, LLN, CLT, Slutsky, continuous mapping, delta method, bootstrap and permutation inference. For i.i.d. observations with finite variance, sqrt(n)(Xbar-μ) converges in distribution to N(0,σ²). Explain when dependence requires different asymptotic theory.

## 5. Point estimation
Unbiasedness, consistency, efficiency, sufficiency, completeness, robustness, method of moments, and maximum likelihood. Likelihood L(θ)=Πf(x_i;θ); log-likelihood ℓ(θ)=Σlog f(x_i;θ). Score U=∂ℓ/∂θ and Fisher information I=E[UU']. Under regularity, sqrt(n)(θhat-θ0) is asymptotically Normal with covariance I(θ0)^(-1). Discuss boundary parameters and misspecification.

## 6. Confidence intervals and testing
Coverage, standard errors, Type I and II error, power, p-values, z/t/chi-square/F tests, likelihood-ratio, Wald and score tests, equivalence tests, and multiple testing. A p-value is computed under the null; it is not the probability that the null is true. Distinguish statistical from practical significance.

## 7. Bayesian inference
Posterior p(θ|y) proportional to likelihood times prior. Derive Beta-Binomial and Normal-Normal conjugacy. Cover credible intervals, posterior prediction, Bayes factors, hierarchical models, prior sensitivity, MCMC, effective sample size, and convergence diagnostics. Explain the conditioning of posterior statements on model and prior assumptions.

## 8. Experimental design
Randomization, blocking, replication, allocation, factorial designs, nested factors, Latin squares, split plots, and power. Randomized block model Y_ij=μ+τ_i+β_j+ε_ij. Two-factor model includes main effects and interaction. Distinguish fixed and random effects and explain estimable contrasts.

## 9. ANOVA and multiple comparisons
One-way ANOVA partitions total sum of squares into between- and within-group components; F=MS_between/MS_within. State independence, variance, and residual distribution assumptions. Use planned contrasts or multiplicity adjustments such as Tukey, Holm, Bonferroni, and Dunnett. Avoid unadjusted multiple testing.

## 10. Regression and generalized linear models
Linear regression, residual diagnostics, leverage, collinearity, robust covariance. GLM: g(E[Y|X])=X'β. Logistic model log[p/(1-p)]=X'β; exp(β_j) is a conditional odds ratio. Poisson log-link models, exposure offsets, overdispersion, negative binomial regression, and robust inference.

## 11. Nonparametric methods
Empirical CDF, quantiles, kernel density, bootstrap, permutation tests, sign and Wilcoxon tests, Mann-Whitney, Kruskal-Wallis, and rank-based factorial procedures. Clarify that rank procedures do not universally test equality of means.

## 12. Power and sample size
Power=1-β. Planning requires target effect, variance, alpha, desired power, design, allocation, and multiplicity. Explain standardized effect sizes, attrition, clustering/design effects, and why post-hoc observed power is not informative beyond the test result.

## Worked exercises
1. Derive Binomial mean and variance using indicator variables.
2. Derive the Normal mean confidence interval with known variance and explain the t interval when variance is estimated.
3. Derive Bernoulli MLE p_hat=Xbar by differentiating the log-likelihood.
4. For k groups and N observations, state ANOVA degrees of freedom: k-1, N-k, N-1.
5. Interpret logistic coefficient β_j through exp(β_j), holding other regressors fixed.

## Computing labs
R: stats, MASS, car, emmeans, lme4, boot. Python: numpy, scipy.stats, pandas, statsmodels, scikit-learn, pymc. Every lab includes data validation, model assumptions, diagnostics, effect sizes, uncertainty intervals, and reproducible code.

## Assessment and references
Problem sets 25%; design/power report 20%; computational labs 20%; midterm 15%; final project 20%. References: Casella and Berger, Statistical Inference; Rice, Mathematical Statistics and Data Analysis; Wasserman, All of Statistics; Lehmann and Romano, Testing Statistical Hypotheses; Montgomery, Design and Analysis of Experiments; Gelman et al., Bayesian Data Analysis.
