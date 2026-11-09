# Statistics — Graduate Problem Sets and Solution Guidance

Show derivations and state assumptions. Numerical answers without a probability model or inferential interpretation are incomplete.

## A. Probability and distributions

**A1. Bernoulli moments.** For \(X\in\{0,1\}\), \(P(X=1)=p\), so \(E(X)=p\), \(E(X^2)=p\), and \(Var(X)=p-p^2=p(1-p)\).

**A2. Total variance.** Write \(Y-EY=[Y-E(Y|X)]+[E(Y|X)-EY]\), square, take expectations, and use the zero conditional expectation of the cross term. This yields \(Var(Y)=E[Var(Y|X)]+Var(E[Y|X])\).

**A3. Covariance matrix.** For any vector \(a\), \(a'\Sigma a=Var(a'X)\ge0\); hence \(\Sigma\) is positive semidefinite.

## B. Asymptotics and estimation

**B1. CLT.** For i.i.d. observations with finite variance, \(\sqrt n(\bar X-\mu)\to_dN(0,\sigma^2)\). The CLT concerns the normalized sample mean, not the raw observations.

**B2. Delta method.** If \(\sqrt n(\widehat\theta-\theta)\to_dN(0,V)\), then for differentiable \(g\), the transformed estimator has asymptotic variance \(g'(\theta)^2V\).

**B3. Bernoulli MLE.** The log likelihood is \(\ell(p)=y\log p+(n-y)\log(1-p)\). The score equation gives \(\widehat p=y/n\).

**B4. MSE.** \(MSE(\widehat\theta)=Var(\widehat\theta)+Bias(\widehat\theta)^2\).

## C. Testing and Bayesian inference

**C1. Coverage.** A 95% confidence procedure covers the fixed parameter in 95% of repeated samples under its assumptions; it is not a 95% posterior probability statement.

**C2. Beta-Binomial.** With prior \(Beta(a,b)\) and binomial data \(y\) successes in \(n\) trials, posterior is \(Beta(a+y,b+n-y)\).

**C3. Multiplicity.** Bonferroni controls family-wise error by testing each of \(m\) hypotheses at \(\alpha/m\). Holm applies a sequentially less conservative step-down procedure.

## D. Experimental design and ANOVA

**D1. Randomized blocks.** Blocking controls nuisance heterogeneity by comparing treatments within blocks; treatment assignment must be randomized according to the design.

**D2. ANOVA degrees of freedom.** For \(k\) groups and \(N\) observations, between-group df is \(k-1\), within-group df is \(N-k\), and total df is \(N-1\).

**D3. Contrast.** A linear contrast \(\sum_i c_i\mu_i\) requires \(\sum_i c_i=0\). State the contrast weights and variance.

## E. Regression and GLM

**E1. Logistic interpretation.** In \(logit(p_i)=x_i'\beta\), \(e^{\beta_j}\) is the conditional odds ratio for a one-unit increase in \(x_j\), holding other regressors fixed.

**E2. Marginal effect.** For continuous \(x_j\), \(\partial p_i/\partial x_{ij}=p_i(1-p_i)\beta_j\). Thus the probability effect varies with covariates.

**E3. Poisson offset.** An exposure offset enters with coefficient fixed at one on the log scale, modeling a rate rather than a raw count.

## F. Nonparametric, power, and mixed models

**F1. Rank tests.** Explain the null/estimand for the selected rank procedure; do not claim every rank test is a test of medians or means.

**F2. Power.** Power is the probability of rejecting the null under a specified alternative. State effect size, variance, alpha, design, and sample size.

**F3. Intraclass correlation.** In a random-intercept model, \(ICC=\tau^2/(\tau^2+\sigma^2)\).

**F4. PCA.** The first principal component maximizes \(a'Sa\) subject to \(a'a=1\); the solution is the eigenvector for the largest eigenvalue.

## Grading rubric
Mathematical correctness 30%; assumptions and model specification 20%; interpretation 20%; computational reproducibility 20%; clarity 10%.
