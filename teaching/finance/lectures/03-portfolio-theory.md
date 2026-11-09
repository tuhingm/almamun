# Lecture 03 — Mean-Variance Portfolio Theory and the Efficient Frontier

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Derive minimum-variance portfolios and efficient frontiers; understand diversification, tangency portfolios, and estimation sensitivity.

## 1. Portfolio moments
For risky-asset weights \(w\), expected return and variance are
\[
E(R_p)=w'\mu,\qquad \sigma_p^2=w'\Sigma w.
\]
A fully invested portfolio satisfies \(\mathbf1'w=1\).

## 2. Global minimum-variance portfolio
Minimize \(w'\Sigma w\) subject to \(\mathbf1'w=1\). The Lagrangian FOC gives \(2\Sigma w-\lambda\mathbf1=0\), so
\[
w_{GMV}=\frac{\Sigma^{-1}\mathbf1}{\mathbf1'\Sigma^{-1}\mathbf1},
\quad
\sigma_{GMV}^2=\frac{1}{\mathbf1'\Sigma^{-1}\mathbf1}.
\]
This expression assumes invertible covariance matrix and unrestricted short positions.

## 3. Efficient frontier
For a target mean \(m\), solve minimum variance subject to \(\mathbf1'w=1\) and \(\mu'w=m\). The resulting frontier is a quadratic locus in mean-standard-deviation space. Only its upper branch is mean-variance efficient.

## 4. Risk-free asset and tangency portfolio
With risk-free rate \(r_f\), the tangency risky-asset weights are proportional to \(\Sigma^{-1}(\mu-r_f\mathbf1)\), normalized to sum to one when the normalization is well-defined. Investors combine the risk-free asset and the tangency portfolio according to risk tolerance.

## 5. Constraints and estimation error
Long-only, leverage, turnover, and position constraints alter the optimization problem. Estimated means are especially noisy; unconstrained optimizers can produce extreme weights. Report estimation window, frequency, shrinkage, and constraint set.

## 6. Worked example
For two assets, derive the GMV weight in asset 1:
\[
w_1^* = \frac{\sigma_2^2-\sigma_{12}}{\sigma_1^2+\sigma_2^2-2\sigma_{12}}.
\]
Evaluate for supplied variances and covariance, then verify the weights sum to one.

## 7. Computational lab
Estimate \(\mu\) and \(\Sigma\), calculate GMV and tangency portfolios, and compare unconstrained and long-only allocations out of sample. Plot the frontier and report turnover and realized risk.

## Problem set
1. Derive the GMV solution using a Lagrangian.
2. Derive the two-asset GMV weight.
3. Explain the role of covariance in diversification.
4. Discuss why estimated mean returns can destabilize portfolio weights.

## Further reading
Markowitz (1952); Merton (1972); Bodie, Kane and Marcus, *Investments*.
