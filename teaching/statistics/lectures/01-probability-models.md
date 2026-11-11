# Lecture 01 — Probability Models, Random Variables, and Expectation

**Course:** Statistics: Probability, Mathematical Inference and Experimental Design  
**Level:** Graduate foundation

## Core question
How does a probability model translate uncertainty about outcomes into quantities that can be estimated and tested?

## Learning objectives
Define probability spaces and random variables; derive expectations and moments; work with conditional distributions; distinguish independence from zero correlation; apply iterated expectations and the law of total variance.

## 1. Probability spaces and events
A probability space is \((\Omega,\mathcal F,P)\). The sample space \(\Omega\) contains possible outcomes, \(\mathcal F\) is a sigma-algebra of measurable events, and \(P\) is a probability measure satisfying non-negativity, normalization, and countable additivity. For pairwise disjoint events \(A_j\),
\[
P\left(\bigcup_{j=1}^{\infty}A_j\right)=\sum_{j=1}^{\infty}P(A_j).
\]
The axioms imply monotonicity, complements, and the addition rule. Conditional probability is defined by \(P(A\mid B)=P(A\cap B)/P(B)\) when \(P(B)>0\). Bayes' rule follows by expressing the joint probability in two ways.

## 2. Random variables and distributions
A random variable is a measurable function \(X:\Omega\to\mathbb R\). Its cumulative distribution function is \(F_X(x)=P(X\le x)\). For a discrete variable, probabilities are represented by a probability mass function; for an absolutely continuous variable, by a density \(f_X\), with
\[
F_X(x)=\int_{-\infty}^{x}f_X(t)dt,\qquad P(a<X\le b)=\int_a^b f_X(t)dt.
\]
A density is not itself a probability at a point. The support and normalization condition must always be stated.

## 3. Expectation and moments
For discrete \(X\), \(E[X]=\sum_x xf_X(x)\); for continuous \(X\), \(E[X]=\int xf_X(x)dx\), when the expectation exists. For a measurable transformation \(g\), the law of the unconscious statistician gives \(E[g(X)]=\int g(x)f_X(x)dx\). The variance is \(\operatorname{Var}(X)=E[(X-\mu)^2]=E[X^2]-\mu^2\). Higher central moments describe skewness and kurtosis.

Linearity, \(E[aX+bY]=aE[X]+bE[Y]\), does not require independence. In contrast, \(E[XY]=E[X]E[Y]\) generally requires independence (or a suitable weaker condition).

## 4. Conditional expectation
For a discrete conditioning variable,
\[
E[Y\mid X=x]=\sum_y yP(Y=y\mid X=x).
\]
The conditional expectation \(E[Y\mid X]\) is a random variable and the best mean-square predictor of \(Y\) among functions measurable with respect to \(X\). It satisfies the tower property:
\[
E\{E(Y\mid X)\}=E(Y).
\]
The law of total variance is
\[
\operatorname{Var}(Y)=E[\operatorname{Var}(Y\mid X)]+\operatorname{Var}(E[Y\mid X]).
\]
The first term is average within-condition uncertainty; the second is variation explained by differences in conditional means.

## 5. Independence, covariance, and correlation
Independence means the joint distribution factors into marginal distributions. Zero covariance is weaker: it rules out linear association but not nonlinear dependence. For example, if \(X\) is symmetric around zero and \(Y=X^2\), covariance may be zero although \(Y\) is completely determined by \(X\). Correlation is covariance standardized by marginal standard deviations and is defined only when both variances are finite and positive.

## Worked example
Let \(X\sim Bernoulli(p)\), and let \(Y=a+bX+\varepsilon\), where \(E(\varepsilon\mid X)=0\). Then \(E(Y\mid X)=a+bX\), \(E(Y)=a+bp\), and
\[
\operatorname{Var}(Y)=b^2p(1-p)+E[\operatorname{Var}(\varepsilon\mid X)].
\]
This separates explained variation due to the binary regressor from average conditional residual variation.

## Problem set
1. Prove the complement and addition rules from the probability axioms.
2. Show that \(E[aX+b]=aE[X]+b\) and derive the variance transformation.
3. Construct dependent variables with zero covariance.
4. Derive the law of total variance using the tower property.
5. Explain why \(E[Y\mid X]\) is generally not equal to \(E[Y]\).

## Computational lab
Simulate Bernoulli and normal variables; compare empirical means, variances, and conditional means with their population values. Simulate \(Y=X^2\) for symmetric \(X\) and verify that sample correlation can be near zero despite dependence.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 1.1 — Bayes' rule with real data: smoking and low birth weight

*Data:* `bwght` — 1,388 U.S. births, 1988 National Health Interview Survey (`bwght` in ounces, `cigs` = cigarettes per day during pregnancy).

Define the events \(S\) = "mother smoked during pregnancy" (`cigs > 0`) and \(L\) = "low birth weight" (below 88 ounces, about 2.5 kg).

1. Estimate \(P(S)\), \(P(L)\), \(P(L\mid S)\) and \(P(L\mid S^c)\).
2. Use Bayes' rule to compute \(P(S\mid L)\), and check it against the direct sample frequency.
3. Compute the relative risk \(P(L\mid S)/P(L\mid S^c)\) and its 95% confidence interval on the log scale.
4. Are \(S\) and \(L\) independent? Test with a \(\chi^2\) test of independence.

```python
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
d = wd.data("bwght")
S, L = d["cigs"] > 0, d["bwght"] < 88
# TODO: probabilities, Bayes' rule, relative risk, stats.chi2_contingency(pd.crosstab(S, L))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd
from scipy import stats
d = wd.data("bwght")
S, L = d["cigs"] > 0, d["bwght"] < 88
pS, pL = S.mean(), L.mean()
pL_S, pL_nS = L[S].mean(), L[~S].mean()
print(f"P(S) = {pS:.4f}   P(L) = {pL:.4f}   P(L|S) = {pL_S:.4f}   P(L|not S) = {pL_nS:.4f}")
bayes = pL_S * pS / (pL_S * pS + pL_nS * (1 - pS))
print(f"P(S|L) by Bayes' rule = {bayes:.4f};  directly = {S[L].mean():.4f}")
a, b, c, e = (S & L).sum(), (S & ~L).sum(), (~S & L).sum(), (~S & ~L).sum()
rr = (a / (a + b)) / (c / (c + e)); se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+e))
print(f"relative risk = {rr:.2f}, 95% CI [{rr*np.exp(-1.96*se):.2f}, {rr*np.exp(1.96*se):.2f}]  (counts: {a} smokers with low weight)")
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(S, L))
print(f"chi-square test of independence: chi2({dof}) = {chi2:.2f}, p = {p:.4f}")
```

Output:

```text
P(S) = 0.1527   P(L) = 0.0598   P(L|S) = 0.0943   P(L|not S) = 0.0536
P(S|L) by Bayes' rule = 0.2410;  directly = 0.2410
relative risk = 1.76, 95% CI [1.09, 2.85]  (counts: 20 smokers with low weight)
chi-square test of independence: chi2(1) = 4.61, p = 0.0318
```

About 15% of mothers smoked, and 6% of babies had low birth weight. Bayes' rule and the direct frequency agree exactly,
as they must: \(P(S\mid L)\approx0.24\), so a quarter of low-weight babies had smoking mothers, against 15% of all babies.
The relative risk is about 1.8, with an interval from about 1.1 to 2.9, and the \(\chi^2\) test rejects independence at 5%.
Only 20 smokers had low-weight babies, so the interval is wide. This is an association: smoking mothers differ in income,
education and other risk factors, so a causal claim needs the tools of Econometrics Lecture 13.

</details>

### Exercise 1.2 — The law of total variance in a wage distribution

*Data:* `wage1` — 526 workers, CPS 1976.

1. Verify numerically that \(\operatorname{Var}(wage)=E[\operatorname{Var}(wage\mid educ)]+\operatorname{Var}(E[wage\mid educ])\), using sample moments with divisor \(n\).
2. What share of wage variance is "explained" by education cells? Compare with the \(R^2\) of a linear regression on `educ`, and explain why the cell-based share is larger.
3. Repeat with `lwage`. Why does the share change?

```python
import wooldridge as wd
d = wd.data("wage1")
g = d.groupby("educ")["wage"]
# TODO: within = (g.var(ddof=0) * g.size()).sum() / n ;  between = ((g.mean() - mean)**2 * g.size()).sum() / n
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("wage1"); n = len(d)
for y in ("wage", "lwage"):
    g = d.groupby("educ")[y]
    total = d[y].var(ddof=0)
    within = (g.var(ddof=0) * g.size()).sum() / n
    between = ((g.mean() - d[y].mean())**2 * g.size()).sum() / n
    r2 = smf.ols(f"{y} ~ educ", data=d).fit().rsquared
    print(f"{y:6s}: total {total:8.4f} = within {within:8.4f} + between {between:7.4f}  (check {within + between:8.4f})")
    print(f"        share explained by education cells {between/total:.3f}   vs linear R^2 {r2:.3f}")
```

Output:

```text
wage  : total  13.6130 = within  10.5644 + between  3.0485  (check  13.6130)
        share explained by education cells 0.224   vs linear R^2 0.165
lwage : total   0.2820 = within   0.2142 + between  0.0678  (check   0.2820)
        share explained by education cells 0.240   vs linear R^2 0.186
```

The decomposition holds exactly (the "check" column reproduces the total). Education cells account for about 22% of the
variance of wages, more than the linear \(R^2\) of 16.5%, because cell means capture the **nonlinear** CEF, while the
regression line can capture only its linear part. Logs raise both shares a little, because the log transformation makes
the CEF closer to linear and dampens the influence of very high wages on the total variance.

</details>

## Reading
Casella and Berger, *Statistical Inference*; DeGroot and Schervish, *Probability and Statistics*; Wasserman, *All of Statistics*.