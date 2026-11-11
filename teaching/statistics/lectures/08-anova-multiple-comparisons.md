# Lecture 08 — ANOVA, Linear Contrasts, and Multiple Comparisons

## Learning objectives
Derive one-way ANOVA; partition sums of squares; formulate contrasts; distinguish fixed and random effects; understand multiplicity adjustments and diagnostics.

## 1. One-way model
For group \(i=1,\ldots,k\),
\[
Y_{ij}=\mu+\tau_i+\varepsilon_{ij},\quad
\varepsilon_{ij}\overset{iid}{\sim}N(0,\sigma^2).
\]
A constraint such as \(\sum_i n_i\tau_i=0\) identifies the intercept and group effects. The omnibus null is equality of all group means.

## 2. Sum-of-squares decomposition
With group means \(\bar Y_i\) and grand mean \(\bar Y\),
\[
SS_T=\sum_i\sum_j(Y_{ij}-\bar Y)^2
=\sum_i n_i(\bar Y_i-\bar Y)^2+
\sum_i\sum_j(Y_{ij}-\bar Y_i)^2.
\]
Between-group and within-group degrees of freedom are \(k-1\) and \(N-k\). The statistic \(F=MS_B/MS_W\) compares explained group variation with residual variation.

## 3. Linear contrasts
A contrast is \(\psi=\sum_i c_i\mu_i\), with \(\sum_i c_i=0\). Its estimator is \(\widehat\psi=\sum_i c_i\bar Y_i\), with variance under the common-variance model
\[
Var(\widehat\psi)=\sigma^2\sum_i c_i^2/n_i.
\]
A planned contrast answers a targeted question; an omnibus rejection does not identify which groups differ.

## 4. Multiple comparisons
Tukey controls family-wise error for all pairwise comparisons under its model. Bonferroni tests each of \(m\) hypotheses at level \(\alpha/m\); Holm's step-down procedure is generally less conservative while controlling family-wise error. False-discovery-rate methods target a different error criterion.

## 5. Fixed and random effects
Fixed effects represent specific levels of direct interest. Random effects treat levels as draws from a population. In a random-intercept model, \(b_j\sim N(0,\tau^2)\), and \(ICC=\tau^2/(\tau^2+\sigma^2)\). This choice changes the estimand and covariance structure.

## 6. Diagnostics
Inspect residuals for unequal variance, outliers, dependence, and non-normality. Welch's ANOVA accommodates unequal group variances in one-way comparisons. Robust and permutation alternatives still require a clearly specified target hypothesis.

## Worked example
For three teaching methods with means 70, 76, and 82, the omnibus test evaluates equality of all three means. The contrast \((-1,0,1)\) compares endpoint methods and is not the same as testing every pair.

## Problem set
1. Derive the one-way sum-of-squares decomposition.
2. State the omnibus F-test hypotheses.
3. Calculate the variance of a specified contrast.
4. Compare Tukey, Bonferroni, and Holm.
5. Distinguish fixed and random treatment levels.

## Computational lab
Fit one-way ANOVA, compute planned contrasts, and compare Tukey and Holm-adjusted pairwise inference. Simulate unequal variances and compare classical and Welch procedures.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 8.1 — One-way ANOVA and Tukey comparisons: wages by occupation

*Data:* `wage1` — CPS 1976, with indicators `profocc` (professional), `clerocc` (clerical), `servocc` (service); the rest are grouped as "other".

1. Build an `occ` factor with four levels and compute group means and SDs of `lwage`.
2. Run a one-way ANOVA (\(F\) test). Report the ANOVA table and \(\eta^2\).
3. Check equal variances (Levene). Run Welch's ANOVA as a robust alternative.
4. Run Tukey's HSD to see which pairs differ, controlling the family-wise error rate.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
d = wd.data("wage1")
d["occ"] = np.select([d.profocc == 1, d.clerocc == 1, d.servocc == 1], ["professional", "clerical", "service"], "other")
# TODO: sm.stats.anova_lm(smf.ols("lwage ~ C(occ)", d).fit()); scipy.stats.levene; pairwise_tukeyhsd
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
d = wd.data("wage1")
d["occ"] = np.select([d.profocc == 1, d.clerocc == 1, d.servocc == 1], ["professional", "clerical", "service"], "other")
print(d.groupby("occ")["lwage"].agg(["size", "mean", "std"]).round(3).to_string())
a = sm.stats.anova_lm(smf.ols("lwage ~ C(occ)", data=d).fit())
print(a.round(4).to_string()); print(f"eta-squared = {a['sum_sq'].iloc[0] / a['sum_sq'].sum():.3f}")
lev = stats.levene(*[g["lwage"] for _, g in d.groupby("occ")])
wel = anova_oneway(d["lwage"], d["occ"], use_var="unequal")
print(f"Levene: W = {lev.statistic:.2f}, p = {lev.pvalue:.4f};  Welch ANOVA: F = {wel.statistic:.2f}, p = {wel.pvalue:.2e}")
print(pairwise_tukeyhsd(d["lwage"], d["occ"]).summary())
```

Output:

```text
size   mean    std
occ                             
clerical        88  1.482  0.364
other          171  1.526  0.437
professional   193  1.934  0.548
service         74  1.206  0.404
             df    sum_sq  mean_sq        F  PR(>F)
C(occ)      3.0   34.8837  11.6279  53.5035     0.0
Residual  522.0  113.4460   0.2173      NaN     NaN
eta-squared = 0.235
Levene: W = 9.88, p = 0.0000;  Welch ANOVA: F = 50.38, p = 3.90e-25
      Multiple Comparison of Means - Tukey HSD, FWER=0.05       
================================================================
   group1       group2    meandiff p-adj   lower   upper  reject
----------------------------------------------------------------
    clerical        other   0.0439 0.8897 -0.1137  0.2016  False
    clerical professional   0.4515    0.0   0.297  0.6061   True
    clerical      service  -0.2767 0.0011 -0.4662 -0.0872   True
       other professional   0.4076    0.0  0.2814  0.5338   True
       other      service  -0.3207    0.0 -0.4878 -0.1535   True
professional      service  -0.7283    0.0 -0.8925  -0.564   True
----------------------------------------------------------------
```

Occupation explains about 24% of the variance of log wages (\(\eta^2\)), and the \(F\) test rejects equal means
overwhelmingly. Levene's test rejects equal variances — professional wages are more dispersed — so the Welch ANOVA is the
safer test; it gives the same conclusion. Tukey's HSD shows that every pair differs except clerical versus "other". The
professional premium over service work is about 0.73 log points, roughly double.

</details>

### Exercise 8.2 — Two-way ANOVA with interaction: gender, marriage and wages

*Data:* `wage1`.

1. Fit the two-way model `lwage ~ female * married` and produce a type-II ANOVA table.
2. Interpret the interaction: compute the four cell means and the "marriage premium" for men and for women.
3. Because the design is unbalanced, explain why type-I (sequential) and type-II sums of squares differ.

```python
import wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
d = wd.data("wage1")
m = smf.ols("lwage ~ C(female) * C(married)", data=d).fit()
# TODO: sm.stats.anova_lm(m, typ=2) and typ=1; cell means with groupby
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import wooldridge as wd, statsmodels.formula.api as smf, statsmodels.api as sm
d = wd.data("wage1")
m = smf.ols("lwage ~ C(female) * C(married)", data=d).fit()
print("type II:\n", sm.stats.anova_lm(m, typ=2).round(4).to_string())
print("type I (female first):\n", sm.stats.anova_lm(m, typ=1).round(4).to_string())
cm = d.groupby(["female", "married"])["lwage"].agg(["mean", "size"]).round(3)
print(cm.to_string())
mu = d.groupby(["female", "married"])["lwage"].mean()
print(f"marriage premium (log points): men {mu[(0,1)] - mu[(0,0)]:.3f}, women {mu[(1,1)] - mu[(1,0)]:.3f}")
```

Output:

```text
type II:
                         sum_sq     df        F  PR(>F)
C(female)              16.4821    1.0  73.7211     0.0
C(married)              6.6370    1.0  29.6860     0.0
C(female):C(married)    4.2752    1.0  19.1219     0.0
Residual              116.7056  522.0      NaN     NaN
type I (female first):
                          df    sum_sq  mean_sq        F  PR(>F)
C(female)               1.0   20.7120  20.7120  92.6405     0.0
C(married)              1.0    6.6370   6.6370  29.6860     0.0
C(female):C(married)    1.0    4.2752   4.2752  19.1219     0.0
Residual              522.0  116.7056   0.2236      NaN     NaN
                 mean  size
female married             
0      0        1.521    86
       1        1.947   188
1      0        1.389   120
       1        1.441   132
marriage premium (log points): men 0.427, women 0.052
```

The interaction is highly significant: marriage is associated with a 0.43 log-point (about 53%) wage premium for men but
almost none for women (0.05). Because the cells are unbalanced (86 to 188 observations), type-I sums of squares depend on
the order of terms: the sum of squares for `female` is larger when it enters first, because it absorbs variation shared
with `married`. Type-II sums of squares test each main effect adjusted for the other. These are associations — the male
marriage premium partly reflects selection of higher earners into marriage.

</details>

## Reading
Montgomery, *Design and Analysis of Experiments*; Hsu, *Multiple Comparisons*; Kutner et al., *Applied Linear Statistical Models*.