# Lecture 10 — Nonparametric Inference, Rank Tests, and Resampling

## Learning objectives
Use empirical distributions and quantiles; derive rank procedures; state the hypotheses actually tested; conduct permutation tests and bootstrap inference.

## 1. Empirical distribution
The empirical CDF is
\[
\widehat F_n(x)=n^{-1}\sum_{i=1}^n1(X_i\le x).
\]
It estimates the population distribution and supports quantile estimation, goodness-of-fit procedures, and resampling. Quantile conventions should be stated when finite-sample results depend on interpolation.

## 2. Sign and signed-rank tests
For paired data define \(D_i=X_i-Y_i\). The sign test uses signs of nonzero differences and targets a median-zero condition under its assumptions. The Wilcoxon signed-rank test ranks absolute differences and uses their signs. Its conventional location interpretation requires a symmetric distribution of paired differences; ties and zero differences need explicit treatment.

## 3. Mann–Whitney and Kruskal–Wallis
The Mann–Whitney statistic compares ranks in two independent groups and is related to \(P(X>Y)+\tfrac12P(X=Y)\). It is not universally a test of equal means or medians; a location-shift interpretation needs additional assumptions. Kruskal–Wallis extends rank comparisons to multiple independent groups.

## 4. Permutation inference
A permutation test constructs a reference distribution by reassigning labels according to exchangeability or the actual randomization design. Under a sharp null and valid assignment mechanism, randomization inference can be exact. Arbitrary permutation is not justified for dependent or selected observational data.

## 5. Bootstrap
The nonparametric bootstrap resamples with replacement from the empirical distribution and recomputes a statistic. Percentile, basic, and studentized intervals differ. Clustered or serially dependent data require resampling schemes that preserve the dependence structure.

## 6. Robustness and assumptions
Nonparametric methods still require assumptions such as independence, exchangeability, or symmetry, depending on the procedure. Robustness to outliers does not imply robustness to dependence or selection.

## Worked example
For paired before-after data, calculate differences, rank absolute nonzero differences, and sum positive ranks. The signed-rank statistic uses both direction and rank magnitude; it is not simply a test of the arithmetic mean difference.

## Problem set
1. Define the empirical CDF and prove monotonicity.
2. State the symmetry condition used by the signed-rank test.
3. Interpret the Mann–Whitney probability-of-superiority parameter.
4. Explain exchangeability in permutation inference.
5. Specify a bootstrap scheme for clustered observations.

## Computational lab
Compare t, sign, and signed-rank tests under normal, skewed, and contaminated distributions. Implement a randomization test and a cluster bootstrap.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 10.1 — Rank tests and distributional comparisons for skewed earnings

*Data:* `jtrain2` — NSW experiment.

1. Compare 1978 earnings of trainees and controls with the Mann–Whitney (Wilcoxon rank-sum) test and the two-sample Kolmogorov–Smirnov test.
2. Compare the quantiles of the two distributions at the 25th, 50th, 75th and 90th percentiles. Where is the training effect concentrated?
3. Explain why the Mann–Whitney test is not a test of equal means or medians in general.

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"], d.loc[d.train == 0, "re78"]
# TODO: stats.mannwhitneyu, stats.ks_2samp, np.quantile
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"].to_numpy(), d.loc[d.train == 0, "re78"].to_numpy()
mw = stats.mannwhitneyu(y1, y0, alternative="two-sided"); ks = stats.ks_2samp(y1, y0)
print(f"Mann-Whitney U = {mw.statistic:.0f}, p = {mw.pvalue:.4f};  P(Y1 > Y0) estimate = {mw.statistic/(len(y1)*len(y0)):.3f}")
print(f"Kolmogorov-Smirnov D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")
q = [0.25, 0.5, 0.75, 0.9]
print("quantile:       ", q)
print("treated:        ", np.quantile(y1, q).round(2))
print("control:        ", np.quantile(y0, q).round(2))
print("difference:     ", (np.quantile(y1, q) - np.quantile(y0, q)).round(2))
```

Output:

```text
Mann-Whitney U = 27402, p = 0.0109;  P(Y1 > Y0) estimate = 0.570
Kolmogorov-Smirnov D = 0.132, p = 0.0410
quantile:        [0.25, 0.5, 0.75, 0.9]
treated:         [ 0.49  4.23  9.64 14.55]
control:         [ 0.    3.14  7.29 11.36]
difference:      [0.49 1.09 2.35 3.2 ]
```

Both tests reject at 5%. The Mann–Whitney statistic estimates \(P(Y_1>Y_0)\approx0.57\): a randomly chosen trainee
out-earns a randomly chosen control 57% of the time. The quantile differences grow along the distribution, from about
\$500 at the 25th percentile to \$3,200 at the 90th, so training raised earnings throughout, most at the top. Mann–Whitney
tests \(P(Y_1>Y_0)=1/2\), which coincides with equal medians or means only under extra assumptions such as a pure location
shift.

</details>

### Exercise 10.2 — Kernel density and a bootstrap interval for a median

*Data:* `wage1` — hourly wages by gender.

1. Estimate kernel densities of `lwage` for men and women (Gaussian kernel, Silverman bandwidth) and report the modes.
2. Compute the difference in median wages (women − men) with a percentile bootstrap 95% interval (2,000 draws, resampling within each group).
3. Compare with a 95% interval for the difference in means. Which is more informative for a skewed variable, and why?

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("wage1")
# TODO: stats.gaussian_kde(..., bw_method="silverman"); bootstrap medians
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("wage1"); grid = np.linspace(-0.5, 3.5, 801)
for g, lab in ((0, "men"), (1, "women")):
    k = stats.gaussian_kde(d.loc[d.female == g, "lwage"], bw_method="silverman")
    print(f"{lab:5s}: KDE mode of log wage {grid[np.argmax(k(grid))]:.3f} (= ${np.exp(grid[np.argmax(k(grid))]):.2f}), bandwidth factor {k.factor:.3f}")
m, f = d.loc[d.female == 0, "wage"].to_numpy(), d.loc[d.female == 1, "wage"].to_numpy()
rng = np.random.default_rng(2)
bm = np.array([np.median(rng.choice(f, len(f))) - np.median(rng.choice(m, len(m))) for _ in range(2000)])
print(f"median difference (women - men) ${np.median(f) - np.median(m):.2f}, bootstrap 95% CI [{np.percentile(bm, 2.5):.2f}, {np.percentile(bm, 97.5):.2f}]")
t = stats.ttest_ind(f, m, equal_var=False); ci = t.confidence_interval()
print(f"mean difference   (women - men) ${f.mean() - m.mean():.2f}, Welch 95% CI [{ci.low:.2f}, {ci.high:.2f}]")
```

Output:

```text
men  : KDE mode of log wage 1.715 (= $5.56), bandwidth factor 0.345
women: KDE mode of log wage 1.190 (= $3.29), bandwidth factor 0.351
median difference (women - men) $-2.25, bootstrap 95% CI [-2.74, -1.74]
mean difference   (women - men) $-2.51, Welch 95% CI [-3.10, -1.93]
```

The density of log wages peaks around \$5.56 for men and \$3.29 for women. The median gap is \$2.25 with a bootstrap
interval of roughly \$1.74 to \$2.74; the mean gap is a little larger (\$2.51), because the right tail is thicker for men.
For skewed variables, the median describes the typical worker and is robust to a few very high wages; the mean matters for
totals (for example, the aggregate wage bill). Report both and say which question each answers.

</details>

## Reading
Hollander, Wolfe, and Chicken, *Nonparametric Statistical Methods*; Wasserman, *All of Nonparametric Statistics*; Efron and Tibshirani, *An Introduction to the Bootstrap*.