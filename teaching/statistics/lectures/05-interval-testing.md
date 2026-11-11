# Lecture 05 — Confidence Intervals and Hypothesis Testing

## Learning objectives
Construct confidence intervals; formulate tests; derive size and power; interpret p-values; understand likelihood-ratio, Wald, and score tests; distinguish statistical and practical significance.

## 1. Confidence intervals
A \((1-\alpha)\) confidence procedure satisfies
\[
P_\theta\{L(X)\le\theta\le U(X)\}\ge1-\alpha.
\]
The parameter is fixed and the interval is random before sampling. Coverage is a repeated-sampling property, not a posterior probability assigned to a realized interval.

## 2. Testing framework
Specify null \(H_0\), alternative \(H_1\), statistic, rejection region, and level \(\alpha\). Type I error is rejection under the null; Type II error is non-rejection under a specified alternative; power is \(1-\beta\). Power depends on the alternative, sample size, variance, design, and test.

## 3. p-values
A p-value is the probability, under the null model, of a test statistic at least as extreme as the observed statistic under the selected extremeness ordering. It is not the probability that the null is true, nor a measure of effect magnitude.

## 4. Exact mean inference
For i.i.d. normal observations with unknown variance,
\[
T=\frac{\bar X-\mu_0}{S/\sqrt n}\sim t_{n-1}
\]
under the null. The two-sided interval is \(\bar X\pm t_{1-\alpha/2,n-1}S/\sqrt n\).

## 5. Likelihood-based tests
The likelihood-ratio statistic is \(2[\ell(\widehat\theta)-\ell(\widetilde\theta)]\), where the second estimate is restricted under the null. Under regularity and \(q\) restrictions it converges to \(\chi_q^2\). Wald tests use unrestricted estimates and covariance; score tests evaluate the score under the restricted estimate. Their finite-sample behavior can differ.

## 6. Multiple testing
Bonferroni and Holm procedures control family-wise error; false-discovery-rate methods target the expected proportion of false rejections among rejections under their conditions. Report estimates, uncertainty intervals, and substantive scale—not p-values alone.

## Worked example
If \(\bar X=12\), \(S=4\), and \(n=25\), the standard error is 0.8. For \(H_0:\mu=10\), the t statistic is 2.5 with 24 degrees of freedom. Interpret the test together with the estimated difference of two units and its interval.

## Problem set
1. Construct a t interval from a sample summary.
2. Explain 95% repeated-sampling coverage.
3. Distinguish Type I error, Type II error, and power.
4. Compare LR, Wald, and score tests.
5. Explain why a p-value is not the posterior probability of the null.

## Computational lab
Simulate null and alternative samples; estimate empirical size and power; compare LR, Wald, and score tests in finite samples; demonstrate multiplicity under repeated testing.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 5.1 — Did job training raise earnings? Welch t-test, permutation test and a confidence interval

*Data:* `jtrain2` — NSW randomized experiment, 445 men; `re78` = 1978 earnings (thousands of dollars).

1. Compute the difference in mean earnings between trainees and controls, with Welch and pooled-variance \(t\)-tests.
2. Run a permutation test of the sharp null of no effect (10,000 permutations, seed 0).
3. Report a 95% Welch confidence interval and interpret it in dollars.
4. Earnings are highly skewed with many zeros. Which of the three procedures relies least on distributional assumptions?

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y1, y0 = d.loc[d.train == 1, "re78"], d.loc[d.train == 0, "re78"]
# TODO: stats.ttest_ind(..., equal_var=False); permutation loop
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
d = wd.data("jtrain2"); y = d["re78"].to_numpy(); t = d["train"].to_numpy().astype(bool)
diff = y[t].mean() - y[~t].mean()
w, p = stats.ttest_ind(y[t], y[~t], equal_var=False), stats.ttest_ind(y[t], y[~t], equal_var=True)
print(f"difference in means = {diff:.3f} thousand dollars")
print(f"Welch t = {w.statistic:.3f} (p = {w.pvalue:.4f});  pooled t = {p.statistic:.3f} (p = {p.pvalue:.4f})")
rng = np.random.default_rng(0); perm = np.empty(10000)
for b in range(10000):
    s = rng.permutation(t); perm[b] = y[s].mean() - y[~s].mean()
print(f"permutation p-value (two-sided) = {np.mean(np.abs(perm) >= abs(diff)):.4f}")
v1, v0, n1, n0 = y[t].var(ddof=1), y[~t].var(ddof=1), t.sum(), (~t).sum()
se = np.sqrt(v1/n1 + v0/n0); df = se**4 / ((v1/n1)**2/(n1-1) + (v0/n0)**2/(n0-1))
q = stats.t.ppf(0.975, df)
print(f"95% Welch CI: [{diff - q*se:.3f}, {diff + q*se:.3f}]  (df = {df:.1f})")
print(f"share with zero earnings: treated {np.mean(y[t] == 0):.3f}, control {np.mean(y[~t] == 0):.3f}")
```

Output:

```text
difference in means = 1.794 thousand dollars
Welch t = 2.674 (p = 0.0079);  pooled t = 2.835 (p = 0.0048)
permutation p-value (two-sided) = 0.0045
95% Welch CI: [0.474, 3.115]  (df = 307.1)
share with zero earnings: treated 0.243, control 0.354
```

Trainees earned about \$1,800 more in 1978. All three procedures agree on significance (p < 0.01). The permutation test
uses only the random assignment — no normality, no equal variances — so it is the most assumption-light, and it is the
natural test for a randomized experiment. The Welch interval runs from about \$470 to \$3,100: the effect is clearly
positive but imprecisely measured. A third of controls and a quarter of trainees had zero earnings, so the distribution is
far from normal; the \(t\)-tests are rescued by the sample size of 445.

</details>

### Exercise 5.2 — Wald, likelihood-ratio and score tests for a proportion

*Data:* `pntsprd` — 553 college basketball games; `favwin` = 1 if the betting favorite won.

Test whether favorites win 75% of games, \(H_0:p=0.75\).

1. Compute the Wald, likelihood-ratio and score statistics and their p-values.
2. Compute the Wald (normal) and Wilson (score-inverted) 95% intervals for \(p\).
3. Why do the three tests give similar answers here, and when would they diverge?

```python
import numpy as np, wooldridge as wd
from scipy import stats
x = wd.data("pntsprd")["favwin"].to_numpy(); n, k = len(x), x.sum(); p0 = 0.75
# TODO: Wald = (ph-p0)^2 / (ph(1-ph)/n); score uses p0 in the variance; LR = 2(logL(ph) - logL(p0))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
x = wd.data("pntsprd")["favwin"].to_numpy(); n, k = len(x), int(x.sum()); p0 = 0.75; ph = k / n
ll = lambda p: k * np.log(p) + (n - k) * np.log(1 - p)
W = (ph - p0)**2 / (ph * (1 - ph) / n); S = (ph - p0)**2 / (p0 * (1 - p0) / n); LR = 2 * (ll(ph) - ll(p0))
print(f"favorites won {k} of {n} games: p-hat = {ph:.4f}")
for name, s in (("Wald", W), ("LR", LR), ("score", S)):
    print(f"  {name:5s}: chi2(1) = {s:.3f}, p = {stats.chi2.sf(s, 1):.4f}")
z = 1.96; se = np.sqrt(ph * (1 - ph) / n)
c = (ph + z**2/(2*n)) / (1 + z**2/n); h = z * np.sqrt(ph*(1-ph)/n + z**2/(4*n**2)) / (1 + z**2/n)
print(f"Wald 95% CI [{ph - z*se:.4f}, {ph + z*se:.4f}]   Wilson 95% CI [{c - h:.4f}, {c + h:.4f}]")
```

Output:

```text
favorites won 422 of 553 games: p-hat = 0.7631
  Wald : chi2(1) = 0.526, p = 0.4684
  LR   : chi2(1) = 0.513, p = 0.4738
  score: chi2(1) = 0.507, p = 0.4765
Wald 95% CI [0.7277, 0.7985]   Wilson 95% CI [0.7259, 0.7967]
```

Favorites won 76.3% of games. None of the three tests rejects \(p=0.75\), and they give nearly identical statistics,
because \(n\) is large and \(\widehat p\) is close to \(p_0\): the log-likelihood is close to quadratic over that range.
They diverge when \(n\) is small or \(p\) is near 0 or 1. The Wilson interval is then much better behaved than the
Wald interval, which can extend outside \([0,1]\) and undercovers.

</details>

## Reading
Casella and Berger, *Statistical Inference*; Lehmann and Romano, *Testing Statistical Hypotheses*; Wasserstein and Lazar, “The ASA Statement on p-Values.”