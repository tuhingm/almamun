title: Multiple Testing and False Discovery Rate
short: Multiple testing and FDR
when: You run many tests at once and want to control the false discoveries among the ones you report
data: Differential gene expression p-values from 3,072 coral genes (Rdatasets, coralPval); one p-value per gene
tools: Python · OES engine · statsmodels
oes: fdr
lead: Use this guide when you test many hypotheses and the number of false rejections matters. It compares the unadjusted p-values, the family-wise corrections of Bonferroni, Holm and Hochberg, and the false discovery rate procedures of Benjamini and Hochberg, Benjamini and Yekutieli and Storey. The guide covers the procedures, the share of true nulls, the genes that survive, a simulation of error control and the write-up.
---

## The question and the model

**With thousands of tests, how many of the rejections are false?** At the 5% level, 5% of true nulls are rejected by chance, so 3,072 tests with no effect would give about 150 false rejections. Family-wise procedures (Bonferroni, Holm, Hochberg) keep the probability of any false rejection below \(\alpha\) and are strict. The Benjamini-Hochberg procedure (1995) keeps the expected share of false rejections among all rejections below \(\alpha\): it sorts the p-values and rejects all up to the largest \(k\) with \(p_{(k)} \le \alpha k/m\). The Benjamini-Yekutieli version is valid under any dependence, and Storey's q-value (2002) uses the estimated share \(\pi_0\) of true nulls to be less conservative.

<div class="write" markdown="1">
**How to write it: the model**

We test 3,072 genes for differential expression and adjust for multiplicity. We report the number of rejections at the 5 per cent level without adjustment, with the Bonferroni, Holm and Hochberg family-wise corrections, with the Benjamini-Hochberg and Benjamini-Yekutieli false discovery rate procedures and with Storey's q-value, estimate the share of true nulls, and check in a simulation that the false discovery proportion stays below the nominal level.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("coralpval.csv")
print(len(d), "genes")
print("share of p-values below 0.05:", round((d.p < 0.05).mean(), 3), ", below 0.001:", round((d.p < 0.001).mean(), 3))
print("histogram of p-values (deciles):", np.histogram(d.p, bins=np.linspace(0, 1, 11))[0].tolist())
```

<div class="read" markdown="1">
**How to read it**

There are 3,072 genes. 42.6% of the p-values are below 0.05 and 16% are below 0.001, far more than the 5% and 0.1% expected if no gene were affected. In the histogram by deciles the first bin holds 1,552 p-values and the others between 285 and 114: a spike near zero on top of a flat floor of about 130 per bin, which is the shape of a mixture of true effects and true nulls.
</div>

## Adjust for multiplicity

```python
spec = dict(p="p", label="id", alpha=0.05)
r = oes.run("fdr", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Number of rejections")
```

<div class="read" markdown="1">
**How to read it**

Without adjustment 1,310 genes are rejected at the 5% level. The family-wise corrections (Bonferroni, Holm and Hochberg) keep 60, because they guard against even one false rejection among 3,072. The Benjamini-Hochberg procedure keeps 1,015, the Benjamini-Yekutieli version, valid under any dependence, 485, and Storey's q-value 1,315. The differences are the price of the guarantee: Bonferroni protects against any false rejection and loses 95% of the discoveries; false discovery rate control accepts a known share of false ones.
</div>

## How many are false?

```python
oes.show(r, "Null proportion")
oes.show(r, "The smallest p-values")
```

<div class="read" markdown="1">
**How to read it**

The estimated share of true nulls is pi0 = 0.42, so about 1,290 genes are unaffected and 1,780 are affected. If all genes were null, about 154 false rejections would be expected at the 5% level without correction. For the Benjamini-Hochberg rejections, the expected number of false discoveries is at most 5% of 1,015, or about 51. The smallest p-values are of the order of 1e-6 and their adjusted values are still tiny: the top genes are not in doubt under any procedure.
</div>

## Robustness: the level

```python
oes.show(r, "Rejections at several levels")
for a in (0.01, 0.05, 0.1):
    e = oes.run("fdr", d.to_csv(index=False), dict(spec, alpha=a))
    q = {row[0]: row[1] for row in e["tables"][0]["rows"]}
    print(f"alpha {a}: upper bound on false discoveries among the BH rejections {a * q['Benjamini-Hochberg']:.0f} of {q['Benjamini-Hochberg']}; unadjusted, if all were null: {a * len(d):.0f} of {q['Unadjusted p-value']}")
```

<div class="read" markdown="1">
**How to read it**

At the 1% level the numbers are 900 (unadjusted), 37 (Bonferroni), 580 (Benjamini-Hochberg), 98 (Benjamini-Yekutieli) and 786 (Storey); at 10% they are 1,552, 93, 1,235, 620 and 1,637. The upper bound on the number of false discoveries among the Benjamini-Hochberg rejections is 6 of 580 at 1%, 51 of 1,015 at 5% and 124 of 1,235 at 10%. The ranking of the procedures is the same at every level: Bonferroni and Holm are the strictest and Benjamini-Hochberg and Storey the most generous.
</div>

## Diagnostics: error control in a simulation

```python
from scipy import stats
rng = np.random.default_rng(6)
m, R = 500, 100
print("500 tests, 20% false nulls (effect 3 s.d.), nominal level 5%; averages over 100 samples")
acc = {}
for _ in range(R):
    alt = rng.random(m) < 0.2
    z = np.where(alt, rng.normal(3, 1, m), rng.normal(0, 1, m)); p = 2 * stats.norm.sf(np.abs(z))
    e = oes.run("fdr", pd.DataFrame(dict(p=p, i=np.arange(m))).to_csv(index=False), dict(p="p", alpha=0.05))
    q = {row[0]: row[1] for row in e["tables"][0]["rows"]}
    for name, cut in (("Unadjusted p-value", p <= 0.05),):
        acc.setdefault(name, []).append((q[name], (cut & ~alt).sum() / max(cut.sum(), 1), (cut & alt).sum() / alt.sum()))
    ps = np.sort(p); idx = np.arange(1, m + 1)
    k = np.max(np.where(ps <= 0.05 * idx / m)[0] + 1) if np.any(ps <= 0.05 * idx / m) else 0
    cb = p <= (ps[k - 1] if k else -1); acc.setdefault("Benjamini-Hochberg", []).append((q["Benjamini-Hochberg"], (cb & ~alt).sum() / max(cb.sum(), 1), (cb & alt).sum() / alt.sum()))
    cbo = p <= 0.05 / m; acc.setdefault("Bonferroni", []).append((q["Bonferroni"], (cbo & ~alt).sum() / max(cbo.sum(), 1), (cbo & alt).sum() / alt.sum()))
print("procedure             rejections   false discovery proportion   power")
for k, v in acc.items():
    v = np.mean(v, axis=0); print(f"{k:22s} {v[0]:9.1f} {v[1]:20.3f} {v[2]:15.3f}")
```

<div class="read" markdown="1">
**How to read it**

With 500 tests, 20% of them false with an effect of 3 standard deviations, and a nominal level of 5%, the unadjusted test rejects 106 hypotheses on average, 85% of the false ones, but 19% of its rejections are false. Benjamini-Hochberg rejects 63, finds 60% of the false ones and has a false discovery proportion of 3.5%, below the 5% level (the theory gives 5% times the share of true nulls, 4%). Bonferroni rejects 19, finds 19% of the false ones and has a false discovery proportion of 0.2%. Each procedure trades power for error control and the right choice depends on the cost of a false discovery.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We test 3,072 genes for differential expression. 1,310 are significant at the 5% level without adjustment, but this includes false discoveries: if all genes were null we would expect about 154. Bonferroni, Holm and Hochberg leave 60 genes, Benjamini-Yekutieli 485 and Benjamini-Hochberg 1,015, which keeps the expected share of false discoveries among them below 5% (about 51 genes at most); Storey's q-value gives 1,315. The estimated share of true nulls is 0.42. We report the Benjamini-Hochberg list as the main result and the Bonferroni list as the most conservative one. In a simulation, Benjamini-Hochberg has a false discovery proportion of 3.5% against 19% without adjustment and keeps 60% of the true effects against 19% for Bonferroni.
</div>

**References.** Benjamini and Hochberg (1995), *Journal of the Royal Statistical Society B*; Benjamini and Yekutieli (2001), *Annals of Statistics*; Storey (2002), *Journal of the Royal Statistical Society B*; Holm (1979), *Scandinavian Journal of Statistics*; Hochberg (1988), *Biometrika*.
