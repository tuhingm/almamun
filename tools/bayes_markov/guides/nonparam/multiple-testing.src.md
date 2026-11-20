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
@@LOAD@@
```

@@READ_LOAD@@

## Adjust for multiplicity

```python
@@EST@@
```

@@READ_EST@@

## How many are false?

```python
@@FALSE@@
```

@@READ_FALSE@@

## Robustness: the level

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: error control in a simulation

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Benjamini and Hochberg (1995), *Journal of the Royal Statistical Society B*; Benjamini and Yekutieli (2001), *Annals of Statistics*; Storey (2002), *Journal of the Royal Statistical Society B*; Holm (1979), *Scandinavian Journal of Statistics*; Hochberg (1988), *Biometrika*.
