# Lecture 13 — Multivariate Analysis: PCA, Covariance, and Dimension Reduction

## Learning objectives
Analyze covariance matrices; derive principal components; interpret eigenvalues and loadings; distinguish PCA from factor analysis; understand scaling and validation.

## 1. Covariance geometry
For centered data matrix \(X\in\mathbb R^{n\times p}\), sample covariance is \(S=X'X/(n-1)\). It is symmetric positive semidefinite. Eigenvectors identify orthogonal directions; eigenvalues measure variance along those directions. Variables on different scales can dominate covariance-based analysis.

## 2. Principal components
The first component solves
\[
\max_{a'a=1}a'Sa.
\]
A Lagrange multiplier yields \(Sa=\lambda a\); the maximizing direction is the eigenvector associated with the largest eigenvalue. Subsequent components maximize variance subject to orthogonality. The proportion of variance explained by component \(j\) is \(\lambda_j/\sum_k\lambda_k\).

## 3. Scores and loadings
For eigenvector matrix \(A\), component scores are \(Z=XA\). Loadings describe contributions of original variables, but scaling conventions differ. Eigenvector signs are arbitrary: multiplying an eigenvector by -1 leaves the component direction unchanged. Component labels require substantive interpretation.

## 4. Scaling
PCA on the covariance matrix preserves original units. PCA on the correlation matrix is equivalent to standardizing variables first. Neither is universally preferable; choose based on whether scale differences are substantively meaningful.

## 5. PCA and factor analysis
PCA decomposes total observed variance into orthogonal linear combinations. Common factor analysis models shared covariance through latent factors and variable-specific errors. PCA is a descriptive dimension-reduction transformation, not automatically a measurement or causal model.

## 6. Dimension selection and validation
Use scree plots, parallel analysis, cumulative explained variance, interpretability, and out-of-sample reconstruction or prediction. When PCA is used before prediction, estimate scaling and components within each training fold to prevent information leakage.

## Worked example
For \(S=\begin{pmatrix}4&2\\2&3\end{pmatrix}\), solve \(|S-\lambda I|=0\), then normalize the eigenvectors. The leading eigenvector gives the direction of maximum sample variance.

## Problem set
1. Prove covariance eigenvalues are nonnegative.
2. Derive the first-component solution using a Lagrange multiplier.
3. Calculate explained-variance proportions.
4. Explain standardization before PCA.
5. Distinguish PCA and common factor analysis.

## Computational lab
Compare covariance- and correlation-based PCA. Report eigenvalues, loadings, scores, scree plot, and reconstruction error; evaluate component stability by bootstrap.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 13.1 — Principal components of the U.S. yield curve

*Data:* `intqrt` — U.S. quarterly Treasury bill yields at 3, 6 and 12 months (`r3`, `r6`, `r12`).

1. Compute the correlation matrix of the three yields.
2. Run PCA on the standardized yields. Report the eigenvalues, the share of variance explained, and the loadings.
3. Interpret the first two components (the "level" and "slope" factors of the term structure).
4. How well does the first component alone reconstruct the 12-month yield?

```python
import numpy as np, wooldridge as wd
d = wd.data("intqrt")[["r3", "r6", "r12"]].dropna()
Z = (d - d.mean()) / d.std()
# TODO: eigenvalues/eigenvectors of the correlation matrix (np.linalg.eigh), sorted descending
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd
d = wd.data("intqrt")[["r3", "r6", "r12"]].dropna()
print(d.corr().round(4).to_string())
Z = ((d - d.mean()) / d.std()).to_numpy()
val, vec = np.linalg.eigh(np.corrcoef(Z, rowvar=False)); o = val.argsort()[::-1]; val, vec = val[o], vec[:, o]
vec = vec * np.sign(vec[0])                               # sign convention: first loading positive
print("\neigenvalues:", val.round(4), " share explained:", (val / val.sum()).round(4))
print(pd.DataFrame(vec, index=d.columns, columns=["PC1", "PC2", "PC3"]).round(3).to_string())
pc1 = Z @ vec[:, 0]
r12_hat = np.outer(pc1, vec[:, 0])[:, 2] * d["r12"].std() + d["r12"].mean()
print(f"\nRMSE of 12-month yield reconstructed from PC1 alone: {np.sqrt(np.mean((r12_hat - d['r12'])**2)):.3f} percentage points (yield sd {d['r12'].std():.2f})")
```

Output:

```text
r3      r6     r12
r3   1.0000  0.9966  0.9893
r6   0.9966  1.0000  0.9970
r12  0.9893  0.9970  1.0000

eigenvalues: [2.9886 0.0107 0.0007]  share explained: [0.9962 0.0036 0.0002]
       PC1    PC2    PC3
r3   0.577  0.716  0.393
r6   0.578 -0.018 -0.816
r12  0.577 -0.698  0.424

RMSE of 12-month yield reconstructed from PC1 alone: 0.224 percentage points (yield sd 3.07)
```

The three yields are almost perfectly correlated, and the first component explains 99.6% of their variance. Its loadings
are equal across maturities: it is the **level** factor. The second component has opposite signs at short and long
maturities: the **slope** factor. The third, with a hump-shaped loading, is curvature. Reconstructing the 12-month yield from
the level factor alone leaves an error of only 0.22 percentage points against a standard deviation of 3.07. This is
Litterman and Scheinkman's (1991) finding in miniature.

</details>

### Exercise 13.2 — PCA and a crime index across U.S. states

*Data:* `statecrime` (statsmodels) — 51 U.S. states and DC: violent crime and murder rates, % high-school graduates, % below poverty, % single-parent households, % white, % urban.

1. Run PCA on the seven standardized variables. How many components have eigenvalue above one (Kaiser rule)?
2. Interpret PC1 using the loadings.
3. Regress the violent crime rate on the scores of the first two components of the five **non-crime** variables only (principal-components regression). Compare its \(R^2\) with the regression on all five covariates, and check how collinear the covariates actually are. What does the comparison teach about PCR?

```python
import numpy as np, statsmodels.api as sm
d = sm.datasets.statecrime.load_pandas().data
# TODO: standardize, eigen-decompose the correlation matrix; principal-components regression
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
d = sm.datasets.statecrime.load_pandas().data
def pca(X):
    Z = (X - X.mean()) / X.std(); val, vec = np.linalg.eigh(Z.corr()); o = val.argsort()[::-1]
    vec = vec[:, o] * np.sign(vec[:, o].sum(0))           # sign convention for readability
    return val[o], pd.DataFrame(vec, index=X.columns), Z.to_numpy() @ vec
val, load, _ = pca(d)
print("eigenvalues:", val.round(3), " -> components with eigenvalue > 1:", int((val > 1).sum()))
print("PC1 loadings:", load[0].round(3).to_dict())
X = d[["hs_grad", "poverty", "single", "white", "urban"]]
v2, l2, S = pca(X)
pcr = sm.OLS(d["violent"], sm.add_constant(S[:, :2])).fit()
full = sm.OLS(d["violent"], sm.add_constant(X)).fit()
print(f"\nPC regression (2 components of 5 covariates): R2 = {pcr.rsquared:.3f};  full regression on 5 covariates: R2 = {full.rsquared:.3f}")
print(f"condition number of the 5 standardized covariates: {np.sqrt(v2.max()/v2.min()):.1f}")
```

Output:

```text
eigenvalues: [4.062 1.456 0.666 0.472 0.156 0.122 0.065]  -> components with eigenvalue > 1: 2
PC1 loadings: {'violent': 0.431, 'murder': 0.451, 'hs_grad': -0.334, 'poverty': 0.297, 'single': 0.474, 'white': -0.37, 'urban': 0.223}

PC regression (2 components of 5 covariates): R2 = 0.546;  full regression on 5 covariates: R2 = 0.681
condition number of the 5 standardized covariates: 4.5
```

Two components have eigenvalues above one. PC1 loads positively on violence, murder, single-parent households and poverty
and negatively on high-school graduation and the white share: a social-disadvantage index. Two components of the five covariates explain about 55% of the variation in violent crime,
against 68% for the regression on all five. The covariates are only mildly collinear (condition number 4.5), so reducing
dimension buys little stability here and costs some fit. More fundamentally, the leading components summarize the
**covariates**, not the **outcome**: PCR can discard a low-variance direction that matters for prediction. Partial least
squares and penalized regression (ridge, lasso) use the outcome when choosing directions. PCR earns its keep when the
number of covariates is large relative to \(n\) or the covariates are nearly collinear, as with the yields in the previous
exercise.

</details>

## Reading
Jolliffe and Cadima, “Principal Component Analysis: A Review and Recent Developments”; Johnson and Wichern, *Applied Multivariate Statistical Analysis*; Anderson, *An Introduction to Multivariate Statistical Analysis*.