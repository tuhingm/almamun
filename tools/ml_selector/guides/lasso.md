title: LASSO
short: LASSO
when: Linear, few predictors matter
data: 814 students, introductory economics course (Wooldridge econmath)
tools: Python · scikit-learn · statsmodels
lead: Use this guide when the relationship is close to linear, no two predictors are strongly correlated, and only a few of many candidate predictors are likely to matter. It takes one real dataset from the question to the finished results table: choosing the penalty, reading which predictors survive, checking that the selection is stable, and testing accuracy on data the model has not seen.
---

## The question and the model

The question is a practical one for any economics department: **which student characteristics predict the final score in an introductory economics course?** The file records fourteen candidate predictors for 814 students: study habits, grades, ACT scores, prior economics and calculus, attendance attitudes and family background. Some surely matter and many probably do not. Choosing among them by hand invites a specification search; LASSO makes the choice by a rule.

LASSO (least absolute shrinkage and selection operator; Tibshirani 1996) estimates a linear model but adds a penalty on the size of the coefficients:

\[
\hat\beta = \arg\min_\beta \; \frac{1}{2n}\sum_{i=1}^{n}\big(y_i - x_i'\beta\big)^2 + \lambda \sum_{j=1}^{p}|\beta_j|
\]

The absolute-value penalty sets the coefficients of weak predictors to **exactly zero**, so LASSO selects variables while it estimates. The penalty weight \(\lambda\) controls how many survive: \(\lambda = 0\) gives OLS; a large \(\lambda\) drops everything.

| Variable | Definition | Variable | Definition |
|---|---|---|---|
| `score` | Course score, % (outcome) | `male` | 1 if male |
| `age` | Age in years | `calculus` | 1 if took calculus |
| `work` | Hours worked per week | `econhs` | 1 if took economics in high school |
| `study` | Hours studied per week | `attexc`, `attgood` | 1 if attendance rated excellent / good |
| `colgpa` | College GPA before the course | `fathcoll`, `mothcoll` | 1 if father / mother has a degree |
| `hsgpa` | High-school GPA | `acteng`, `actmth` | ACT English / maths score |

<div class="read" markdown="1">
**Prediction, not causation**

LASSO answers *which variables help to predict the score*. It does not say that studying more would raise a student's score: a selected variable may stand in for an omitted one, and a dropped variable may still have a causal effect that is small relative to the noise. If the question is causal, use double machine learning instead (see the DML guides).
</div>

## Load the data and confirm the starting point

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("econmath.csv")
y = df["score"]
X = df.drop(columns="score")
print(f"N = {len(df)} students, p = {X.shape[1]} candidate predictors")

r = X.corr().abs().where(~np.eye(X.shape[1], dtype=bool))
print(f"largest |r| between predictors: {r.max().max():.2f} ({r.stack().idxmax()[0]} and {r.stack().idxmax()[1]})")

cv = KFold(5, shuffle=True, random_state=0)
rmse = lambda m: (-cross_val_score(m, X, y, cv=cv, scoring="neg_mean_squared_error").mean()) ** 0.5
lasso = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))
forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, random_state=0)
r_l, r_f = rmse(lasso), rmse(forest)
print(f"cross-validated RMSE: LASSO {r_l:.2f}, random forest {r_f:.2f}, forest gain {1 - r_f / r_l:+.1%}")
```

The selection tool's rules are met: there are more than 100 observations, the random forest does **not** beat LASSO out of sample (it is worse), so the relationship is treated as linear, and no two predictors are correlated above 0.8. The last check, sparsity, comes in Step 4.

<div class="write" markdown="1">
**How to write it: data section**

The sample contains 814 students with complete records. The outcome is the course score (in per cent). Fourteen candidate predictors describe study effort, prior achievement, preparation in mathematics and economics, attitudes to attendance and family background. A random forest does not improve on a linear model in five-fold cross-validation, so we model the score as a linear function of the predictors and use LASSO to select among them.
</div>

## Standardise, then choose the penalty by cross-validation

The penalty treats every coefficient alike, so the predictors must be on the same scale: a coefficient on hours per week and one on a 0/1 dummy are otherwise not comparable. Standardise first (mean 0, standard deviation 1), inside the pipeline, so that each cross-validation fold is standardised with its own training data.

\(\lambda\) is chosen by five-fold cross-validation. Two choices are standard (Hastie, Tibshirani and Friedman 2009):

- \(\lambda_{\min}\): the penalty with the lowest cross-validated error, the best for **prediction**;
- \(\lambda_{1se}\): the largest penalty whose error is within one standard error of the minimum, the more conservative choice for **selection**, because it keeps only predictors whose contribution is clearly above noise.

```python
import matplotlib.pyplot as plt

Xs = StandardScaler().fit_transform(X)
cvfit = LassoCV(cv=KFold(5, shuffle=True, random_state=0), random_state=0).fit(Xs, y)
mse, se = cvfit.mse_path_.mean(1), cvfit.mse_path_.std(1) / np.sqrt(5)
i_min = mse.argmin()
lam_min = cvfit.alphas_[i_min]
lam_1se = cvfit.alphas_[mse <= mse[i_min] + se[i_min]].max()
print(f"lambda_min = {lam_min:.3f}   lambda_1se = {lam_1se:.3f}")

fig, ax = plt.subplots()
ax.errorbar(np.log(cvfit.alphas_), mse, yerr=se, fmt="o", ms=3, color="#0033a0", ecolor="#9fb3d6", capsize=0)
for lam, lab in ((lam_min, "λ min"), (lam_1se, "λ 1se")):
    ax.axvline(np.log(lam), ls="--", color="#c46000"); ax.text(np.log(lam), mse.max(), " " + lab, color="#c46000")
ax.set_xlabel("log λ"); ax.set_ylabel("cross-validated MSE"); ax.set_title("Choosing the LASSO penalty")
```

The error curve is flat near its minimum and rises steeply once the penalty starts to remove the strong predictors. The one-standard-error rule moves to the right along the flat part: a simpler model that predicts almost as well.

## Which predictors survive?

```python
def fit_at(lam):
    return Lasso(alpha=lam, max_iter=50000).fit(Xs, y).coef_

tab = pd.DataFrame({"λ min": fit_at(lam_min), "λ 1se": fit_at(lam_1se)}, index=X.columns).round(3)
tab["kept at 1se"] = np.where(tab["λ 1se"] != 0, "yes", "")
print(tab.sort_values("λ 1se", key=abs, ascending=False))
kept = list(tab.index[tab["λ 1se"] != 0])
print(f"\nkept at lambda_1se: {len(kept)} of {X.shape[1]} predictors ({len(kept) / X.shape[1]:.0%})")
```

The coefficients are in units of the score per **one standard deviation** of each predictor, so their sizes can be compared directly. At \(\lambda_{1se}\) LASSO keeps fewer than half of the predictors: the problem is sparse, which is the condition for LASSO in the selection tool (Rule A4). College GPA and the ACT maths score carry most of the signal; high-school GPA, calculus, work hours and sex add smaller amounts.

The coefficient path shows the whole story at once: as the penalty falls (to the right), predictors enter one by one.

```python
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(Xs, y, alphas=np.logspace(np.log10(lam_min / 20), np.log10(cvfit.alphas_.max()), 80))
fig, ax = plt.subplots()
for j, name in enumerate(X.columns):
    ax.plot(np.log(alphas), coefs[j], lw=1.6 if name in kept else 0.8, alpha=1 if name in kept else 0.45)
    if name in kept:
        ax.text(np.log(alphas[-1]) - 0.1, coefs[j, -1], name, fontsize=7, ha="right", va="center")
ax.axvline(np.log(lam_1se), ls="--", color="#c46000"); ax.invert_xaxis()
ax.set_xlabel("log λ (penalty falls to the right)"); ax.set_ylabel("standardised coefficient"); ax.set_title("LASSO coefficient path")
```

## How stable is the selection?

A single LASSO fit can be fragile: with a slightly different sample a borderline predictor may enter or leave. A simple check is to refit LASSO on many bootstrap samples, with the same \(\lambda_{1se}\), and count how often each predictor is kept (the idea behind *stability selection*; Meinshausen and Bühlmann 2010).

```python
rng = np.random.default_rng(0)
B, picks = 200, np.zeros(X.shape[1])
for b in range(B):
    idx = rng.integers(0, len(y), len(y))
    picks += Lasso(alpha=lam_1se, max_iter=50000).fit(Xs[idx], y.iloc[idx]).coef_ != 0
freq = pd.Series(picks / B, index=X.columns).sort_values(ascending=False)
print((freq * 100).round(0).astype(int).astype(str).add(" %").to_string())
```

Predictors kept in nearly every bootstrap sample are robust findings. Predictors kept about half the time are borderline: report them as such and do not build an argument on them.

<div class="read" markdown="1">
**Reading the frequencies**

A common rule is to call a predictor stable if it is selected in at least 80 % of the resamples (Meinshausen and Bühlmann suggest thresholds between 60 % and 90 %). Below that, the evidence that it adds predictive power is weak, even if the full-sample fit keeps it.
</div>

## Accuracy on data the model has not seen

Cross-validation chose \(\lambda\), so the cross-validated error at \(\lambda_{\min}\) is slightly optimistic. The honest test is **nested** cross-validation: an outer loop holds out each fold in turn, and the whole procedure, including the choice of \(\lambda\), is run on the other four.

```python
from sklearn.dummy import DummyRegressor

outer = KFold(5, shuffle=True, random_state=1)
models = {"mean only": DummyRegressor(),
          "OLS, all 14": make_pipeline(StandardScaler(), LinearRegression()),
          "LASSO (λ min)": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)),
          "random forest": forest}
res = {k: (-cross_val_score(m, X, y, cv=outer, scoring="neg_mean_squared_error")) ** 0.5 for k, m in models.items()}
out = pd.DataFrame({"RMSE": {k: v.mean() for k, v in res.items()}, "s.d. across folds": {k: v.std() for k, v in res.items()}}).round(2)
out["R² (out of sample)"] = (1 - out["RMSE"] ** 2 / y.var()).round(3)
print(out)
```

LASSO predicts slightly better than OLS with all fourteen predictors (an RMSE of 10.18 against 10.26 points), while setting five of them to zero even at \(\lambda_{\min}\); both explain about 40 % of the variance out of sample, far better than predicting the mean. The forest is worse. When OLS and LASSO predict equally well, the advantage of LASSO is the **interpretation**: it shows which few predictors carry the signal.

## Post-LASSO OLS, and why its p-values are not valid

A common next step is to re-estimate OLS on the predictors LASSO selected (*post-LASSO*; Belloni and Chernozhukov 2013). This removes the shrinkage bias in the kept coefficients and gives them their natural units.

```python
import statsmodels.api as sm

post = sm.OLS(y, sm.add_constant(X[kept])).fit(cov_type="HC1")
print(post.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(3))
```

<div class="warn" markdown="1">
**Do not report these p-values as tests**

The predictors were chosen *because* they fit well in this sample, so the usual standard errors are too small and the p-values too optimistic (the post-selection inference problem). Report post-LASSO coefficients as descriptions of the selected model. If you need a valid test of one variable's effect, use the double-selection or DML methods in the causal guides, which are designed for exactly that.
</div>

## Write it up

```python
final = pd.DataFrame({"LASSO coefficient (per s.d.)": tab.loc[kept, "λ 1se"],
                      "post-LASSO OLS (natural units)": post.params[kept].round(3),
                      "selected in bootstrap": (freq[kept] * 100).round(0).astype(int).astype(str) + " %"})
print(final)
print(f"\nNested CV RMSE: LASSO {out.loc['LASSO (λ min)', 'RMSE']}, OLS {out.loc['OLS, all 14', 'RMSE']}, mean-only {out.loc['mean only', 'RMSE']}")
```

<div class="write" markdown="1">
**How to write it: results**

We select predictors of the course score with LASSO, choosing the penalty by five-fold cross-validation and using the one-standard-error rule for selection. Of the fourteen candidate predictors, LASSO keeps six (Table X): college GPA and the ACT mathematics score are the strongest, followed by high-school GPA, a calculus background, hours of paid work (negative) and sex. College GPA and the ACT mathematics score are selected in every one of 200 bootstrap resamples and calculus in 97 %; high-school GPA and work hours in about 80 %, and sex in only 55 %, so we treat sex as a borderline predictor. In nested cross-validation, LASSO predicts the score with an RMSE of 10.2 points (out-of-sample R² of 0.42), slightly better than OLS with all fourteen predictors and far better than the sample mean (13.3 points); a random forest does not improve on it. Because the predictors were chosen on the same data, we report the post-LASSO coefficients as descriptive and do not interpret their p-values as tests.
</div>

<div class="read" markdown="1">
**Checklist for the paper**

- State the aim (prediction or selection) and why a penalised model is used.
- Report how λ was chosen (CV folds, λ min or λ 1se) and that predictors were standardised.
- Show the selected predictors, the coefficient path, and the selection frequencies.
- Report out-of-sample accuracy against OLS and a mean-only benchmark (nested CV).
- Do not give causal readings or post-selection p-values.
</div>
