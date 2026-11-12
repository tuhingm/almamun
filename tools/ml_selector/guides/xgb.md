title: Gradient Boosting (XGBoost)
short: XGBoost
when: Nonlinear, at least 1,000 observations
data: 4,596 U.S. air routes × years, 1997–2000 (Wooldridge airfare)
tools: Python · XGBoost · scikit-learn
lead: Use this guide when a random forest predicts clearly better than LASSO, so the relationship is nonlinear, and there are enough observations (1,000 or more) to tune a boosted model without overfitting. It takes one real dataset through validation that respects its structure, early stopping, tuning, a fair comparison with the forest, and interpretation with SHAP values and partial dependence.
---

## The question and the model

**How are airline fares related to route distance, traffic and market concentration?** The file holds the average fare on 1,149 U.S. domestic routes in each of four years, 1997–2000: 4,596 route-years. Fares are not linear in distance (short flights carry large fixed costs per mile) and the effect of market power may depend on the route. A boosted model can learn both.

**Gradient boosting** (Friedman 2001) builds trees one after another. Each new tree is fitted to the errors that the current model still makes, and is added with a small weight \(\eta\) (the learning rate):

\[
\hat f_m(x) = \hat f_{m-1}(x) + \eta\, T_m(x), \qquad T_m \text{ fitted to the residuals of } \hat f_{m-1}.
\]

Where a random forest averages many independent deep trees, boosting adds many small dependent ones, each correcting the last. **XGBoost** (Chen and Guestrin 2016) adds penalties on tree size and leaf weights, and is the standard implementation in applied work.

| Variable | Definition |
|---|---|
| `lfare` | Log average one-way fare, dollars (outcome) |
| `dist` | Route distance, miles |
| `passen` | Average passengers per day |
| `concen` | Market share of the largest carrier on the route |
| `year` | 1997 to 2000 |
| `id` | Route identifier (used only to build the validation folds) |

## Load the data and confirm the starting point

Each route appears four times. If the same route is in both the training and the validation fold, the model can memorise it and look better than it is. So every fold below keeps **all years of a route together** (`GroupKFold`).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("airfare_full.csv")
print(f"{len(df)} route-years, {df.id.nunique()} routes, years {df.year.min()}-{df.year.max()}")
print(f"correlation of concen and bmktshr: {df.concen.corr(df.bmktshr):.3f}  -> the same variable; keep concen")
y = df["lfare"]
X = df[["dist", "passen", "concen", "year"]].astype(float)
g = df["id"]

gkf = GroupKFold(n_splits=5)
rmse = lambda m: (-cross_val_score(m, X, y, cv=gkf, groups=g, scoring="neg_mean_squared_error").mean()) ** .5
r_l = rmse(make_pipeline(StandardScaler(), LassoCV(cv=5)))
r_f = rmse(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=.5, random_state=0, n_jobs=-1))
print(f"grouped-CV RMSE: LASSO {r_l:.4f}, random forest {r_f:.4f}, forest gain {1 - r_f / r_l:.1%}")
```

The forest is far more than 5 % better than LASSO (Rule A3), and with over 4,000 observations there is enough data to tune a boosted model (Rule A5).

<div class="write" markdown="1">
**How to write it: data section**

The data cover 1,149 U.S. domestic air routes observed annually from 1997 to 2000 (4,596 route-years). The outcome is the log average fare; the predictors are route distance, average daily passengers, the largest carrier's market share and the year. Because each route appears in several years, all cross-validation folds keep the observations of a route together. In this grouped cross-validation a random forest predicts log fares with an error about 16 % lower than LASSO, so we model fares with gradient-boosted trees.
</div>

## The parameters that matter, and early stopping

Boosting has more tuning parameters than a forest, and unlike a forest it **can** overfit if it runs too long. The ones that matter most:

- **learning rate \(\eta\)**: smaller is better but needs more trees; 0.05 is a common choice;
- **number of trees**: chosen by **early stopping**, i.e. stop adding trees when the error on a validation set has not improved for a while;
- **`max_depth`**: the depth of each tree, which sets how many variables can interact;
- **`min_child_weight`**, **`subsample`**, **`colsample_bytree`**: regularisation, as in a forest.

```python
import xgboost as xgb
import matplotlib.pyplot as plt

tr, va = next(GroupKFold(n_splits=5).split(X, y, g))
m = xgb.XGBRegressor(n_estimators=3000, learning_rate=.05, max_depth=4, subsample=.8, colsample_bytree=1.0,
                     min_child_weight=5, early_stopping_rounds=100, eval_metric="rmse", random_state=0)
m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[tr], y.iloc[tr]), (X.iloc[va], y.iloc[va])], verbose=False)
h = m.evals_result()
print(f"best number of trees: {m.best_iteration + 1}; validation RMSE {h['validation_1']['rmse'][m.best_iteration]:.4f}")
fig, ax = plt.subplots()
ax.plot(h["validation_0"]["rmse"], label="training routes"); ax.plot(h["validation_1"]["rmse"], label="validation routes")
ax.axvline(m.best_iteration, ls="--", color="#c46000"); ax.set_ylim(0, .5)
ax.set_xlabel("number of trees"); ax.set_ylabel("RMSE"); ax.set_title("Early stopping"); ax.legend()
```

The training error keeps falling as trees are added; the validation error falls, flattens and starts to rise. Early stopping picks the number of trees at the bottom of the validation curve.

## Tune depth and regularisation

Search a small grid with the grouped folds. The number of trees is fixed near the early-stopping value; the learning rate stays at 0.05.

```python
from sklearn.model_selection import GridSearchCV

base = xgb.XGBRegressor(n_estimators=m.best_iteration + 1, learning_rate=.05, subsample=.8, random_state=0, n_jobs=-1)
grid = GridSearchCV(base, {"max_depth": [2, 3, 4, 6], "min_child_weight": [1, 5, 20]}, cv=gkf,
                    scoring="neg_mean_squared_error").fit(X, y, groups=g)
res = pd.DataFrame(grid.cv_results_["params"]).assign(RMSE=np.sqrt(-grid.cv_results_["mean_test_score"]))
print(res.pivot(index="max_depth", columns="min_child_weight", values="RMSE").round(4))
best = grid.best_params_; print("best:", best)
```

## A fair comparison with the alternatives

Nested grouped cross-validation: the outer folds hold out whole routes, and the boosted model is tuned inside each training set.

```python
from sklearn.dummy import DummyRegressor

outer = GroupKFold(n_splits=5)
inner_xgb = GridSearchCV(xgb.XGBRegressor(n_estimators=m.best_iteration + 1, learning_rate=.05, subsample=.8, random_state=0, n_jobs=-1),
                         {"max_depth": [3, 4, 6], "min_child_weight": [5, 20]}, cv=GroupKFold(4), scoring="neg_mean_squared_error")
models = {"mean only": DummyRegressor(), "OLS": make_pipeline(StandardScaler(), LinearRegression()),
          "LASSO": make_pipeline(StandardScaler(), LassoCV(cv=5)),
          "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=.5, random_state=0, n_jobs=-1),
          "XGBoost (tuned)": inner_xgb}
out = {}
for k, mm in models.items():
    kw = {"groups": g} if k == "XGBoost (tuned)" else {}
    s = []
    for trn, tst in outer.split(X, y, g):
        fit = mm.fit(X.iloc[trn], y.iloc[trn], **({"groups": g.iloc[trn]} if kw else {}))
        s.append(np.sqrt(np.mean((y.iloc[tst] - fit.predict(X.iloc[tst])) ** 2)))
    out[k] = {"RMSE": np.mean(s), "s.d. across folds": np.std(s), "R² (out of sample)": 1 - np.mean(s) ** 2 / y.var()}
out = pd.DataFrame(out).T.round(4)
print(out)
```

Both tree ensembles beat the linear models by a wide margin. Whether boosting also beats the forest, and by how much, is the empirical question this table answers; report it whichever way it goes.

## Open the black box: SHAP values

**SHAP values** (Lundberg and Lee 2017) split each prediction into additive contributions from each predictor, measured from the average prediction. They are consistent with how the model actually computes, and XGBoost calculates them exactly for trees (`pred_contribs=True`).

```python
fin = xgb.XGBRegressor(n_estimators=m.best_iteration + 1, learning_rate=.05, subsample=.8, random_state=0, **best).fit(X, y)
shap = fin.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]     # last column = bias
imp = pd.Series(np.abs(shap).mean(0), index=X.columns).sort_values(ascending=False)
print("mean |SHAP| (log points):"); print(imp.round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
imp[::-1].plot.barh(ax=axes[0], color="#0033a0"); axes[0].set_xlabel("mean |SHAP value|"); axes[0].set_title("Importance")
axes[1].scatter(X["dist"], shap[:, 0], s=4, c=X["concen"], cmap="viridis", alpha=.6)
axes[1].set_xlabel("distance (miles)"); axes[1].set_ylabel("SHAP value of distance"); axes[1].set_title("Effect of distance (colour: concentration)")
fig.tight_layout()
```

Distance is by far the most important predictor, and its effect is strongly concave: fares rise quickly over the first few hundred miles and much more slowly after that. Market concentration comes next.

## Partial dependence

```python
from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
PartialDependenceDisplay.from_estimator(fin, X, ["dist", "concen", "passen"], ax=axes)
for a in axes: a.set_ylabel("predicted log fare")
fig.suptitle("Partial dependence of the log fare", fontsize=10, fontweight="bold"); fig.tight_layout()
```

Holding the other predictors at their observed values, the predicted fare rises with the largest carrier's market share, and falls with traffic on the route (busier routes are cheaper). These are predictive associations, not causal effects of market power: concentrated routes may differ in ways the model does not see.

## Write it up

```python
print(out)
print("\nfinal model:", {**best, "learning_rate": .05, "n_estimators": m.best_iteration + 1, "subsample": .8})
print("importance (mean |SHAP|):", ", ".join(f"{k} {v:.3f}" for k, v in imp.items()))
```

<div class="write" markdown="1">
**How to write it: results**

We predict log fares with gradient-boosted trees (XGBoost), using a learning rate of 0.05, choosing the number of trees by early stopping and the tree depth and minimum leaf weight by grouped cross-validation, so that no route ever appears in both a training and a validation fold. In nested grouped cross-validation, the boosted model and a random forest both predict far better than OLS and LASSO (Table X). SHAP values show that route distance dominates the predictions and enters strongly concavely, followed by market concentration; partial-dependence plots show higher fares on more concentrated routes and lower fares on busier ones. We interpret these patterns as predictive, not causal.
</div>

<div class="read" markdown="1">
**Checklist for the paper**

- Build validation folds that respect the data's structure (groups, time).
- Report the learning rate, early stopping and the tuned depth and regularisation.
- Compare with a random forest and with linear models in nested cross-validation.
- Interpret with SHAP values and partial dependence, and say that the patterns are predictive.
</div>
