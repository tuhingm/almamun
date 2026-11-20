title: Interpreting a Machine-Learning Model: Importance, Partial Dependence and Interactions
short: Interpretation
when: A forest or boosting model predicts well and you must say which variables matter, how they matter and whether they interact
data: 663 U.S. men, wages and personal characteristics (Wooldridge wage2, complete cases)
oes: interpret
tools: Python · OES engine · scikit-learn · statsmodels
lead: Use this guide after a flexible model has been fitted and a reader asks what it learned. It computes permutation importance on a test set, partial dependence profiles for the leading predictors, and Friedman's H statistic for their interactions, and compares them with the coefficients of a linear model. It also shows what these summaries cannot do: share importance among correlated variables, and say anything about causation.
---

## The question and the tools

**What did the model learn?** A forest or a boosted ensemble has no coefficients. Three summaries describe the fitted function \(\hat f(x)\):

- **Permutation importance:** shuffle one predictor in the test set and record how much the test \(R^2\) falls. A variable the model relies on costs a lot when shuffled. Because it is measured on data the model has not seen, it does not reward overfitting.
- **Partial dependence:** \(\mathrm{PD}_j(v) = \frac1n\sum_i \hat f(x_{ij}=v,\, x_{i,-j})\), the average prediction when predictor \(j\) is set to \(v\) and the others keep their observed values. It shows the shape of the effect (linear, saturating, threshold).
- **Interaction strength:** Friedman's \(H^2_{jk}\) is the share of the variance of the joint partial dependence of \(j\) and \(k\) that is not explained by the two separate partial dependences. It is 0 for an additive model.

<div class="read" markdown="1">
**Three cautions before you report importance**

1. **Correlated predictors split their importance.** If two variables carry the same information, shuffling one leaves the other to compensate, so each looks less important than the pair is.
2. **Importance is not an effect.** A variable can be important because it proxies for something omitted.
3. **A weak model has little to interpret.** If the test \(R^2\) is low, the importance of a variable is small in absolute terms, so report it next to the test fit.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We ask which characteristics a random forest uses to predict the log wage of 663 U.S. men, and how. We train the forest (200 trees, minimum leaf size 5) on 75% of the sample and compute the permutation importance of each of 15 predictors on the other 25%, as the fall in test R-squared averaged over 30 random shuffles. We plot the partial dependence of the log wage on the three most important predictors, measure pairwise interaction strength with Friedman's H statistic (Friedman and Popescu 2008) and compare the ranking with standardised OLS coefficients (Molnar 2022).
</div>

## Load the data and look at the correlated predictors

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("wage2.csv").read()
df = pd.read_csv("wage2.csv")
xs = [c for c in df.columns if c != "lwage"]
print(len(df), "men;", len(xs), "predictors")
print(df[["lwage", "IQ", "KWW", "educ", "exper", "tenure", "age"]].corr().round(2))
```

<div class="read" markdown="1">
**How to read it**

The predictors are correlated: education correlates 0.54 with the IQ score and 0.41 with the knowledge-of-the-world-of-work score (KWW), and the two test scores correlate 0.41. Each of the three correlates 0.31 to 0.32 with the log wage. These are the variables whose importance will be shared, so the ranking among them is less firm than the ranking between them and the rest.
</div>

## Fit the model and measure importance

```python
spec = dict(y="lwage", x=xs, top=3, repeats=30)
r = oes.run("interpret", data, spec)
oes.summary(r)
oes.show(r, "Permutation importance")
```

<div class="read" markdown="1">
**How to read it**

The forest explains 26.5% of the variance in the test sample, so there is a modest signal to interpret. The most important predictor is the KWW score: shuffling it lowers the test R-squared by 0.086 (standard deviation over shuffles 0.023), against 0.052 for education, 0.052 for IQ, 0.040 for urban residence and about 0.02 each for mother's education and age. Six predictors (region, hours, marital status, siblings, race and birth order) have importance below 0.01, and birth order is slightly negative, which means that shuffling it does not hurt the fit at all. The importance share of KWW is 26%. The rank correlation between importance and the absolute standardised OLS coefficient is 0.48: the two models agree that education and urban residence matter (OLS coefficients 0.094 and 0.093 per standard deviation) but not on KWW, which the forest ranks first and OLS gives only 0.037, or on marital status, which has the third-largest OLS coefficient (0.066) and almost no importance in the forest.
</div>

## Shape of the effects: partial dependence

```python
oes.show(r, "Partial dependence")
oes.plot(r, "Partial dependence of", ncols=3, size=(4.2, 3))
```

<div class="read" markdown="1">
**How to read it**

Moving KWW from its 5th percentile (23) to its 95th (48) raises the predicted log wage from 6.78 to 6.95, a range of 0.17 (about 17% in wages), and the profile is not a straight line (R-squared 0.73 for a linear fit), so the gain per point of KWW differs along the range; the chart shows where. Education raises the prediction by 0.15 between 11 and 18 years, and IQ by 0.10 between 77 and 127, with a nearly straight profile (R-squared 0.96). The forest does not use the whole range of the data equally, so partial dependence shows an average pattern and not the effect for any individual.
</div>

## Do the leading predictors interact?

```python
oes.show(r, "Interaction")
```

<div class="read" markdown="1">
**How to read it**

Friedman's H-squared is 0.060 for KWW and education, 0.031 for KWW and IQ and 0.011 for education and IQ. Between 1% and 6% of the variance of the joint partial dependence is therefore left after adding the separate effects: the interactions among the leading predictors are weak, and an additive model would capture most of what the forest has learned about them. The largest, KWW and education, is still small, so there is no evidence here that the value of education depends on the test score.
</div>

## How stable is the ranking?

```python
tops = {}
for name, sp in {"forest": dict(model="rf"), "boosting": dict(model="gb"), "forest, test share 40%": dict(model="rf", test_share=0.4)}.items():
    rr = oes.run("interpret", data, dict(y="lwage", x=xs, top=2, repeats=20, **sp))
    tops[name] = [f"{x[0]} ({x[1]})" for x in rr["tables"][0]["rows"][:4]] + [f"test R2 = {rr['summary'][2]['value']}"]
print(pd.DataFrame(tops, index=["1st", "2nd", "3rd", "4th", "fit"]).T.to_string())
```

<div class="read" markdown="1">
**How to read it**

The forest's top predictors are KWW, IQ and education in the first split and KWW, education and urban residence with a 40% test sample; boosting, which fits worse (test R-squared 0.15 against 0.26 for the forest), puts education, urban residence and KWW first. The same few variables appear at the top in all three, in a different order. The order among correlated predictors, and the importance numbers themselves, depend on the model and the split, so say which variables are important and avoid ranking them to the second decimal.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

The forest explains 26.5% of the variance of the log wage in a test sample of 25%. Permutation importance is largest for the KWW score (a fall in test R-squared of 0.086), followed by education (0.052), IQ (0.052) and urban residence (0.040); six of the 15 predictors have importance below 0.01. The partial dependence of the log wage on KWW, education and IQ is increasing, with ranges of 0.17, 0.15 and 0.10 between the 5th and 95th percentiles; the profile for KWW is the least linear (R-squared of a straight line 0.73). Interactions among the three leading predictors are weak (H-squared 0.01 to 0.06). Education, IQ and KWW are correlated (0.41 to 0.54), so their importances are not separable, and their order changes with the model and the test sample. These are descriptions of the fitted prediction function and not estimates of the effects of the variables on wages.
</div>

| Report | What | Where |
|---|---|---|
| Tools | Importance, partial dependence, H statistic | Step 1 |
| Data | Correlated predictors | Step 2 |
| Importance | Fall in test R-squared by predictor | Step 3 |
| Shape | Partial dependence profiles | Step 4 |
| Interactions | Friedman's H | Step 5 |
| Robustness | Model and test share | Step 6 |

**References.** Breiman, L. (2001), "Random Forests," *Machine Learning* 45. Friedman, J. (2001), "Greedy Function Approximation: A Gradient Boosting Machine," *Annals of Statistics* 29. Friedman, J. and Popescu, B. (2008), "Predictive Learning via Rule Ensembles," *Annals of Applied Statistics* 2. Fisher, A., Rudin, C. and Dominici, F. (2019), "All Models Are Wrong, but Many Are Useful," *Journal of Machine Learning Research* 20. Molnar, C. (2022), *Interpretable Machine Learning*, 2nd ed.
