title: Granger Causality in Levels (Toda-Yamamoto)
short: Granger causality
when: You ask whether one variable helps to predict another, and the series may have unit roots
data: U.S. quarterly log price level, log M2 and log GDP, 1960–2019
tools: Python · statsmodels · scipy
lead: Use this guide when your paper asks whether money predicts prices, whether prices predict output, or any "does X lead Y" question with trending macro series. The usual Granger test is invalid when the series have unit roots. The Toda-Yamamoto procedure fixes this by adding extra lags to a VAR in levels and testing only the original lags, so it does not need the series to be pre-tested for unit roots and cointegration. The guide runs it, checks that the conclusion does not depend on the lag length, and writes the result carefully, since predictive content is not causation.
---

## The question and the model

**Does money growth help predict inflation, once we use the past of prices and output too?** Variable \(x\) Granger-causes \(y\) if past values of \(x\) improve the forecast of \(y\) given the past of \(y\) and of the other variables. In a VAR in levels with \(p\) lags,

\[
y_t = c + \sum_{l=1}^{p}A_l\, y_{t-l} + \varepsilon_t ,
\]

the null that \(x\) does not cause \(y\) is that the coefficients of \(x\) in the equation of \(y\) are zero for all \(p\) lags. With I(1) variables the Wald statistic does not have a chi-squared distribution. Toda and Yamamoto (1995) estimate a VAR with \(p + d_{max}\) lags, where \(d_{max}\) is the highest order of integration, and test only the first \(p\) lags of \(x\). The Wald statistic is then asymptotically chi-squared with \(p\) degrees of freedom.

<div class="read" markdown="1">
**Before you test anything**

1. **Find \(d_{max}\).** Run the unit-root tests (see the unit-root guide) and take the highest order of integration; for log prices, money and GDP it is usually 1.
2. **Choose \(p\) by an information criterion on the VAR in levels.** The result must not depend on that choice; check it.
3. **Do not difference or detrend the data first.** The method is designed for levels.
</div>

<div class="write" markdown="1">
**How to write it: methods sentence**

Because the variables may be integrated, we test Granger non-causality with the procedure of Toda and Yamamoto (1995): a VAR in levels with \(p\) lags chosen by AIC plus one extra lag (the maximum order of integration), and a Wald test on the first \(p\) lags of the causing variable.
</div>

## Load the data

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

d = pd.read_csv("money.csv"); names = ["log_price", "log_m2", "log_gdp"]
x = d[names].to_numpy(float); n, k = x.shape
print(f"{n} quarters, {d.date.iloc[0]} to {d.date.iloc[-1]}")
rows = [[c, round(adfuller(d[c], regression="ct", autolag="BIC")[1], 3), round(adfuller(d[c].diff().dropna(), regression="c", autolag="BIC")[1], 4)] for c in names]
print(pd.DataFrame(rows, columns=["series", "ADF p (level, trend)", "ADF p (first difference)"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

M2 and GDP are I(1): a unit root is not rejected in levels and is rejected in first differences. The price level is the awkward case: its first difference also fails to reject a unit root (p = 0.196), because inflation is very persistent in this sample, so the price level may be I(2). The method only needs an upper bound on the order of integration, so a safe choice is \(d_{max}=2\). We use \(d_{max}=1\) in the main test, as is usual, and show in Step 4 which conclusions survive \(d_{max}=2\).
</div>

## Step 1: Choose the lag length

```python
sel = VAR(x).select_order(8)
print(pd.DataFrame({c.upper(): [sel.selected_orders[c]] for c in ["aic", "bic", "hqic", "fpe"]}, index=["selected lag"]).to_string())
```

<div class="read" markdown="1">
**How to read it**

The criteria do not agree: AIC and FPE choose 4 lags, HQIC 3 and BIC 2. We use 4 and show in Step 3 what happens with other lengths. Too few lags leave serial correlation and invalidate the test; too many reduce its power.
</div>

## Step 2: The Toda-Yamamoto test

```python
p, dmax = 4, 1; L = p + dmax
def design(L):
    return np.column_stack([np.ones(n - L)] + [x[L - l:n - l, j] for l in range(1, L + 1) for j in range(k)])
def wald(i, j, p):
    Lp = p + dmax; X = design(Lp); yv = x[Lp:, i]
    b = np.linalg.lstsq(X, yv, rcond=None)[0]; e = yv - X @ b; s2 = e @ e / (len(yv) - X.shape[1])
    V = s2 * np.linalg.inv(X.T @ X); idx = [1 + (l * k + j) for l in range(p)]      # only the first p lags are tested
    W = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
    return W, stats.chi2.sf(W, p)
rows = []
for i in range(k):
    for j in range(k):
        if i != j:
            W, pv = wald(i, j, p); rows.append([f"{names[j]} -> {names[i]}", round(W, 2), p, round(pv, 4), "reject" if pv < .05 else "do not reject"])
print(pd.DataFrame(rows, columns=["direction", "Wald chi2", "df", "p-value", "H0: no causality"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Money does not help to predict prices (p = 0.27), and prices do not help to predict money (p = 0.20). GDP predicts prices at the 5% level (p = 0.041), and prices (p = 0.002) and money (p = 0.020) predict GDP. GDP does not significantly predict money (p = 0.069). The evidence for money leading prices, a classic hypothesis, is therefore weak in this sample, once output is controlled for.
</div>

## Step 3: Is the conclusion sensitive to the lag length?

```python
rows = []
for pp in range(1, 7):
    rows.append([pp] + [round(wald(i, j, pp)[1], 3) for i in range(k) for j in range(k) if i != j])
cols = ["lags"] + [f"{names[j][4:]}->{names[i][4:]}" for i in range(k) for j in range(k) if i != j]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Money to prices is not significant at 5% for 1 to 5 lags (p between 0.26 and 0.83) and is significant with 6 lags (p = 0.016): an isolated rejection at one lag length is weak evidence. Prices to output and money to output are significant at 5% for every lag length from 1 to 6, so these two results are robust. GDP to prices is significant only with 4 lags (p = 0.041), which makes it fragile.
</div>

## Step 4: A more cautious dmax

```python
dmax = 2                                                    # the wald() function uses this global
rows = [[f"{names[j]} -> {names[i]}", round(wald(i, j, 4)[0], 2), round(wald(i, j, 4)[1], 4)] for i in range(k) for j in range(k) if i != j]
print(pd.DataFrame(rows, columns=["direction", "Wald chi2", "p-value (dmax = 2, p = 4)"]).to_string(index=False))
dmax = 1
```

<div class="read" markdown="1">
**How to read it**

Adding a second extra lag costs some power, and it matters here. Money still does not predict prices (p = 0.28), and money still predicts GDP (p = 0.023), so these two results are robust. But the effect of prices on GDP, which had p = 0.002 with \(d_{max}=1\), is no longer significant (p = 0.12), and GDP to prices moves from 0.041 to 0.056. These two conclusions depend on treating the price level as I(1) and should be reported as such.
</div>

## What to report in the paper

Report \(d_{max}\) and how you found it, the lag length and the criterion, the Wald statistics and p-values for every direction, and the p-values for other lag lengths. Use the word "predicts" or "Granger-causes", not "causes".

<div class="write" markdown="1">
**How to write it: results**

Table 6 gives Toda-Yamamoto tests from a VAR in levels with four lags plus one extra lag. Past money growth does not help to predict the price level (p = 0.27), and the result holds for most lag lengths. Money helps to predict GDP (p = 0.020), and this is significant at 5% for every lag length from one to six and with one more extra lag (p = 0.023). The predictive content of prices for GDP (p = 0.002) depends on treating the price level as I(1): with two extra lags it is not significant (p = 0.12). We interpret these as evidence of predictive content, not of structural causation.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Running an ordinary Granger test on levels with unit roots.** The p-values are wrong.
2. **Testing all \(p + d_{max}\) lags.** The extra lags are for the distribution and must not be tested.
3. **Differencing the series to avoid the problem.** If the variables are cointegrated, differencing loses the long-run information.
4. **Choosing the lag to get a significant result.** Report the sensitivity.
5. **Reading Granger causality as causation.** An omitted variable that moves both can produce it.
6. **Leaving out an important variable.** The result changes with the other variables in the VAR.
7. **Ignoring serial correlation in the residuals.** Check the residuals of the VAR.
8. **Too many variables for the sample.** The number of parameters grows with \(k^2 p\).
</div>
