# Lecture 02 — Risk, Probability, and Expected Utility

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Represent uncertainty with probability distributions; derive expected utility and certainty equivalents; understand risk aversion and downside risk.

## 1. Random returns
A risky payoff is a random variable with distribution \(F(R)\). Expected return and variance are
\[
E(R)=\int R\,dF(R),\qquad \operatorname{Var}(R)=E[(R-E(R))^2].
\]
Higher moments capture skewness and tail thickness, which variance alone cannot describe.

## 2. Expected utility
An investor chooses a portfolio to maximize \(E[u(W_1)]\). Absolute risk aversion is \(A(W)=-u''(W)/u'(W)\); relative risk aversion is \(R(W)=-Wu''(W)/u'(W)\). Concavity, \(u''<0\), implies risk aversion.

## 3. Certainty equivalent and risk premium
The certainty equivalent \(CE\) solves \(u(CE)=E[u(W)]\). The risk premium is \(E(W)-CE\). For small risks, a second-order approximation gives a risk premium near \(\frac12 A(W)\operatorname{Var}(W)\), evaluated at a suitable wealth level.

## 4. Mean-variance preferences
A quadratic objective can be written
\[
U(w)=w'\mu-\frac{\gamma}{2}w'\Sigma w.
\]
This is exact for quadratic utility and is also used as a portfolio criterion. It does not imply that all investors care only about variance under arbitrary return distributions.

## 5. Worked example
An investment pays 120 or 80 with equal probability. Compute expected payoff and variance. For logarithmic utility, compare expected utility with utility at expected wealth to establish the certainty-equivalent relation.

## 6. Computational lab
Simulate normal and negatively skewed return distributions with equal means and variances. Compare expected shortfall and expected utility to demonstrate why equal variance need not imply equal downside risk.

## Problem set
1. Define absolute and relative risk aversion.
2. Derive the second-order risk-premium approximation.
3. Explain the certainty equivalent.
4. Give a case where variance fails to distinguish economically relevant risk.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 2.1 — How much in stocks? Expected-utility portfolio choice with historical returns

*Data:* Fama–French monthly market and T-bill returns, compounded to annual returns, 1927–2017.

A CRRA investor chooses the share \(w\) of wealth in stocks (the rest in bills) to maximize \(E[u(1+r_f+w\,r^e)]\), using the empirical distribution of annual returns.

1. Solve numerically for \(w^*\) with \(\gamma=2,5,10\), allowing \(0\le w\le2\).
2. Compare with the Merton/mean–variance approximation \(w=\mu_e/(\gamma\sigma^2)\).
3. Compute the certainty-equivalent gain from holding \(w^*\) rather than all bills.

```python
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

a = (1 + ff).groupby(ff.index.year).prod() - 1                    # annual compounded returns
a = a.loc[1927:2017]; rf = a["RF"]; rex = (1 + a["Mkt-RF"] + a["RF"]) / (1 + rf) - 1  # approx
# TODO: maximize mean(u(1 + rf + w * (rm - rf))) over w
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from scipy import optimize
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

m = ff.assign(RM=ff["Mkt-RF"] + ff["RF"])
a = (1 + m[["RM", "RF"]]).groupby(m.index.year).prod() - 1
a = a.loc[1927:2017]; rm, rf = a["RM"].to_numpy(), a["RF"].to_numpy(); re = rm - rf
mu, s2 = re.mean(), re.var()
print(f"annual excess return: mean {100*mu:.2f}%, sd {100*np.sqrt(s2):.2f}%, {len(re)} years")
u = lambda W, g: W**(1 - g) / (1 - g)
for g in (2, 5, 10):
    EU = lambda w: -np.mean(u(1 + rf + w * re, g))
    w = optimize.minimize_scalar(EU, bounds=(0, 2), method="bounded").x
    ce = lambda w: ((1 - g) * np.mean(u(1 + rf + w * re, g)))**(1 / (1 - g))
    print(f"gamma = {g:2d}: optimal share {w:.3f};  mean-variance mu/(g s2) = {mu/(g*s2):.3f};  certainty-equivalent gain vs bills {100*(ce(w) - ce(0)):.2f}% per year")
```

Output:

```text
annual excess return: mean 8.51%, sd 20.30%, 91 years
gamma =  2: optimal share 0.959;  mean-variance mu/(g s2) = 1.032;  certainty-equivalent gain vs bills 4.44% per year
gamma =  5: optimal share 0.406;  mean-variance mu/(g s2) = 0.413;  certainty-equivalent gain vs bills 1.90% per year
gamma = 10: optimal share 0.210;  mean-variance mu/(g s2) = 0.206;  certainty-equivalent gain vs bills 1.03% per year
```

With the historical annual premium of about 8.5% and volatility of about 20%, an investor with \(\gamma=5\) would hold about
40% in stocks, and one with \(\gamma=2\) almost everything. The exact expected-utility solution is very close to the
mean–variance rule \(\mu/(\gamma\sigma^2)\): with annual horizons, higher moments matter little. The certainty-equivalent
gains from participating are large (1–4% a year), which makes the empirical fact that many households hold no stocks at all
a puzzle ("non-participation"). These calculations treat historical averages as known — Exercise 3.2 shows the cost of that
assumption.

</details>

### Exercise 2.2 — The equity premium puzzle in numbers

*Data:* Fama–French annual market excess returns and U.S. real per-capita consumption growth (`macrodata`, 1960–2008).

With CRRA utility, the consumption CAPM implies \(E[R^e]\approx\gamma\,\operatorname{Cov}(\Delta\log c,R^e)\).

1. For 1960–2008, compute the average annual equity premium, the volatility of consumption growth, and the covariance between consumption growth and excess returns.
2. Solve for the implied coefficient of relative risk aversion \(\gamma\).
3. Also compute the Hansen–Jagannathan bound: the volatility of the SDF must be at least \(SR\), and with CRRA, \(\sigma(m)\approx\gamma\sigma(\Delta\log c)\). What \(\gamma\) does the bound require?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

mac = sm.datasets.macrodata.load_pandas().data
c = (mac["realcons"] / mac["pop"]).groupby(mac["year"]).mean()
# TODO: annual excess returns 1960-2008; dc = diff(log c); gamma = mean(re) / cov(dc, re)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

mac = sm.datasets.macrodata.load_pandas().data
c = (mac["realcons"] / mac["pop"]).groupby(mac["year"]).mean()
dc = np.log(c).diff().loc[1960:2008]
m = ff.assign(RM=ff["Mkt-RF"] + ff["RF"])
a = (1 + m[["RM", "RF"]]).groupby(m.index.year).prod() - 1
re = (a["RM"] - a["RF"]).loc[1960:2008]
cov = np.cov(dc, re)[0, 1]; sr = re.mean() / re.std()
print(f"1960-2008: equity premium {100*re.mean():.2f}%, sd of excess return {100*re.std():.1f}%, Sharpe ratio {sr:.3f}")
print(f"consumption growth: mean {100*dc.mean():.2f}%, sd {100*dc.std():.2f}%, correlation with excess returns {np.corrcoef(dc, re)[0,1]:.3f}")
print(f"implied gamma from E[Re] = gamma Cov(dc, Re): {re.mean()/cov:.1f}")
print(f"Hansen-Jagannathan: gamma >= SR / sd(dc) = {sr/dc.std():.1f}")
```

Output:

```text
1960-2008: equity premium 5.24%, sd of excess return 17.9%, Sharpe ratio 0.293
consumption growth: mean 2.33%, sd 1.68%, correlation with excess returns 0.131
implied gamma from E[Re] = gamma Cov(dc, Re): 132.9
Hansen-Jagannathan: gamma >= SR / sd(dc) = 17.4
```

Consumption growth is smooth (SD about 1.7%) and only weakly correlated with stock returns (0.13), so the covariance between
the two is tiny. Matching a 5% equity premium then requires a coefficient of relative risk aversion of over 130. Even the
Hansen–Jagannathan bound, which ignores the low correlation, requires \(\gamma\ge17\). Economists find risk aversion above
about 10 implausible, so this is the equity premium puzzle (Mehra and Prescott, 1985). Habit formation, long-run risks and
rare disasters are the main proposed resolutions.

</details>

## Further reading
Pratt (1964); Arrow, *Essays in the Theory of Risk-Bearing*; Cochrane, *Asset Pricing*.
