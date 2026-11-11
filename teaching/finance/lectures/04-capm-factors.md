# Lecture 04 — CAPM, Beta, and Empirical Factor Models

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Derive CAPM implications; estimate beta and alpha; interpret factor models and distinguish systematic from idiosyncratic risk.

## 1. CAPM
Under its benchmark assumptions, the CAPM implies
\[
E(R_i)-R_f=\beta_i[E(R_M)-R_f],\quad
\beta_i=\frac{\operatorname{Cov}(R_i,R_M)}{\operatorname{Var}(R_M)}.
\]
Beta measures covariance exposure to the market portfolio, not total volatility.

## 2. Security market line
The security market line maps beta into expected return. Assets above the line have positive model-implied alpha; assets below it have negative alpha. This interpretation is conditional on the benchmark model and measurement choices.

## 3. Time-series regression
Estimate
\[
R_{it}-R_{ft}=\alpha_i+\beta_i(R_{Mt}-R_{ft})+\varepsilon_{it}.
\]
The intercept is Jensen's alpha. Standard errors should account for heteroskedasticity and serial dependence where present. Nonsynchronous trading can bias beta estimates.

## 4. Multifactor models
A linear factor model is
\[
R_{it}-R_{ft}=\alpha_i+\boldsymbol\beta_i'\mathbf f_t+\varepsilon_{it}.
\]
Fama-French factors include market, size, and value; later specifications add profitability and investment. Momentum is another commonly studied factor. Factors can be traded portfolios or constructed mimicking portfolios.

## 5. Cross-sectional tests
Fama-MacBeth procedures estimate factor prices in repeated cross-sectional regressions. Inference must consider generated regressors, beta estimation error, and cross-sectional dependence.

## 6. Worked example
If \(R_f=3\%\), market expected return is 8%, and beta is 1.2, CAPM expected return is 9%. Explain why an asset with low standalone variance can still have high beta.

## 7. Computational lab
Estimate rolling market betas and CAPM/factor regressions. Report confidence intervals, residual diagnostics, factor definitions, and out-of-sample alpha stability.

## Problem set
1. Derive beta from covariance and variance.
2. Interpret Jensen's alpha.
3. Compare CAPM and a multifactor specification.
4. Explain beta measurement error.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 4.1 — Does the CAPM price size and value? Alphas and the GRS test

*Data:* Fama–French monthly `SMB`, `HML` and `Mkt-RF`, July 1963 – November 2018.

1. Run time-series CAPM regressions \(R^e_{it}=\alpha_i+\beta_i\,(R_m-R_f)_t+\varepsilon_{it}\) for SMB and HML. Report annualized alphas, betas and HAC \(t\)-statistics.
2. Test \(\alpha_{SMB}=\alpha_{HML}=0\) jointly with the Gibbons–Ross–Shanken (GRS) statistic.
3. Interpret: what does a significant HML alpha say about the CAPM?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

d = ff.loc["1963-07":]
# TODO: OLS of SMB and HML on Mkt-RF; GRS = (T-N-K)/N * a' S^-1 a / (1 + mu' O^-1 mu) ~ F(N, T-N-K)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

d = ff.loc["1963-07":]; X = sm.add_constant(d["Mkt-RF"]); T, N, K = len(d), 2, 1
alphas, resid = [], []
for y in ("SMB", "HML"):
    r = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alphas.append(r.params["const"]); resid.append(r.resid)
    print(f"{y}: alpha {1200*r.params['const']:.2f}% per year (HAC t = {r.tvalues['const']:.2f}), beta {r.params['Mkt-RF']:.3f}, R2 {r.rsquared:.3f}")
a = np.array(alphas); Sig = np.cov(np.array(resid)); mu = d["Mkt-RF"].mean(); om = d["Mkt-RF"].var()
grs = (T - N - K) / N * (a @ np.linalg.solve(Sig, a)) / (1 + mu**2 / om)
print(f"GRS = {grs:.2f}, p = {stats.f.sf(grs, N, T - N - K):.4f}")
```

Output:

```text
SMB: alpha 1.23% per year (HAC t = 0.88), beta 0.204, R2 0.085
HML: alpha 4.98% per year (HAC t = 3.12), beta -0.167, R2 0.068
GRS = 8.69, p = 0.0002
```

SMB's CAPM alpha is small and insignificant in 1963–2018, but HML earns an alpha of about 5% a year (\(t\approx3.1\)) — with a
**negative** beta, so its average return cannot be explained as compensation for market risk. The GRS test rejects that both
alphas are zero. This is the anomaly behind the Fama–French three-factor model: the CAPM fails to price value stocks.
Whether HML is a risk factor (distress risk) or mispricing (overreaction) remains debated (Lecture 14).

</details>

### Exercise 4.2 — Has the value premium disappeared?

*Data:* Fama–French monthly `HML` and `Mkt-RF`, 1926–2018.

1. Compute HML's average return and CAPM alpha (annualized) in four periods: 1926–1962, 1963–1990, 1991–2006, 2007–2018.
2. Compute a 10-year rolling CAPM beta of HML.
3. Is there statistical evidence that the value premium after 2007 differs from its pre-2007 average? Test with a dummy-variable regression and HAC standard errors.

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

# TODO: loop over periods; rolling cov/var for beta; regression HML ~ post2007
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from arch.data import frenchdata
ff = frenchdata.load() / 100
ff.index = pd.period_range("1926-07", periods=len(ff), freq="M")   # monthly, decimal returns

d = ff.rename(columns={"Mkt-RF": "mkt"}).copy()
for lo, hi in (("1926-07", "1962-12"), ("1963-01", "1990-12"), ("1991-01", "2006-12"), ("2007-01", "2018-11")):
    s = d.loc[lo:hi]; r = sm.OLS(s["HML"], sm.add_constant(s["mkt"])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print(f"{lo[:4]}-{hi[:4]}: mean HML {1200*s['HML'].mean():5.2f}%/yr, CAPM alpha {1200*r.params['const']:5.2f}% (t = {r.tvalues['const']:.2f}), beta {r.params['mkt']:.2f}")
beta = d["HML"].rolling(120).cov(d["mkt"]) / d["mkt"].rolling(120).var()
print(f"rolling 10-year beta of HML: min {beta.min():.2f} ({beta.idxmin()}), max {beta.max():.2f} ({beta.idxmax()}), latest {beta.iloc[-1]:.2f}")
d["post"] = (d.index >= pd.Period("2007-01", "M")).astype(int)
t = smf.ols("HML ~ post", data=d.loc["1963-07":]).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(f"1963-2018: HML = {1200*t.params['Intercept']:.2f}% + {1200*t.params['post']:.2f}% x post-2007  (t = {t.tvalues['post']:.2f}, p = {t.pvalues['post']:.3f})")
```

Output:

```text
1926-1962: mean HML  4.95%/yr, CAPM alpha  1.17% (t = 0.56), beta 0.37
1963-1990: mean HML  5.57%/yr, CAPM alpha  6.45% (t = 3.59), beta -0.20
1991-2006: mean HML  6.42%/yr, CAPM alpha  9.58% (t = 2.82), beta -0.37
2007-2018: mean HML -2.54%/yr, CAPM alpha -3.98% (t = -1.47), beta 0.17
rolling 10-year beta of HML: min -0.49 (2001-07), max 0.55 (1942-06), latest 0.23
1963-2018: HML = 5.69% + -8.23% x post-2007  (t = -2.54, p = 0.011)
```

The value premium was strong from 1963 to 2006, with large CAPM alphas, but **negative** from 2007 to 2018 (about \(-2.5\%\) a
year). A dummy regression finds the post-2007 drop statistically significant (\(p\approx0.01\)). HML's market beta also
changes sign over time, so its CAPM alpha and its raw return can tell different stories. Possible explanations include
arbitrage after publication (McLean and Pontiff, 2016), the rise of intangible capital that book values miss, and bad luck.
Twelve years of negative returns are still consistent with a positive long-run premium at these volatilities.

</details>

## Further reading
Sharpe (1964); Lintner (1965); Fama and French (1993, 2015); Campbell, Lo and MacKinlay.
