# Lecture 06 — Fixed-Income Securities, Bond Pricing, and Yield Curves

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Price fixed cash flows; distinguish coupon, spot, forward, and yield-to-maturity rates; understand credit spreads and discount curves.

## 1. Bond valuation
A default-free bond with cash flows \(C_t\) and maturity \(T\) has price
\[
P=\sum_{t=1}^{T}\frac{C_t}{(1+s_t)^t},
\]
where \(s_t\) are maturity-matched spot rates under annual compounding. For a zero-coupon bond, \(P_T=F/(1+s_T)^T\).

## 2. Yield to maturity
YTM is the single internal discount rate solving the bond price equation. It is a summary measure, not necessarily the realized return if coupons are reinvested at different rates or the bond is sold before maturity.

## 3. Spot and forward rates
Spot rates discount cash flows at each maturity. Forward rates are implied future borrowing rates from spot rates. Under annual compounding,
\[
(1+s_n)^n=(1+s_{n-1})^{n-1}(1+f_{n-1,n}).
\]

## 4. Price conventions and credit
Clean price excludes accrued interest; dirty price includes it. Credit spreads compensate for expected default losses, risk premia, liquidity, and other factors. Promised yield is not expected return when default is possible.

## 5. Worked example
A two-year bond pays 5 at year one and 105 at year two. Given spot rates \(s_1\) and \(s_2\), compute price as \(5/(1+s_1)+105/(1+s_2)^2\). Compare with price using a single YTM.

## 6. Computational lab
Build a cash-flow pricing function, bootstrap a spot curve from coupon bonds, calculate forward rates, and quantify sensitivity to curve shifts. Check compounding and day-count conventions.

## Problem set
1. Price a coupon bond from a spot curve.
2. Distinguish YTM from realized holding-period return.
3. Derive a one-period forward rate.
4. Explain clean and dirty prices.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 6.1 — Corporate credit spreads and implied default probabilities

*Data:* Moody's AAA and BAA corporate bond yields, monthly, 1919–2018 (`arch.data.default`).

1. Compute the BAA–AAA spread. Report its mean, minimum and maximum, and the dates of the three highest monthly values.
2. Under risk neutrality, a bond with annual default probability \(p\) and recovery rate \(R\) must offer a spread that compensates for expected losses: approximately \(\text{spread}\approx p(1-R)\). With \(R=40\%\), compute the implied annual default probability at the median spread and at the 2008–09 peak.
3. Historical BAA default rates are about 0.2–0.5% per year. What does the gap tell you about the **credit risk premium** (the "credit spread puzzle")?

```python
import numpy as np, pandas as pd
from arch.data import default
d = default.load(); d["spread"] = d["BAA"] - d["AAA"]
# TODO: describe; nlargest; p = spread / (1 - R)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd
from arch.data import default
d = default.load(); d["spread"] = d["BAA"] - d["AAA"]
s = d["spread"]
print(f"BAA-AAA spread (pp): mean {s.mean():.2f}, median {s.median():.2f}, min {s.min():.2f}, max {s.max():.2f}")
print("three highest months:", {k.strftime('%Y-%m'): round(v, 2) for k, v in s.nlargest(3).items()})
print("peaks in later crises:", {k: round(s.loc[k].max(), 2) for k in ("1982", "2001", "2009")})
R = 0.4
for lab, v in (("median", s.median()), ("2008-09 peak", s.loc["2008":"2009"].max())):
    print(f"implied risk-neutral default probability at the {lab} spread: {v/(1-R):.2f}% per year")
```

Output:

```text
BAA-AAA spread (pp): mean 1.18, median 0.94, min 0.32, max 5.64
three highest months: {'1932-05': 5.64, '1932-06': 5.59, '1932-07': 5.53}
peaks in later crises: {'1982': np.float64(2.69), '2001': np.float64(1.28), '2009': np.float64(3.09)}
implied risk-neutral default probability at the median spread: 1.57% per year
implied risk-neutral default probability at the 2008-09 peak spread: 5.63% per year
```

The BAA–AAA spread averaged about 1.2 points, with peaks in the Great Depression (above 5.5 points in 1932), 1982 and the
2008–09 crisis (about 3.1 points). With 40% recovery, the median spread implies a risk-neutral default probability of about
1.6% a year, and the 2008–09 peak about 5.6%. Actual BAA default rates average only about 0.2–0.5% a year, so most of the
spread is compensation for risk and illiquidity rather than expected losses — the "credit spread puzzle" (Huang and Huang,
2012). Risk-neutral probabilities are not physical probabilities.

</details>

### Exercise 6.2 — Testing the expectations hypothesis with T-bill holding yields

*Data:* `intqrt` — quarterly holding yield on 6-month bills held for 3 months (`hy6`), 3-month yield (`r3`).

As in a classic example in Wooldridge's *Introductory Econometrics*: if the expectations hypothesis holds, the excess holding return of a
6-month bill over three months, \(hy6_t-r3_{t-1}\), should be unpredictable from information at \(t-1\).

1. Regress \(hy6_t\) on \(hy3_{t-1}\) and the lagged spread \((r6-r3)_{t-1}\), as in Wooldridge's example.
2. Test whether the coefficient on \(hy3_{t-1}\) is 1 and the spread coefficient is 0.
3. What does a significant spread coefficient imply about time-varying term premia?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
# TODO: hy6 ~ hy3_1 + spr63_1 (both are in the dataset)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("intqrt")
r = smf.ols("hy6 ~ hy3_1 + spr63_1", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(r.params.round(4).to_string()); print("HAC se:", r.bse.round(4).to_dict())
w = r.wald_test("hy3_1 = 1, spr63_1 = 0", scalar=True)
print(f"joint test (hy3_1 = 1, spread = 0): chi2(2) = {w.statistic:.2f}, p = {w.pvalue:.4f};  n = {int(r.nobs)}")
print(f"average excess holding return hy6 - hy3_1: {(d['hy6'] - d['hy3_1']).mean():.3f} pp per year")
```

Output:

```text
Intercept   -0.1229
hy3_1        1.0535
spr63_1      0.4800
HAC se: {'Intercept': 0.0666, 'hy3_1': 0.0428, 'spr63_1': 0.1059}
joint test (hy3_1 = 1, spread = 0): chi2(2) = 23.31, p = 0.0000;  n = 123
average excess holding return hy6 - hy3_1: 0.109 pp per year
```

The coefficient on the lagged 3-month holding yield is close to one, but the lagged spread has a coefficient of about 0.48
(\(t\approx4.5\)), and the joint test rejects the expectations hypothesis decisively. When the yield curve is steep, the
6-month bill earns a higher excess return over the next quarter: the term premium is **time-varying** and predictable
from the slope of the curve. This is the short-maturity version of Campbell and Shiller's (1991) and Cochrane and
Piazzesi's (2005) findings for longer bonds.

</details>

## Further reading
Tuckman and Serrat, *Fixed Income Securities*; Fabozzi, *Bond Markets, Analysis, and Strategies*.
