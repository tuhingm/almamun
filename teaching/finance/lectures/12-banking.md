# Lecture 12 — Banking, Financial Intermediation, and Systemic Risk

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Analyze bank balance sheets, maturity transformation, capital and liquidity risk, and mechanisms of systemic instability.

## 1. Bank balance sheet
Assets include loans and securities; liabilities include deposits and wholesale funding; equity absorbs losses. Leverage magnifies the effect of asset-value changes on equity.

## 2. Maturity transformation
Banks fund relatively illiquid long-term assets with shorter-term liabilities. This supports liquidity provision but creates rollover and run risk. Solvency and liquidity are distinct: a solvent institution can fail from inability to meet immediate cash demands.

## 3. Capital and liquidity
Capital ratios compare loss-absorbing equity with risk-weighted or total assets. Liquidity metrics compare liquid assets and expected inflows with outflows under specified horizons. Regulatory definitions must be used precisely.

## 4. Credit risk
Expected credit loss is often decomposed into probability of default, loss given default, and exposure at default:
\[
ECL=PD\times LGD\times EAD
\]
under a simplified one-period representation. Dependence across borrowers matters for portfolio tail loss.

## 5. Runs and systemic risk
First-mover incentives, fire sales, interconnected exposures, common asset holdings, and payment-system links can transmit distress. Deposit insurance and lender-of-last-resort facilities alter incentives and confidence.

## 6. Worked example
A bank has assets 100, liabilities 92, and equity 8. A 5% asset loss reduces equity to 3, a 62.5% decline. Calculate the post-loss leverage ratio and interpret amplification.

## 7. Computational lab
Construct a stylized bank balance sheet. Stress loan defaults, deposit withdrawals, and asset haircuts; report capital and liquidity shortfalls separately.

## Problem set
1. Explain maturity transformation.
2. Distinguish liquidity and solvency.
3. Calculate ECL from supplied PD, LGD, and EAD.
4. Explain how fire sales create externalities.

## Coding exercises

These exercises use **real market data** in Python: monthly Fama–French factor returns (1926–2018), daily S&P 500 and
NASDAQ prices (1999–2018), the VIX, Moody's AAA and BAA corporate bond yields (1919–2018), all bundled with the `arch`
package, plus U.S. Treasury yields and household data from the `wooldridge` package. Install once with
`pip install arch wooldridge statsmodels scipy numpy pandas`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 12.1 — Credit scoring and the Boston Fed mortgage study

*Data:* `loanapp` — 1,989 mortgage applications in Boston, 1990 (Munnell et al., 1996): approval `approve`, `white`, housing-expense and total-debt ratios `hrat`, `obrat`, loan-to-value `loanprc`, credit history `cosign`, `chist`, `pubrec`, `mortlat1`, `mortlat2`, `unem`, `male`, `married`, `dep`, `sch`.

1. Fit a logit of approval on the financial and credit-history variables (without race). Compute the in-sample AUC (area under the ROC curve).
2. Add `white`. Report the average marginal effect of being white on the approval probability, with a robust standard error.
3. Interpret the coefficient as evidence on discrimination. What omitted-variable concerns did critics raise, and why does a high AUC not settle them?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score     # or compute AUC by ranks
d = wd.data("loanapp")
X = "hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr"
# TODO: logit without and with white; AUC; get_margeff
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
from scipy.stats import rankdata
d = wd.data("loanapp")
X = "hrat + obrat + loanprc + unem + male + married + dep + sch + cosign + chist + pubrec + mortlat1 + mortlat2 + vr"
d = d.dropna(subset=["approve", "white"] + X.split(" + "))
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
m0 = smf.logit(f"approve ~ {X}", data=d).fit(disp=0)
m1 = smf.logit(f"approve ~ white + {X}", data=d).fit(disp=0, cov_type="HC0")
print(f"n = {len(d)}, approval rate {d['approve'].mean():.3f} (white {d.loc[d.white == 1, 'approve'].mean():.3f}, non-white {d.loc[d.white == 0, 'approve'].mean():.3f})")
print(f"AUC without race {auc(d['approve'].to_numpy(), m0.predict()):.3f}; with race {auc(d['approve'].to_numpy(), m1.predict()):.3f}")
me = m1.get_margeff().summary_frame().loc["white"]
print(f"average marginal effect of white on approval: {me['dy/dx']:.3f} (se {me['Std. Err.']:.3f}, p = {me['Pr(>|z|)']:.4f})")
print(f"obrat AME: {m1.get_margeff().summary_frame().loc['obrat', 'dy/dx']:.4f} per percentage point")
```

Output:

```text
n = 1971, approval rate 0.876 (white 0.908, non-white 0.703)
AUC without race 0.781; with race 0.794
average marginal effect of white on approval: 0.083 (se 0.015, p = 0.0000)
obrat AME: -0.0047 per percentage point
```

The approval rate was 91% for white applicants and 70% for others. Financial and credit-history variables discriminate well
(AUC about 0.78), and adding race barely changes the AUC. Yet, controlling for those variables, being white raises the
approval probability by about 8 percentage points (\(p<0.001\)), close to Munnell et al.'s (1996) estimate. Critics argued
that the regression omits variables lenders observe (data errors, unobserved credit quality). A high AUC means the
**included** variables predict approval well; it says nothing about whether an omitted variable correlated with race
explains the gap. The burden is on that omitted variable to be large enough.

</details>

### Exercise 12.2 — Portfolio credit risk: the Vasicek model behind Basel capital

*Data:* Default probabilities calibrated to corporate credit (1% and 2% a year); asset correlation 0.15 as in the Basel IRB formula; loss given default 45%.

In the one-factor Vasicek model, the fraction of a large loan portfolio that defaults, conditional on the common factor \(Z\), is \(\Phi\big((\Phi^{-1}(PD)-\sqrt\rho Z)/\sqrt{1-\rho}\big)\).

1. Simulate the portfolio loss distribution (one million draws of \(Z\)) for \(PD=1\%\) and 2%.
2. Compute expected loss, the 99.9% quantile of losses, and the Basel "unexpected loss" capital charge (99.9% quantile minus expected loss), and check against the closed form.
3. Show how the capital charge changes when \(\rho\) rises from 0.12 to 0.24. Why is correlation, not just PD, central to bank capital?

```python
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0); Z = rng.standard_normal(1_000_000); LGD = 0.45
loss = lambda pd_, rho: LGD * norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * Z) / np.sqrt(1 - rho))
# TODO: EL, quantile, closed form: LGD * Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(0.999)) / sqrt(1 - rho))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0); Z = rng.standard_normal(1_000_000); LGD = 0.45
loss = lambda p, rho: LGD * norm.cdf((norm.ppf(p) - np.sqrt(rho) * Z) / np.sqrt(1 - rho))
closed = lambda p, rho: LGD * norm.cdf((norm.ppf(p) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho))
for p in (0.01, 0.02):
    for rho in (0.12, 0.15, 0.24):
        L = loss(p, rho); q = np.quantile(L, 0.999)
        print(f"PD {p:.0%}, rho {rho:.2f}: expected loss {100*L.mean():.2f}%, 99.9% loss {100*q:.2f}% (closed form {100*closed(p, rho):.2f}%), capital = UL {100*(q - L.mean()):.2f}% of exposure")
```

Output:

```text
PD 1%, rho 0.12: expected loss 0.45%, 99.9% loss 4.08% (closed form 4.06%), capital = UL 3.63% of exposure
PD 1%, rho 0.15: expected loss 0.45%, 99.9% loss 4.99% (closed form 4.96%), capital = UL 4.54% of exposure
PD 1%, rho 0.24: expected loss 0.45%, 99.9% loss 7.95% (closed form 7.91%), capital = UL 7.50% of exposure
PD 2%, rho 0.12: expected loss 0.90%, 99.9% loss 6.66% (closed form 6.63%), capital = UL 5.76% of exposure
PD 2%, rho 0.15: expected loss 0.90%, 99.9% loss 7.97% (closed form 7.93%), capital = UL 7.07% of exposure
PD 2%, rho 0.24: expected loss 0.90%, 99.9% loss 12.11% (closed form 12.05%), capital = UL 11.21% of exposure
```

Simulated 99.9% losses match the closed-form Vasicek quantile. With a 1% PD and Basel's correlation range, the capital charge
(unexpected loss) is about 4–5% of exposure, roughly ten times expected loss. Doubling the asset correlation from 0.12 to
0.24 doubles the capital requirement at a given PD: correlated defaults fatten the loss tail. This is why the Basel formula
assigns correlations by asset class, and why underestimated default correlation (as in mortgage-backed securities before
2008) left banks undercapitalized.

</details>

## Further reading
Diamond and Dybvig (1983); Freixas and Rochet, *Microeconomics of Banking*; Basel Committee standards.
