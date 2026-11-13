title: Structural VAR (Cholesky, Long-Run and Proxy Identification)
short: SVAR
when: You want the effect of a structural shock on several variables, and you can defend how the shock is identified
data: U.S. quarterly data, 1960–2019: real GDP growth and the unemployment rate; U.S. monthly data, 1990–2019: policy rate, industrial production, prices, credit spread and a high-frequency monetary shock
tools: Python · OES engine · statsmodels
oes: svar
lead: A VAR describes how variables move together, but its residuals are mixtures of structural shocks. A structural VAR (SVAR) adds the assumptions that separate them. This guide covers the three identification schemes most used in applied work, a recursive ordering, long-run restrictions (Blanchard and Quah 1989) and an external instrument (proxy-SVAR), and shows how to choose the lags, check stability, build bootstrap bands, test how much the answer depends on the identifying assumption, and report it.
---

## The question and the model

**What does a structural shock do to the system?** The reduced-form VAR is

\[
y_t = c + A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t, \qquad \mathrm{E}[u_t u_t'] = \Sigma,
\]

and the residuals are linked to the structural shocks \(\varepsilon_t\) (uncorrelated, unit variance) by \(u_t = B \varepsilon_t\). The data give \(\Sigma = BB'\), which does not pin down \(B\): any rotation of \(B\) fits equally well, so identification needs \(n(n-1)/2\) extra restrictions. The impulse response of variable \(i\) to shock \(j\) at horizon \(h\) is then \(\Psi_h B\), where \(\Psi_h\) are the moving-average coefficients of the VAR.

Three schemes are common. **Recursive (Cholesky):** \(B\) is lower triangular, so a variable ordered earlier does not respond to later shocks within the period. **Long-run restrictions:** the cumulative effect of some shocks on some variables is zero, for example demand shocks have no permanent effect on output (Blanchard and Quah 1989). **External instrument (proxy-SVAR):** a series \(z_t\) correlated with one structural shock and uncorrelated with the others identifies that shock's column of \(B\) (Stock and Watson 2012, Mertens and Ravn 2013).

<div class="read" markdown="1">
**Three checks before you trust a structural VAR**

1. **Can you defend the identifying assumption in words?** The restrictions are not tested by the data (a recursive or long-run scheme is exactly identified). Write down the economic argument, and test how much the answer depends on it by changing the ordering or the scheme.
2. **Is the VAR well specified?** Stationary or differenced variables, enough lags to leave the residuals white, and a stable system. Long-run identification also needs a stationary VAR.
3. **Is the instrument valid (proxy-SVAR)?** Relevance is testable (the first-stage F statistic) and exogeneity is not, so the argument for exogeneity must come from how the instrument was built.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We estimate a structural VAR for real GDP growth and the unemployment rate and identify supply and demand shocks with the long-run restriction of Blanchard and Quah (1989): only the supply shock has a permanent effect on the level of output. The VAR has four lags, a compromise between the Akaike criterion (eight lags) and the Schwarz and Hannan-Quinn criteria (two lags); two, six and eight lags are robustness checks. Confidence bands are 95% percentile intervals from a wild residual bootstrap with 500 replications, which is robust to heteroskedasticity. We check the result against a recursive identification with both orderings.
</div>

## Load the data and check the variables

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine
from statsmodels.tsa.stattools import adfuller, kpss

data = open("yield.csv").read()
df = pd.read_csv("yield.csv")[["date", "gdp_growth", "unemployment"]].dropna()
print(df.date.iloc[0], "to", df.date.iloc[-1], "|", len(df), "quarters")
print(df[["gdp_growth", "unemployment"]].describe().round(2).loc[["mean", "std", "min", "max"]])
for c in ["gdp_growth", "unemployment"]:
    print(f"{c:12s} ADF p = {adfuller(df[c], autolag='AIC')[1]:.3f}   KPSS p = {kpss(df[c], nlags='auto')[1]:.3f}")
```

<div class="read" markdown="1">
**How to read it**

The sample has 240 quarters (1960Q1-2019Q4). Real GDP growth (annualised) averages 3.01% with a standard deviation of 3.25; the unemployment rate averages 5.97% with a standard deviation of 1.61. Both series are treated as stationary: the ADF test rejects a unit root for growth (p < 0.001) and for unemployment (p = 0.045, borderline), and the KPSS test does not reject stationarity for either (p = 0.059 and p = 0.10). Unemployment is very persistent, which is why the largest VAR root is close to one. Long-run identification needs a stationary VAR, so this check matters.
</div>

## Choose the lags and check stability

```python
r = oes.run("svar", data, dict(vars=["gdp_growth", "unemployment"], time="date", identification="longrun", horizon=20, boot=50))
oes.show(r, "Lag-order")
oes.show(r, "Stability")
```

<div class="read" markdown="1">
**How to read it**

The Akaike criterion keeps falling slowly up to eight lags (-1.120), but most of the gain comes by two lags (-1.087 against -1.118 at four). The Schwarz and Hannan-Quinn criteria both choose two lags. We use four lags as a compromise and test two, six and eight below. With eight lags the largest root of the VAR is 0.93, so the system is stable and the long-run restriction is well defined.
</div>

## Identify with long-run restrictions (Blanchard–Quah)

```python
spec = dict(vars=["gdp_growth", "unemployment"], time="date", identification="longrun", lags=4, horizon=20, boot=500)
supply = oes.run("svar", data, dict(spec, shock="gdp_growth"))
demand = oes.run("svar", data, dict(spec, shock="unemployment"))
oes.show(supply, "Long-run impact")
oes.show(supply, "Share of forecast-error variance")
oes.show(demand, "Share of forecast-error variance")
oes.plot(supply, "Response of", ncols=2, size=(4.2, 2.6))
```

<div class="read" markdown="1">
**How to read it**

The long-run impact matrix shows the identifying restriction: the entry for the effect of the unemployment ("demand") shock on GDP growth is exactly zero, so only the supply shock moves the level of output permanently. The supply shock raises the cumulated response of annualised growth by 3.97, which is about 1% of the level of output. The supply shock explains 87% of the forecast-error variance of GDP growth at one quarter and 80% at five years, and 5% of that of unemployment at one quarter, rising to 38% at five years. The demand shock explains 95% of the variance of unemployment at one quarter and 62% at five years, and 13% to 20% of that of GDP growth.
</div>

```python
rows = lambda r, v: [x for x in r["tables"][0]["rows"] if x[0] == v]
for name, r in [("supply shock", supply), ("demand shock", demand)]:
    print(name)
    for v in ["gdp_growth", "unemployment"]:
        print("  ", v, [(x[1], x[2], x[3], x[4]) for x in rows(r, v) if x[1] in (0, 4, 8, 20)])
```

<div class="read" markdown="1">
**How to read it**

A supply shock raises GDP growth by 2.69 points on impact (band 1.46 to 2.82), 0.42 after one year, and the effect dies out after two years, which leaves the level of output permanently higher. Unemployment falls by 0.30 points after four to eight quarters (band at four quarters -0.46 to 0.05, at eight quarters -0.47 to 0.00), a borderline-significant decline that fades by year five. The demand-side shock raises unemployment by 0.22 points on impact and 0.45 after one year, and lowers GDP growth by 1.02 points on impact (band -2.37 to 0.03), after which growth is above its baseline (0.17 at two years, band 0.02 to 0.41) as unemployment returns to normal. The unemployment response is back to zero within five years (band -0.05 to 0.17), as the identification requires for the demand shock.
</div>

## Compare with a recursive identification

```python
chol = {}
for order in (["gdp_growth", "unemployment"], ["unemployment", "gdp_growth"]):
    for shock in order:
        r = oes.run("svar", data, dict(vars=order, time="date", identification="cholesky", lags=4, horizon=20, boot=100, shock=shock))
        chol[(" -> ".join(order), shock)] = {v: [x[2] for x in rows(r, v) if x[1] in (0, 4, 8, 20)] for v in order}
for k, v in chol.items():
    print(k, {a: [round(z, 2) for z in b] for a, b in v.items()})
```

<div class="read" markdown="1">
**How to read it**

The recursive schemes give the same shapes but different impact effects, because each ordering assumes away one of the two contemporaneous responses. With growth ordered first, a growth shock lowers unemployment by 0.44 after one year and 0.39 after two; an unemployment shock has no effect on growth on impact by assumption. With unemployment ordered first, the unemployment shock raises unemployment by 0.50 after one year and lowers growth by 1.59 on impact. The long-run scheme (-1.02 on impact) lies between the two. The persistence of the unemployment response (a peak of 0.31 to 0.50 after about a year, mostly gone after five years) is similar in all three, so the conclusion about dynamics does not depend on the identification, while the impact effect of the unemployment shock on growth does.
</div>

## Proxy-SVAR: check the instrument before the responses

```python
mp = open("monetary_proxy.csv").read()
for ind in ["gs1", "ffr"]:
    spec_p = dict(vars=[ind, "log_ip", "log_cpi", "credit_spread"], shock=ind, identification="proxy", instrument=["mp_shock"], time="date",
                  lags=12, horizon=36, boot=300)
    rp = oes.run("svar", mp, spec_p)
    F = [s["value"] for s in rp["summary"] if s["label"] == "First-stage F"][0]
    print(f"policy indicator {ind}: first-stage robust F = {F:.1f}")
    oes.show(rp, "Instrument")
```

<div class="read" markdown="1">
**How to read it**

The same instrument gives very different first stages depending on the policy indicator it is paired with. For the one-year Treasury yield the robust first-stage F is 4.4 (R-squared 0.026): the instrument is weak and the engine's warning applies, so the responses would be unreliable. For the federal funds rate the F is 17.4 (R-squared 0.049), above the rule-of-thumb value of 10. Choose the policy indicator and the sample for reasons stated in advance and report the F statistic for the choice you use, not for the best of several; exogeneity of the instrument cannot be tested.
</div>

```python
rp = oes.run("svar", mp, dict(spec_p, vars=["ffr", "log_ip", "log_cpi", "credit_spread"], shock="ffr"))
oes.plot(rp, "Response of", ncols=2, size=(4.2, 2.6))
for v in ["ffr", "log_ip", "log_cpi", "credit_spread"]:
    print(v, [(x[1], x[2], x[3], x[4]) for x in rows(rp, v) if x[1] in (0, 12, 24, 36)])
```

<div class="read" markdown="1">
**How to read it**

The shock is scaled to raise the federal funds rate by one percentage point on impact. The rate peaks at 2.31 points after twelve months (band 1.12 to 2.94) and returns to the baseline by month 36. Industrial production rises insignificantly (1.32 after twelve months, band -1.92 to 4.29), and prices are close to zero throughout (-0.11 on impact, 0.20 after 24 months, bands including zero). The credit spread falls by 0.71 after twelve months (band -1.27 to -0.17). The estimates are too imprecise to say anything about the real effect of monetary policy; they show the workflow, and they show how wide proxy-SVAR bands can be with a short sample and a weak-to-moderate instrument.
</div>

## Robustness: lags, transformation and bootstrap

```python
base = dict(vars=["gdp_growth", "unemployment"], time="date", identification="longrun", shock="gdp_growth", horizon=20)
variants = {"4 lags": dict(lags=4), "2 lags": dict(lags=2), "6 lags": dict(lags=6), "8 lags": dict(lags=8),
            "90% band": dict(lags=4, level="90"), "different seed": dict(lags=4, seed=777)}
out = {}
for name, v in variants.items():
    r = oes.run("svar", data, dict(base, boot=200, **v))
    g = lambda var, h: [x for x in rows(r, var) if x[1] == h][0]
    out[name] = [g("gdp_growth", 0)[2], g("unemployment", 4)[2], g("unemployment", 4)[3], g("unemployment", 4)[4]]
print(pd.DataFrame(out, index=["GDP growth impact", "unemployment h=4", "lower", "upper"]).T.round(3))
```

<div class="read" markdown="1">
**How to read it**

The impact response of GDP growth is almost unchanged (2.62 to 2.69) with two, four, six or eight lags, and the response of unemployment after one year is between -0.25 and -0.35. The upper end of the 95% band is just below zero with four and eight lags and just above it with two and six, so the borderline significance of that response depends on the lag length; the 90% band excludes zero (-0.44 to -0.04). Changing the bootstrap seed moves the band limits in the second decimal place only.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

A supply shock identified with the long-run restriction of Blanchard and Quah (1989) raises GDP growth by 2.69 percentage points (annualised) on impact and 0.42 points after four quarters, with a permanent effect on the level of output of about 1%. Unemployment falls by about 0.3 percentage points after one to two years, a response that is borderline significant at the 5% level (95% band -0.46 to 0.05 after four quarters, -0.47 to 0.00 after eight) and fades within five years. The supply shock explains 80% of the forecast-error variance of GDP growth and 38% of that of unemployment at a five-year horizon. A demand-side shock raises unemployment by 0.45 points after one year with no permanent effect on output. The shape of the responses is similar under recursive identifications with either ordering, which differ mainly in whether unemployment shocks move growth within the quarter (0 or -1.59 against -1.02 here). The results are stable across two to eight lags (impact on growth 2.62 to 2.69).
</div>

| Report | What | Where |
|---|---|---|
| Model | Variables, lags, identification, bootstrap | Step 1 |
| Data | Stationarity checks | Step 2 |
| Specification | Lag selection, stability | Step 3 |
| Estimates | Impulse responses, variance shares, long-run matrix | Step 4 |
| Sensitivity | Recursive orderings against the long-run scheme | Step 5 |
| Instrument | First-stage strength, responses | Step 6 |
| Robustness | Lags, band level, seed | Step 7 |

**References.** Blanchard, O. and Quah, D. (1989), "The Dynamic Effects of Aggregate Demand and Supply Disturbances," *American Economic Review* 79. Gertler, M. and Karadi, P. (2015), "Monetary Policy Surprises, Credit Costs, and Economic Activity," *American Economic Journal: Macroeconomics* 7. Jarociński, M. and Karadi, P. (2020), "Deconstructing Monetary Policy Surprises," *American Economic Journal: Macroeconomics* 12. Mertens, K. and Ravn, M. (2013), "The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States," *American Economic Review* 103. Stock, J. and Watson, M. (2012), "Disentangling the Channels of the 2007–09 Recession," *Brookings Papers on Economic Activity*. Sims, C. (1980), "Macroeconomics and Reality," *Econometrica* 48.
