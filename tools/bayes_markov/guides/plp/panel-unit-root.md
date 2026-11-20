title: Panel Unit-Root and Stationarity Tests
short: Panel unit roots
when: You have a panel of long series, such as inflation or output across countries, and need to know whether the variable is stationary before choosing between levels, differences or a cointegration model
data: Annual CPI inflation for 34 countries, 1990-2023 (World Bank WDI), and log gross state product for 48 US states, 1970-86 (Munnell 1990)
tools: Python · OES engine · statsmodels
oes: punitroot
lead: Use this guide when the order of integration of a panel variable decides the next step. Panel unit-root tests pool the information in many short or medium series and are more powerful than testing each series alone, but they answer different questions, and cross-section dependence can mislead them. The guide runs the Levin-Lin-Chu type, Im-Pesaran-Shin, Fisher-type and Hadri tests on a country inflation panel, shows how to read them together with the unit-by-unit results, and then repeats the analysis on a trending panel where the tests disagree for a reason you can name.
---

## The question and the tests

**Do the series in this panel have unit roots?** Each unit has the augmented Dickey-Fuller regression

\[
\Delta y_{it} = \alpha_i + \rho_i\, y_{i,t-1} + \sum_{j=1}^{p_i} \phi_{ij}\, \Delta y_{i,t-j} + e_{it},
\]

and the null of a unit root is \(\rho_i = 0\). The tests differ in the alternative. **Levin-Lin-Chu** assumes one common \(\rho\) for all units and pools the regressions: it is powerful if the series are alike, and says little if some are stationary and some are not. **Im-Pesaran-Shin** averages the individual \(t\) statistics (the \(\bar t\) statistic) and allows a different \(\rho_i\) for every unit. **Fisher-type** tests combine the individual \(p\)-values, \(P=-2\sum \ln p_i\), which is chi-squared with \(2N\) degrees of freedom under the null of independence. All three have a unit root in every series as the null. **Hadri** reverses the null: all series are stationary.

In this engine the null distributions of the Levin-Lin-Chu type and IPS statistics are simulated for the panel's own number of units, series length and lag selection rule, rather than taken from published tables. They assume independence across units.

<div class="read" markdown="1">
**Three checks before you trust a panel unit-root test**

1. **What does the alternative say?** A rejection by IPS or Fisher means that at least one unit is stationary, not that all are. Look at the unit-by-unit table.
2. **Is there a common shock?** Global inflation or a world business cycle makes the series dependent, and the tests then reject too often. Re-run with cross-sectional means removed.
3. **Are the deterministic terms right?** A trending variable tested with a constant only will look like a unit root, and a driftless one tested with a trend loses power.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We test for unit roots in annual CPI inflation in 34 countries, 1990-2023, using four panel tests: the Levin-Lin-Chu type test with a common autoregressive coefficient, the Im-Pesaran-Shin test, the Fisher-type tests that combine the individual Dickey-Fuller p-values, and the Hadri test of the null of stationarity. Individual regressions include a constant, with up to two lagged differences chosen by the AIC. Critical values for the Levin-Lin-Chu type and Im-Pesaran-Shin tests are simulated for the length of the panel and the lag-selection rule. As a robustness check we remove the cross-sectional mean in every year, which removes a common time effect.
</div>

## Load the data and look at the series

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("inflation_panel.csv").read()
df = pd.read_csv("inflation_panel.csv")
print(f"{df.country.nunique()} countries, {df.year.min()}-{df.year.max()}, observations per country: {sorted(int(n) for n in df.groupby('country').size().unique())}")
g = df.groupby("country").inflation
print(pd.DataFrame({"mean": g.mean(), "sd": g.std(), "max": g.max()}).round(1).sort_values("mean", ascending=False).head(6))
wide = df.pivot(index="year", columns="country", values="inflation")
print("average pairwise correlation of the series:", round((wide.corr().values.sum() - len(wide.columns)) / (len(wide.columns) ** 2 - len(wide.columns)), 2))
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 34 countries with 34 annual observations each. The highest mean inflation rates are in Egypt (10.7%), Colombia (10.3%), Hungary (10.3%) and Mexico (9.4%), and Indonesia has the largest single value (58.5%, the 1998 crisis). The average pairwise correlation of the country series is 0.51, which is high: inflation across countries moves together with global shocks such as the commodity booms of 2008 and 2022. The tests below assume independence across units, so this is the first warning that their p-values may be too small.
</div>

## Run the panel tests

```python
spec = dict(y="inflation", unit="country", time="year", trend="c", ic="aic", maxlag=2, sims=300)
r = oes.run("punitroot", data, spec)
oes.summary(r)
oes.show(r, "Panel unit-root")
```

<div class="read" markdown="1">
**How to read it**

Every test that has a unit root as its null rejects it at p < 0.001: the Levin-Lin-Chu type statistic is -8.32 after standardisation, the IPS t-bar is -3.19 (W = -9.64), and the Fisher statistics are P = 284.1 (68 degrees of freedom) and Z = -11.8. In 67.6% of countries (23 of 34) the individual test also rejects at 5%. The Hadri test points the other way: its null of stationarity is rejected (LM = 1.385, against 1/6 expected under the null; z = 47.7). The two sets of tests are not contradictory, but they say that inflation is neither a clean unit root nor a clean stationary process. The Hadri statistic used here has no correction for serial correlation and is known to over-reject when series are persistent or have shifts in the mean, which both apply to inflation over 1990-2023. Treat its rejection as a sign of persistence rather than as evidence of a unit root.
</div>

## Which units drive the result?

```python
oes.show(r, "Unit-by-unit")
tab = pd.DataFrame(r["tables"][1]["rows"], columns=r["tables"][1]["columns"])
print("units that do not reject a unit root at 5%:", tab[tab["5% decision"] == "unit root"]["Unit"].tolist())
print("lags chosen:", tab["Lags"].value_counts().sort_index().to_dict())
```

<div class="read" markdown="1">
**How to read it**

Eleven countries do not reject a unit root at 5%: Austria, Colombia, Germany, Egypt, France, Hungary, India, Israel, Italy, Korea and Mexico. Three of them are close to the line (India p = 0.058, Israel 0.061, Korea 0.072), and two countries that do reject are also near it (Japan 0.043, South Africa 0.047). With 34 observations an individual test has little power, so a non-rejection is not evidence that the series has a unit root, and with 34 tests about one or two rejections at 5% would occur by chance even if all series had unit roots. The lag selection picks no lagged differences for 22 countries, one for 7 and two for 5. The panel tests are informative precisely because they use all 34 series at once.
</div>

## Common shocks and deterministic terms

```python
variants = {"baseline": {}, "remove cross-section means": dict(demean="yes"), "constant and trend": dict(trend="ct"), "BIC lags": dict(ic="bic"), "no lags": dict(maxlag=0)}
out = {}
for name, v in variants.items():
    rr = oes.run("punitroot", data, dict(spec, **v))
    t = {x[0]: x for x in rr["tables"][0]["rows"]}
    out[name] = [t["Levin-Lin-Chu type (common unit root)"][3], t["Im-Pesaran-Shin t-bar (W)"][3], t["Fisher-type, inverse normal (Z)"][3],
                 t["Hadri LM (stationarity)"][3], [s["value"] for s in rr["summary"] if s["label"].startswith("Share")][0]]
print(pd.DataFrame(out, index=["LLC p", "IPS p", "Fisher Z p", "Hadri p", "share rejecting"]).T.round(4))
```

<div class="read" markdown="1">
**How to read it**

The panel conclusions survive every variant: all four unit-root tests reject at the 1% level, with the smallest margin for the constant-and-trend specification (Levin-Lin-Chu p = 0.0004, IPS p = 0.0001). What changes is how many countries reject individually: 67.6% in the baseline, 41.2% (14 countries) after the cross-sectional mean is removed, 41.2% with a trend, 70.6% with the BIC and 64.7% with no lagged differences. A large share of the individual rejections therefore depends on the common component of inflation. The Hadri test rejects stationarity in every variant.
</div>

## A trending panel where the tests disagree

```python
prod = open("produc.csv").read()
spec2 = dict(y="log_gsp", unit="state", time="year", trend="ct", maxlag=2, sims=300)
r2 = oes.run("punitroot", prod, spec2)
oes.show(r2, "Panel unit-root")
print(r2["notes"][0])
r3 = oes.run("punitroot", prod, dict(spec2, demean="yes"))
oes.show(r3, "Panel unit-root")
```

<div class="read" markdown="1">
**How to read it**

Log gross state product behaves as the series that it is. With a constant and trend, the Levin-Lin-Chu type test (p = 0.70), the IPS test (p = 0.90) and the inverse-normal Fisher test (p = 0.25) do not reject a unit root, and the Hadri test rejects trend stationarity (z = 28.2). The inverse chi-squared Fisher test is the exception (P = 154.1, p = 0.0002), and the note printed below the table explains why it should not be trusted here: with only 17 years, the asymptotic MacKinnon p-values that the Fisher tests combine are too small, whereas the IPS and Levin-Lin-Chu type tests use null distributions simulated for T = 17. After the cross-sectional means are removed none of the unit-root tests rejects (p between 0.92 and 1.00) and Hadri still rejects. The practical lesson is that a disagreement among panel tests is often a sign that one of them is outside its valid range, so check the sample length and the notes before choosing a test.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Panel unit-root tests reject a unit root in annual inflation for 34 countries over 1990-2023: the Levin-Lin-Chu type test (z = -8.3), the Im-Pesaran-Shin test (W = -9.6) and the Fisher-type tests (inverse normal Z = -11.8) all have p-values below 0.001, with simulated critical values and individual regressions that include a constant and up to two lags chosen by the AIC. The conclusions hold with a trend, with BIC lag selection and after removing cross-sectional means. The rejection means that at least some countries have stationary inflation: at the 5% level the individual tests reject a unit root for 23 of 34 countries (14 of 34 after removing cross-sectional means). The Hadri test rejects stationarity (p < 0.001), which we attribute to persistence and mean shifts, since the statistic is not corrected for serial correlation. All tests assume independence across countries, and the average pairwise correlation of the series is 0.51, so the p-values should be read as indicative. In the log-GSP panel, by contrast, no unit-root test that is valid for T = 17 rejects, and we treat that variable as non-stationary.
</div>

| Report | What | Where |
|---|---|---|
| Model | Tests, deterministic terms, lag selection, simulated critical values | Step 1 |
| Data | Persistence and cross-section correlation | Step 2 |
| Panel tests | Four tests and their different nulls | Step 3 |
| Units | Which series reject, lags chosen | Step 4 |
| Robustness | Common shocks, trend, lag rule | Step 5 |
| Short panels | Why the tests can disagree | Step 6 |

**References.** Levin, A., Lin, C.-F. and Chu, C.-S. J. (2002), "Unit Root Tests in Panel Data: Asymptotic and Finite-Sample Properties," *Journal of Econometrics* 108. Im, K. S., Pesaran, M. H. and Shin, Y. (2003), "Testing for Unit Roots in Heterogeneous Panels," *Journal of Econometrics* 115. Maddala, G. S. and Wu, S. (1999), "A Comparative Study of Unit Root Tests with Panel Data and a New Simple Test," *Oxford Bulletin of Economics and Statistics* 61. Choi, I. (2001), "Unit Root Tests for Panel Data," *Journal of International Money and Finance* 20. Hadri, K. (2000), "Testing for Stationarity in Heterogeneous Panel Data," *Econometrics Journal* 3. Munnell, A. (1990), "How Does Public Infrastructure Affect Regional Economic Performance?" *New England Economic Review*.
