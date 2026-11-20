title: Production and Cost Functions
short: Production and cost functions
when: You want the output elasticities, returns to scale or cost structure of an industry from data on inputs, output or costs
data: Total cost, output and the prices of labour, capital and fuel for 158 U.S. electric utilities in 1970 (Nerlove 1963; Christensen and Greene 1976; Ecdat, Electricity1970)
tools: Python · OES engine · NumPy
oes: prodfun
lead: Use this guide to estimate a cost function and read the returns to scale from it. The guide fits the log-linear (Cobb-Douglas) cost function of Nerlove (1963), tests the homogeneity of degree one in input prices that cost theory requires, imposes it, compares the Cobb-Douglas form with the translog form of Christensen, Jorgenson and Lau (1973), and shows the same machinery for a production function. It covers the model, the estimates, the tests, robustness, a simulation and the write-up.
---

## The question and the model

**Does a larger firm produce at lower unit cost, and how do costs respond to input prices?** The Cobb-Douglas cost function in logarithms is

\[
\ln C = \gamma_0 + \gamma_y \ln y + \gamma_L \ln p_L + \gamma_K \ln p_K + \gamma_F \ln p_F + \varepsilon .
\]

Cost theory requires that doubling all input prices doubles cost, so \(\gamma_L + \gamma_K + \gamma_F = 1\); this is imposed by dividing cost and two of the prices by the third. The cost elasticity of output is \(\gamma_y\), and returns to scale are \(1/\gamma_y\): a value above 1 means that unit cost falls with scale. The translog form adds squares and cross-products of the logarithms; with the variables centred at their means the first-order terms are the elasticities at the mean. For a production function, \(\ln y\) on the logarithms of the inputs, the returns to scale are the sum of the input elasticities.

<div class="write" markdown="1">
**How to write it: the model**

We estimate the log cost function of 158 electric utilities on the logarithms of output and of the prices of labour, capital and fuel, by OLS with robust standard errors. We test that the price elasticities sum to one, impose it by dividing by the price of fuel, compute the cost elasticity of output and the returns to scale, and test the Cobb-Douglas form against the translog.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("electricity1970.csv")
print(len(d), "utilities")
print(d[["cost", "output", "labor", "capital", "fuel"]].describe().round(2).loc[["mean", "min", "max"]])
print("shares of cost: labour", round(d.laborshare.mean(), 3), ", capital", round(d.capitalshare.mean(), 3), ", fuel", round(d.fuelshare.mean(), 3))
print("average cost per unit of output: smallest fifth", round((d.cost / d.output)[d.output <= d.output.quantile(.2)].mean(), 4), ", largest fifth", round((d.cost / d.output)[d.output >= d.output.quantile(.8)].mean(), 4))
```

<div class="read" markdown="1">
**How to read it**

The sample has 158 utilities in 1970. Output ranges from 4 to 115,500 (millions of kilowatt-hours) and cost from 0.13 to 737 (millions of dollars). Fuel is 63% of cost on average, capital 23% and labour 14%. Average cost per unit of output is 0.0108 for the smallest fifth of utilities and 0.0051 for the largest fifth: costs per unit fall with size, which is the pattern that the returns to scale measure.
</div>

## Estimate the cost function

```python
spec = dict(y="cost", output="output", x=["labor", "capital", "fuel"], kind="cost", form="cobb-douglas")
r = oes.run("prodfun", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Cobb-Douglas cost function")
oes.show(r, "Elasticities and tests")
```

<div class="read" markdown="1">
**How to read it**

In the log-linear cost function, the elasticity of cost with respect to output is 0.838 (robust standard error 0.019), so a 1% increase in output raises cost by 0.84%. Returns to scale, 1 over this, are 1.19: scale economies, and the hypothesis of constant returns is rejected at any conventional level. The price elasticities are 0.04 for labour, 0.19 for capital and 0.71 for fuel; they should be the cost shares, and they are the right order of magnitude (14%, 23%, 63%). Their sum is 0.946, and cost theory requires 1; the Wald test does not reject (p = 0.70), so homogeneity is consistent with the data. The R-squared is 0.982.
</div>

## Impose homogeneity and test the form

```python
rn = oes.run("prodfun", d.to_csv(index=False), dict(spec, normalise="fuel"))
oes.show(rn, "Cobb-Douglas cost function")
oes.show(rn, "Elasticities and tests")
rt = oes.run("prodfun", d.to_csv(index=False), dict(spec, normalise="fuel", form="translog"))
oes.show(rt, "Translog")
oes.show(rt, "Elasticities and tests")
tl = {row[0]: row[1] for row in rt["tables"][0]["rows"]}
ly = np.log(d.output); gm = np.exp(ly.mean())
for q in (0.1, 0.5, 0.9):
    o = d.output.quantile(q)
    e = tl["ln output"] + tl["1/2 ln output^2"] * (np.log(o) - ly.mean())
    print(f"output {o:,.0f} (quantile {q}): cost elasticity at mean prices {e:.3f}, returns to scale {1 / e:.3f}")
print("output at which the cost elasticity reaches 1:", round(gm * np.exp((1 - tl["ln output"]) / tl["1/2 ln output^2"])), " (largest in the sample:", int(d.output.max()), ")")
```

<div class="read" markdown="1">
**How to read it**

Imposing homogeneity (dividing cost and the prices of labour and capital by the price of fuel) leaves the output elasticity at 0.838 (returns 1.194) and the price coefficients are 0.073 for labour and 0.211 for capital, so the implied fuel elasticity is 0.717. The translog form changes the picture. The Wald test of all six second-order terms rejects the Cobb-Douglas form (144.1, p < 0.001), and the square of log output has a coefficient of 0.060 (t = 10.7): the cost elasticity rises with output. At the mean it is 0.904. At the 10th percentile of output (376) it is 0.764 (returns 1.31), at the median (5,646) 0.927 (1.08) and at the 90th percentile (26,700) 1.020 (0.98); it reaches 1 at an output of about 19,100. Scale economies are exhausted at the upper end of the distribution. The log-linear function averages the economies of large and small firms into one number, 0.84, which describes none of them well.
</div>

## Robustness: size groups and outliers

```python
d["grp"] = pd.qcut(d.output, 5, labels=False)
for g, s in d.groupby("grp"):
    rs = oes.run("prodfun", s.to_csv(index=False), dict(spec, normalise="fuel"))
    print(f"output group {g + 1} (median output {s.output.median():.0f}): n = {rs['nobs']}, cost elasticity {rs['tables'][0]['rows'][1][1]:.3f}, returns to scale {rs['summary'][2]['value']}")
big = d[d.output < d.output.max()]
rb = oes.run("prodfun", big.to_csv(index=False), dict(spec, normalise="fuel"))
print("without the largest utility: cost elasticity", rb["tables"][0]["rows"][1][1], ", returns to scale", rb["summary"][2]["value"])
```

<div class="read" markdown="1">
**How to read it**

Splitting the sample into five size groups gives cost elasticities of 0.686, 0.847, 0.997, 0.920 and 1.001 from the smallest to the largest, with returns to scale of 1.46, 1.18, 1.00, 1.09 and 1.00: strong economies of scale for small utilities and roughly constant returns for the larger half, in line with the findings of Christensen and Greene (1976) for this industry. The group estimates are noisy (about 31 observations each) and the fourth group's 0.92 breaks the monotone pattern. Dropping the largest utility changes little (0.834, returns 1.199), so the result does not depend on one point.
</div>

## Diagnostics: do the estimates recover known elasticities?

```python
rng = np.random.default_rng(0)
n = 500
y = rng.uniform(1, 5, n); pl, pk, pf = rng.uniform(1, 3, (3, n))
for gy in (1.0, 0.8):
    lc = 0.5 + gy * np.log(y) + 0.3 * np.log(pl) + 0.2 * np.log(pk) + 0.5 * np.log(pf) + rng.normal(0, 0.15, n)
    s = pd.DataFrame(dict(cost=np.exp(lc), y=y, pl=pl, pk=pk, pf=pf))
    rs = oes.run("prodfun", s.to_csv(index=False), dict(y="cost", output="y", x=["pl", "pk", "pf"], kind="cost", normalise="pf"))
    e = {row[0]: row for row in rs["tables"][1]["rows"]}
    k = [row for row in rs["tables"][1]["rows"] if row[0].startswith("Cost elasticity")][0]
    print(f"true cost elasticity {gy}: estimated {k[1]:.3f}; p-value for constant returns {k[2]}; returns to scale {1 / k[1]:.3f}")
# homogeneity violated: prices enter with weights summing to 1.3
lc = 0.5 + 1.0 * np.log(y) + 0.5 * np.log(pl) + 0.3 * np.log(pk) + 0.5 * np.log(pf) + rng.normal(0, 0.15, n)
s = pd.DataFrame(dict(cost=np.exp(lc), y=y, pl=pl, pk=pk, pf=pf))
rs = oes.run("prodfun", s.to_csv(index=False), dict(y="cost", output="y", x=["pl", "pk", "pf"], kind="cost"))
h = [row for row in rs["tables"][1]["rows"] if row[0].startswith("Homogeneity")][0]
print("price elasticities sum to 1.3: estimate", h[1], ", p =", h[2])
```

<div class="read" markdown="1">
**How to read it**

We simulate cost with elasticities of 0.3, 0.2 and 0.5 for the three prices and an output elasticity of 1.0 (500 firms). The estimate is 0.985 and constant returns are not rejected (p = 0.34). With a true output elasticity of 0.8 the estimate is 0.803, returns to scale are 1.245 and constant returns are rejected. When the price elasticities in the data sum to 1.3, the unrestricted regression gives 1.338 and the homogeneity test rejects (p < 0.001); the cost-function form that divides by one price would be misspecified in that case. The test suite compares the coefficients and robust standard errors with statsmodels.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We estimate the cost function of 158 U.S. electric utilities in 1970. In the log-linear form with homogeneity in input prices imposed, the cost elasticity of output is 0.84 (s.e. 0.02), so returns to scale are 1.19 and constant returns are rejected; homogeneity of degree one in prices is not rejected (p = 0.70). The log-linear form is rejected against the translog (Wald 144.1, p < 0.001): the cost elasticity rises with output, from 0.76 at the 10th percentile of output to 0.93 at the median and 1.02 at the 90th, so economies of scale are exhausted at an output of about 19,000 million kilowatt-hours. By size group, the elasticity is 0.69 for the smallest fifth and about 1.0 for the largest. The estimates treat input prices as exogenous to each utility, which is a reasonable assumption for a regulated industry with competitive input markets, but the result describes cost at a point in time and not the effect of merging utilities.
</div>

**References.** Nerlove (1963), *Estimation and Identification of Cobb-Douglas Production Functions*; Christensen and Greene (1976), *Journal of Political Economy*; Christensen, Jorgenson and Lau (1973), *Review of Economics and Statistics*; Marschak and Andrews (1944), *Econometrica*; Greene (2018), *Econometric Analysis*.
