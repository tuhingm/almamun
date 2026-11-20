title: Malmquist Productivity Index
short: Malmquist index
when: You have a panel of units with inputs and outputs over several periods and want productivity growth split into catch-up to the frontier and a shift of the frontier
data: Output and inputs of the 48 contiguous U.S. states, 1970 to 1986, 816 state-years (Munnell 1990; Ecdat, Produc); output is gross state product, inputs are private capital and employment
tools: Python · OES engine · SciPy (linear programming)
oes: malmquist
lead: Use this guide to measure total factor productivity growth for a panel of units without a production function. The Malmquist index (Caves, Christensen and Diewert 1982) compares each unit's input-output bundle with the best-practice frontier of two adjacent periods; the decomposition of Fare, Grosskopf, Norris and Zhang (1994) separates catch-up (efficiency change) from the movement of the frontier (technical change). The guide covers the index, the estimates by period and by state, robustness, a simulation with known answers and the write-up.
---

## The question and the model

**Did productivity grow, and was it because laggards caught up with the best or because the best got better?** With distance functions \(D^t(x,y)\) measured against the DEA frontier of period \(t\), the input-oriented Malmquist index between periods \(t\) and \(t+1\) is

\[
M = \underbrace{\frac{D^{t+1}(x_{t+1},y_{t+1})}{D^{t}(x_{t},y_{t})}}_{\text{efficiency change}} \times \underbrace{\left[\frac{D^{t}(x_{t+1},y_{t+1})}{D^{t+1}(x_{t+1},y_{t+1})}\,\frac{D^{t}(x_{t},y_{t})}{D^{t+1}(x_{t},y_{t})}\right]^{1/2}}_{\text{technical change}} .
\]

A value above 1 is an improvement. Efficiency change measures movement toward the frontier; technical change measures the shift of the frontier at the unit's own input and output mix. The frontiers here are constant-returns DEA frontiers for each period, as in the standard application.

<div class="write" markdown="1">
**How to write it: the model**

We compute the input-oriented Malmquist productivity index for the 48 states in each pair of adjacent years from 1970 to 1986, from constant-returns DEA frontiers with gross state product as the output and private capital and employment as inputs. We report the geometric mean over states of total factor productivity change and its two components for each period and over the whole sample, list the fastest and slowest states, and check the results against orientation and input choices and against simulations with known answers.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("produc.csv")
print(len(d), "state-years:", d.state.nunique(), "states,", d.year.min(), "to", d.year.max())
g = d.groupby("year")[["gsp", "pc", "emp"]].sum()
print((g.loc[[1970, 1980, 1986]] / g.loc[1970]).round(3))
print("aggregate output per worker, 1970 and 1986:", (g.gsp / g.emp).loc[[1970, 1986]].round(2).tolist())
```

<div class="read" markdown="1">
**How to read it**

The panel is balanced: 816 state-years, 48 states from 1970 to 1986. Between 1970 and 1986 total output rose by 54%, private capital by 65% and employment by 41%, so output per worker rose from 33.65 to 36.81 (9%). Inputs grew nearly as fast as output; whether productivity grew depends on how the inputs are weighted, which is what the frontier does.
</div>

## Compute the index

```python
spec = dict(inputs=["pc", "emp"], outputs=["gsp"], id="state", time="year")
r = oes.run("malmquist", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "Cumulative change")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

Total factor productivity grew by 0.16% a period on average (geometric mean of 1.0016 over 16 year-pairs) and by 2.6% in all (1.0257). The components pull in opposite directions: efficiency change is +6.3% cumulative (1.0633; the states moved closer to the frontier) and technical change is -3.5% (0.9647; the frontier moved *inward*). The index is the product of the two, up to rounding. Technical change below 1 means that the best states at the end of the sample produced less from the same inputs than the best states at the start. The chart shows the three series by period.
</div>

## By period and by state

```python
oes.show(r, "Malmquist productivity index by period")
oes.show(r, "Units with the fastest")
```

<div class="read" markdown="1">
**How to read it**

Productivity change is cyclical. The index is below 1 in 1973 to 1974 (0.963), 1974 to 1975 (0.975), 1979 to 1980 (0.978) and 1981 to 1982 (0.982): the oil-shock and recession years, in which only 8%, 10%, 12.5% and 12.5% of states improved, and in which total output fell. It is highest in 1983 to 1984 (1.029), when 96% of states improved. Catch-up dominates in 1972 to 1973 (efficiency 1.025) and 1980 to 1981 (1.049), while the frontier shifts in 1983 to 1984 (technical change 1.023). By state, the fastest are New Hampshire (1.0140 a period), Tennessee (1.0125) and Maine (1.0098); the slowest are Wyoming (0.9820), Louisiana (0.9837) and Massachusetts (0.9932). Wyoming's average efficiency change is exactly 1, so its productivity change is almost all the movement of the frontier. Wyoming and Louisiana are the two states with the highest output per worker in 1986; a decline of the leaders is the technical regress.
</div>

## Robustness: orientation and inputs

```python
ro = oes.run("malmquist", d.to_csv(index=False), dict(spec, orientation="output"))
print("output orientation: mean TFP / efficiency / technical change", [s["value"] for s in ro["summary"][2:5]])
print("input orientation:  mean TFP / efficiency / technical change", [s["value"] for s in r["summary"][2:5]])
r3 = oes.run("malmquist", d.to_csv(index=False), dict(spec, inputs=["pc", "emp", "pcap", "util"]))
print("four inputs:", [s["value"] for s in r3["summary"][2:5]])
r1 = oes.run("malmquist", d[d.year >= 1976].to_csv(index=False), spec)
print("1976-86 only:", [s["value"] for s in r1["summary"][2:5]])
```

<div class="read" markdown="1">
**How to read it**

*Orientation.* Input and output orientation give identical numbers: with constant-returns frontiers the two distance functions are reciprocals of each other. *Inputs.* Adding public capital and the utilities capital stock as inputs gives a mean TFP change of 1.0028 (efficiency 1.0018, technical change 1.0010): the sign of the technical change flips, because the frontier is now spanned by different states. *Period.* From 1976 to 1986 the numbers are 1.0041, 1.0049 and 0.9992. The decomposition is sensitive to the specification, and the technical change, which is determined by a few states on the frontier, is the less reliable part; the total change is more stable (between 1.0016 and 1.0041).
</div>

## Diagnostics: known shifts and catch-up

```python
rng = np.random.default_rng(0)
n = 40; x = rng.uniform(1, 3, n); y0 = x * rng.uniform(0.5, 0.95, n)
def panel(y1):
    return pd.concat([pd.DataFrame(dict(id=range(n), t=1, x=x, y=y0)), pd.DataFrame(dict(id=range(n), t=2, x=x, y=y1))]).to_csv(index=False)
sp = dict(inputs=["x"], outputs=["y"], id="id", time="t")
a = oes.run("malmquist", panel(y0 * 1.2), sp)
print("technology improves 20% for every unit: TFP, efficiency, technical", [round(v, 4) for v in a["tables"][0]["rows"][0][1:4]])
best = (y0 / x).max()
b = oes.run("malmquist", panel(np.minimum(y0 * 1.15, best * x)), sp)
print("laggards improve 15%, frontier unchanged: TFP, efficiency, technical", [round(v, 4) for v in b["tables"][0]["rows"][0][1:4]])
c = oes.run("malmquist", panel(y0), sp)
print("nothing changes:", [round(v, 4) for v in c["tables"][0]["rows"][0][1:4]])
```

<div class="read" markdown="1">
**How to read it**

We build a two-period panel of 40 units with a known answer. When every unit's output rises by 20% at the same inputs, the index is 1.200, efficiency change 1.000 and technical change 1.200: a pure shift of the technology. When the laggards improve by 15% but are capped at the existing frontier, which does not move, the index is 1.124 with efficiency change 1.124 and technical change 1.000: pure catch-up. When nothing changes, all three are 1.000. The test suite checks the identity index = efficiency change x technical change in every period of the real data.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We compute the Malmquist productivity index for the 48 contiguous U.S. states for 1970 to 1986 (816 state-years) from constant-returns DEA frontiers, with gross state product as the output and private capital and employment as inputs. Productivity grew by 0.16% a period on average and by 2.6% over the 16 year-to-year changes. States moved closer to the frontier (efficiency change +6.3% in all) but the frontier itself moved inward (technical change -3.5%). The index is below 1 in the recession and oil-shock years 1973 to 1975, 1979 to 1980 and 1981 to 1982, and highest in 1983 to 1984 (1.029). New Hampshire, Tennessee and Maine are the fastest states and Wyoming, Louisiana and Massachusetts the slowest. The mean index is stable across specifications (1.0016 to 1.0041), but its decomposition is not: with two more inputs technical change is 1.001. The index has no standard error, so differences of this size should be read as descriptions of this sample.
</div>

**References.** Caves, Christensen and Diewert (1982), *Econometrica*; Fare, Grosskopf, Norris and Zhang (1994), *American Economic Review*; Charnes, Cooper and Rhodes (1978), *European Journal of Operational Research*; Munnell (1990), *New England Economic Review*.
