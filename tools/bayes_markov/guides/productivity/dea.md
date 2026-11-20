title: Data Envelopment Analysis
short: DEA
when: You have several inputs and outputs for a set of comparable units and want an efficiency score and a list of peers for each, without assuming a functional form
data: Output and inputs of the 48 contiguous U.S. states in 1986 (Munnell 1990; Ecdat, Produc); output is gross state product, inputs are private capital and employment
tools: Python · OES engine · SciPy (linear programming)
oes: dea
lead: Use this guide to score units against the best practice in the sample without a production function. Data envelopment analysis (Charnes, Cooper and Rhodes 1978; Banker, Charnes and Cooper 1984) solves a linear programme for each unit and finds the largest proportional cut in its inputs that the best units show to be possible. The guide covers the method, the scores under constant and variable returns, scale efficiency, peers, super-efficiency, robustness, a simulation of the small-sample bias and the write-up.
---

## The question and the model

**Which units are efficient, how far are the others from them, and which units should they copy?** For unit \(o\) with inputs \(x_o\) and outputs \(y_o\), the input-oriented CCR score is the smallest \(\theta\) such that a mix of the other units, with weights \(\lambda\), uses no more than \(\theta x_o\) of the inputs and produces at least \(y_o\):

\[
\min_{\theta,\lambda}\ \theta \quad \text{s.t.}\quad \sum_j \lambda_j x_j \le \theta x_o,\quad \sum_j \lambda_j y_j \ge y_o,\quad \lambda \ge 0 .
\]

Adding \(\sum_j \lambda_j = 1\) gives the BCC score (variable returns to scale). Scale efficiency is CCR divided by BCC. A score of 1 means that no combination of other units does better. The units with positive weights \(\lambda_j\) are the *peers* of unit \(o\). DEA has no error term: noise counts as inefficiency, and the frontier is built from the sample.

<div class="write" markdown="1">
**How to write it: the model**

We apply input-oriented DEA to the 48 contiguous U.S. states in 1986, with gross state product as the output and private capital and employment as the inputs. We compute CCR (constant returns) and BCC (variable returns) technical efficiency, scale efficiency and the returns to scale of each state, list the reference peers of the least efficient states, rank the efficient states by super-efficiency, and check the sensitivity of the results to the choice of orientation, of inputs and of outliers, and the small-sample bias in a simulation.
</div>

## Load the data

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

d = pd.read_csv("produc_1986.csv")
print(len(d), "states in", d.year.iloc[0])
print(d[["gsp", "pc", "emp"]].describe().round(0).loc[["mean", "min", "max"]])
print("output per worker (gsp/emp): mean", round((d.gsp / d.emp).mean(), 2), ", highest:", d.assign(ap=d.gsp / d.emp).nlargest(3, "ap")[["state", "ap"]].round(2).values.tolist())
```

<div class="read" markdown="1">
**How to read it**

There are 48 states in 1986. Gross state product averages 75,458 (millions of dollars), private capital 72,260 and employment 2,050 (thousands), and the largest state is more than 60 times the smallest in output. Output per worker averages 36.1 and is highest in Wyoming (55.4), Louisiana (45.1) and Texas (42.1). Wyoming is an extreme point and will shape the frontier.
</div>

## Score the units

```python
spec = dict(inputs=["pc", "emp"], outputs=["gsp"], label="state")
r = oes.run("dea", d.to_csv(index=False), spec)
oes.summary(r)
oes.show(r, "DEA efficiency scores")
oes.show(r, "Returns to scale")
oes.plot(r)
```

<div class="read" markdown="1">
**How to read it**

Mean technical efficiency under constant returns (CCR) is 0.862: on average a state could produce its output with 86% of its capital and labour, if it matched the best combination of the others. Four states are on the CCR frontier (8%). Under variable returns (BCC) the mean is 0.899 and nine states (19%) are efficient: states that are efficient only because no larger or smaller state is comparable are counted as efficient. The ratio, scale efficiency, has a mean of 0.960 and a minimum of 0.800. Returns to scale are increasing for 40 of the 48 states, constant for 4 and decreasing for 4: most states are smaller than their efficient scale. The histogram shows the CCR scores between 0.70 and 1.
</div>

## Peers and super-efficiency

```python
oes.show(r, "Least efficient")
oes.show(r, "Average potential")
rs = oes.run("dea", d.to_csv(index=False), dict(spec, super="yes"))
oes.show(rs, "Super-efficiency")
```

<div class="read" markdown="1">
**How to read it**

The least efficient state is South Carolina (CCR 0.703): it should be able to produce its output with 70% of its inputs by combining 0.17 of Wyoming and 0.08 of California. Alabama (0.777) and Mississippi (0.780) follow, with Wyoming as the main peer. The units with the largest gap between CCR and BCC are small states, such as Vermont (CCR 0.800, BCC 1.000), Delaware (scale 0.844) and South Dakota (0.841): they are efficient for their size, and the gap is scale. The potential saving is 10.8% of capital and 10.7% of employment in total, less than the 13.8% (1 - 0.862) that the mean score suggests, because the large states are the more efficient ones. Super-efficiency (the score when the unit is left out of its own reference set) ranks the four efficient states: Wyoming 1.227, Rhode Island 1.044, California 1.028 and New York 1.017; Wyoming is far above the others.
</div>

## Robustness: orientation, inputs and outliers

```python
ro = oes.run("dea", d.to_csv(index=False), dict(spec, orientation="output"))
print("output orientation, mean CCR / BCC / scale:", [round(row[1], 3) for row in ro["tables"][0]["rows"]])
print("input orientation,  mean CCR / BCC / scale:", [round(row[1], 3) for row in r["tables"][0]["rows"]])
r1 = oes.run("dea", d.to_csv(index=False), dict(spec, inputs=["emp"]))
print("employment only: mean CCR", r1["summary"][1]["value"], ", units on the frontier", r1["summary"][4]["value"])
r3 = oes.run("dea", d.to_csv(index=False), dict(spec, inputs=["pc", "emp", "pcap", "util"]))
print("four inputs: mean CCR", r3["summary"][1]["value"], ", units on the frontier", r3["summary"][4]["value"])
dw = d[d.state != "WYOMING"]
rw = oes.run("dea", dw.to_csv(index=False), spec)
print("without Wyoming: mean CCR", rw["summary"][1]["value"], ", units on the frontier", rw["summary"][4]["value"])
```

<div class="read" markdown="1">
**How to read it**

*Orientation.* Minimising inputs and maximising output give nearly the same mean scores (CCR 0.862 in both; BCC 0.899 against 0.894), as expected when returns to scale are close to constant. *Inputs.* With employment as the only input, the mean CCR score falls to 0.653 and only one state is on the frontier; adding public capital and utilities capital as inputs raises the mean to 0.893 and puts 8 states on the frontier. More inputs means more ways for a unit to look special: the scores rise with the number of variables and are not comparable across specifications. *Outliers.* Without Wyoming the mean is 0.870 and four states are still efficient, since California, New York and Rhode Island stay and Louisiana, which has the third highest output per worker, joins them.
</div>

## Diagnostics: bias in small samples and noise

```python
rng = np.random.default_rng(0)
def sim(n, noise):
    x = rng.uniform(1, 10, n); u = rng.beta(5, 1.5, n)          # true efficiency, mean about 0.77
    y = x ** 0.6 * u * np.exp(rng.normal(0, noise, n))
    return pd.DataFrame(dict(x=x, y=y)), u
for n in (20, 50, 200):
    est = []; tru = []
    for _ in range(15):
        s, u = sim(n, 0.0)
        e = oes.run("dea", s.to_csv(index=False), dict(inputs=["x"], outputs=["y"], orientation="output"))
        est.append(e["tables"][0]["rows"][1][1]); tru.append(u.mean())
    print(f"n = {n:3d}: mean BCC efficiency {np.mean(est):.3f}, true mean efficiency {np.mean(tru):.3f}")
for noise in (0.0, 0.15):
    s, u = sim(100, noise)
    e = oes.run("dea", s.to_csv(index=False), dict(inputs=["x"], outputs=["y"], orientation="output"))
    print(f"noise s.d. {noise}: BCC {e['tables'][0]['rows'][1][1]:.3f}, units on the CCR frontier {e['summary'][4]['value']} of 100")
```

<div class="read" markdown="1">
**How to read it**

With a known frontier (output = input^0.6 times a true efficiency with mean 0.77), the BCC estimate of mean efficiency is biased upward in small samples: 0.844 with 20 units, 0.810 with 50 and 0.788 with 200, against a true mean of about 0.767. The estimate of the frontier sits inside the true one, so the estimated distances are too small. The bias shrinks with the sample but slowly. With noise in output (standard deviation 0.15 in logarithms) the mean BCC score for 100 units falls from 0.802 to 0.686, because positive noise draws push the frontier out and negative ones are counted as inefficiency. Do not read a DEA score as pure inefficiency, and do not compare scores from samples of different sizes. The test suite checks the scores against hand-computed linear programmes.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

We measure the efficiency of the 48 contiguous U.S. states in 1986 with input-oriented data envelopment analysis (gross state product; private capital and employment as inputs). Mean technical efficiency is 0.86 under constant returns and 0.90 under variable returns; four states (Wyoming, California, New York and Rhode Island) are on the constant-returns frontier and nine on the variable-returns frontier. Mean scale efficiency is 0.96, and 40 of the 48 states operate under increasing returns to scale. South Carolina (0.70), Alabama (0.78) and Mississippi (0.78) are the least efficient, with Wyoming and California as their main peers. The results are similar for output orientation, but the scores depend on the inputs (the mean falls to 0.65 with employment alone) and on the extreme unit (Wyoming), and in simulations DEA overstates efficiency in small samples and counts noise as inefficiency, so the scores describe this sample and are not estimates of a population frontier.
</div>

**References.** Charnes, Cooper and Rhodes (1978), *European Journal of Operational Research*; Banker, Charnes and Cooper (1984), *Management Science*; Andersen and Petersen (1993), *Management Science*; Simar and Wilson (1998), *Management Science*; Munnell (1990), *New England Economic Review*.
