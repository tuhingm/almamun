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
@@LOAD@@
```

@@READ_LOAD@@

## Score the units

```python
@@EST@@
```

@@READ_EST@@

## Peers and super-efficiency

```python
@@PEER@@
```

@@READ_PEER@@

## Robustness: orientation, inputs and outliers

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: bias in small samples and noise

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Charnes, Cooper and Rhodes (1978), *European Journal of Operational Research*; Banker, Charnes and Cooper (1984), *Management Science*; Andersen and Petersen (1993), *Management Science*; Simar and Wilson (1998), *Management Science*; Munnell (1990), *New England Economic Review*.
