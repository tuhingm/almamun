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
@@LOAD@@
```

@@READ_LOAD@@

## Compute the index

```python
@@EST@@
```

@@READ_EST@@

## By period and by state

```python
@@UNITS@@
```

@@READ_UNITS@@

## Robustness: orientation and inputs

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: known shifts and catch-up

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Caves, Christensen and Diewert (1982), *Econometrica*; Fare, Grosskopf, Norris and Zhang (1994), *American Economic Review*; Charnes, Cooper and Rhodes (1978), *European Journal of Operational Research*; Munnell (1990), *New England Economic Review*.
