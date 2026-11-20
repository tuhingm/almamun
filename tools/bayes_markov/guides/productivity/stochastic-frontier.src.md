title: Stochastic Frontier Analysis
short: Stochastic frontier
when: You have inputs and output for many firms, farms or plants and want to measure how far each is below the best practice, separating inefficiency from noise
data: Indonesian rice farms, 1,026 farm-year observations on 171 farms in six villages (Erwidodo 1990; Ecdat, RiceFarms); output is gross rice production in kilograms
tools: Python · OES engine · SciPy
oes: sfa
lead: Use this guide to measure technical efficiency. A production frontier gives the most output that the inputs can produce; the stochastic frontier model of Aigner, Lovell and Schmidt (1977) and Meeusen and van den Broeck (1977) lets actual output fall below it by a one-sided inefficiency term, on top of ordinary noise. The guide covers the model, the maximum likelihood fit, the tests of whether there is any inefficiency, the efficiency scores, the choice of distribution, robustness, a simulation check and the write-up.
---

## The question and the model

**How far is each farm below the best practice, and how much of the gap is inefficiency rather than bad luck?** Ordinary regression of output on inputs gives the *average* technology. A frontier gives the *best*: the output that a farm could reach with its inputs. The stochastic frontier model writes, in logarithms,

\[
\ln y_i = \beta_0 + \sum_j \beta_j \ln x_{ij} + v_i - u_i, \qquad v_i \sim N(0,\sigma_v^2),\quad u_i \ge 0 .
\]

The term \(v_i\) is noise (weather, measurement error); \(u_i\) is inefficiency, which can only lower output. The two are told apart by the shape of the error: the sum \(v - u\) is skewed to the left when there is inefficiency. The common choices for \(u\) are a half-normal, an exponential or a truncated normal distribution. The efficiency of farm \(i\) is \(TE_i = e^{-u_i}\), the ratio of its output to the frontier output; it is estimated by its expectation given the composed residual (Jondrow et al. 1982; Battese and Coelli 1988).

<div class="write" markdown="1">
**How to write it: the model**

We estimate a Cobb-Douglas stochastic production frontier for rice farms: the logarithm of gross output on the logarithms of land (hectares), seed, urea and total labour, with a half-normal inefficiency term, by maximum likelihood. We test for inefficiency with the skewness of the OLS residuals and a likelihood-ratio test against OLS, test constant returns to scale, compute the technical efficiency of each observation, and check the results against exponential and truncated-normal inefficiency, an extra input and a simulation.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the frontier

```python
@@EST@@
```

@@READ_EST@@

## Technical efficiency

```python
@@EFF@@
```

@@READ_EFF@@

## The distribution of inefficiency

```python
@@DIST@@
```

@@READ_DIST@@

## Robustness: another input and the sample

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: does the estimator recover a known frontier?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Aigner, Lovell and Schmidt (1977), *Journal of Econometrics*; Meeusen and van den Broeck (1977), *International Economic Review*; Jondrow, Lovell, Materov and Schmidt (1982), *Journal of Econometrics*; Battese and Coelli (1988), *Journal of Econometrics*; Coelli (1995), *Journal of Productivity Analysis*; Kumbhakar and Lovell (2000), *Stochastic Frontier Analysis*; Erwidodo (1990), Ecdat RiceFarms.
