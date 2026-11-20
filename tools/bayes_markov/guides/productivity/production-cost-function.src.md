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
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the cost function

```python
@@EST@@
```

@@READ_EST@@

## Impose homogeneity and test the form

```python
@@TL@@
```

@@READ_TL@@

## Robustness: size groups and outliers

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: do the estimates recover known elasticities?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Nerlove (1963), *Estimation and Identification of Cobb-Douglas Production Functions*; Christensen and Greene (1976), *Journal of Political Economy*; Christensen, Jorgenson and Lau (1973), *Review of Economics and Statistics*; Marschak and Andrews (1944), *Econometrica*; Greene (2018), *Econometric Analysis*.
