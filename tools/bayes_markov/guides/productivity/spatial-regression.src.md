title: Spatial Regression: Moran's I, Spatial Lag and Spatial Error
short: Spatial regression
when: Your observations are places (neighbourhoods, regions, countries) and you suspect that nearby places are alike or that outcomes spill over from one to the next
data: Crime, household income and house value in 49 neighbourhoods of Columbus, Ohio, 1980, with the coordinates of the neighbourhood centres (Anselin 1988)
tools: Python · OES engine · SciPy
oes: spatial_cs
lead: Use this guide to test for spatial dependence and to estimate spatial regression models. It follows Anselin (1988): the OLS regression, Moran's I and the Lagrange multiplier tests on its residuals, the spatial lag model (SAR) and the spatial error model (SEM) by maximum likelihood, the choice between them, and the direct and indirect effects of a spatial lag. It covers the weights matrix, robustness to the number of neighbours, a simulation with known spatial parameters and the write-up.
---

## The question and the model

**Do nearby places resemble each other, and why?** Ordinary regression assumes that errors are independent across places. Two models relax this. The *spatial error* model puts the dependence in the unobserved factors:

\[
y = X\beta + u, \qquad u = \lambda W u + \varepsilon .
\]

The *spatial lag* model puts it in the outcome itself, so that the outcome of a place depends on those of its neighbours:

\[
y = \rho W y + X\beta + \varepsilon .
\]

\(W\) is a weights matrix with \(w_{ij} > 0\) if \(j\) is a neighbour of \(i\), row-standardised so that each row sums to one. Here the neighbours of a place are its \(k\) nearest places by the distance between centres. Both models are estimated by maximum likelihood, with the Jacobian \(\ln|I - \rho W|\) from the eigenvalues of \(W\). In the lag model \(\beta_k\) is not the marginal effect, because a change in \(x\) in one place feeds through the neighbours; the *direct*, *indirect* and *total* effects are averages of the matrix \((I - \rho W)^{-1}\beta_k\) (LeSage and Pace 2009).

<div class="write" markdown="1">
**How to write it: the model**

We regress the crime rate of the 49 Columbus neighbourhoods on household income and house value by OLS, test the residuals for spatial dependence with Moran's I and the Lagrange multiplier tests for a spatial lag and a spatial error, using a weights matrix of the five nearest neighbours, and estimate the spatial lag and spatial error models by maximum likelihood. We compare the three models by the log-likelihood, AIC and BIC, report the direct and indirect effects for the lag model, and check the results against the number of neighbours and a simulation.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## OLS and the tests for spatial dependence

```python
@@EST@@
```

@@READ_EST@@

## Spatial lag and spatial error models

```python
@@MOD@@
```

@@READ_MOD@@

## Robustness: the weights matrix

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: do the tests and estimates recover known dependence?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Anselin (1988), *Spatial Econometrics: Methods and Models*; Anselin, Bera, Florax and Yoon (1996), *Regional Science and Urban Economics*; LeSage and Pace (2009), *Introduction to Spatial Econometrics*; Moran (1950), *Biometrika*; Cliff and Ord (1981), *Spatial Processes*.
