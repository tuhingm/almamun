title: Kernel Density Estimation
short: Kernel density
when: You want to see the shape of a distribution, including whether it has more than one mode, without assuming a family
data: Waiting times between eruptions of Old Faithful geyser, 272 eruptions (Azzalini and Bowman 1990; datasets, faithful)
tools: Python · OES engine · NumPy
oes: kde
lead: Use this guide to estimate a density without assuming a distribution and to count its modes. A kernel density estimate puts a small bump on every observation and adds them up; the bandwidth is the only choice that matters. The guide covers the estimator, four bandwidth selectors, the Silverman test for the number of modes, a comparison with a normal fit, a simulation of the selectors and the write-up.
---

## The question and the model

**What does the distribution look like, and is it one hump or several?** The estimate is

\[
\hat f(x) = \frac{1}{nh}\sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right).
\]

The bandwidth \(h\) is chosen by a rule (Scott, Silverman), by least-squares cross-validation, or by the Sheather and Jones (1991) plug-in, which is close to the optimum for non-normal densities. Silverman (1981) tests the null of at most \(k\) modes through the critical bandwidth, the smallest \(h\) at which the estimate has \(k\) modes, and a smoothed bootstrap.

<div class="write" markdown="1">
**How to write it: the model**

We estimate the density of waiting times between eruptions with a Gaussian kernel. We compare the Scott, Silverman, cross-validation and Sheather-Jones bandwidths, take the Sheather-Jones value, report the modes and test the null of a single mode with Silverman's (1981) smoothed bootstrap. We check the choice of bandwidth in a simulation with a known two-component mixture.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Estimate the density

```python
@@EST@@
```

@@READ_EST@@

## Modes and the normal comparison

```python
@@MODES@@
```

@@READ_MODES@@

## Robustness: the bandwidth

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: bandwidth selectors in a simulation

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Silverman (1981), *Journal of the Royal Statistical Society B*; Silverman (1986), *Density Estimation for Statistics and Data Analysis*; Sheather and Jones (1991), *Journal of the Royal Statistical Society B*; Azzalini and Bowman (1990), *Journal of the Royal Statistical Society C*.
