title: Nelson-Siegel Yield Curve
short: Yield curve
when: You have yields at several maturities and want to summarise the curve with a few interpretable factors (level, slope, curvature), see how they move, and forecast them
data: U.S. Treasury yields at 1 to 120 months maturity, monthly December 1946 to February 1991 (McCulloch and Kwon 1993); 531 months
tools: Python · OES engine · SciPy
oes: nelson_siegel
lead: Use this guide to compress a yield curve into three factors and to forecast it. The Nelson-Siegel model of the curve, in the dynamic form of Diebold and Li (2006), fits level, slope and curvature factors to the yields of each date by least squares. The guide covers the model, the choice of the decay parameter, the fit by maturity, the behaviour of the factors, a forecast comparison with the random walk, robustness to the decay parameter, a simulation check and the write-up.
---

## The question and the model

**What is the shape of the yield curve, and how does it move?** The Nelson-Siegel curve for the yield of maturity \(\tau\) is

\[
y_t(\tau) = \beta_{0t} + \beta_{1t}\,\frac{1 - e^{-\lambda\tau}}{\lambda\tau} + \beta_{2t}\left(\frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right).
\]

Each factor has a meaning:

| Factor | Loading | Interpretation |
|---|---|---|
| \(\beta_0\) | 1 at all maturities | **Level**: the long-term yield |
| \(\beta_1\) | Decays from 1 to 0 | **Slope**: the short rate minus the long rate with a negative sign (a negative \(\beta_1\) is a normal, upward-sloping curve) |
| \(\beta_2\) | Hump-shaped | **Curvature**: the medium-term yield relative to the short and long ones |

The parameter \(\lambda\) fixes where the curvature loading peaks (at about \(1.79/\lambda\) months). With \(\lambda\) given, the factors are linear and are estimated by OLS at every date; \(\lambda\) is estimated here by minimising the squared fitting error over the whole sample. Diebold and Li (2006) model the factors as AR(1) processes and forecast the curve by forecasting the factors.

<div class="write" markdown="1">
**How to write it: the model**

We fit the Nelson-Siegel curve to the yields of seven maturities (1, 3, 6, 12, 36, 60 and 120 months) at every month, with the decay parameter \(\lambda\) chosen to minimise the squared fitting error over the whole sample. The level, slope and curvature factors are estimated by OLS at each date. We forecast the yields 12 months ahead by AR(1) forecasts of the factors, estimated on an expanding window, and compare them with the random walk and with AR(1) forecasts of each yield, with root mean squared errors on the last 40% of the sample.
</div>

## Load the data

```python
@@LOAD@@
```

@@READ_LOAD@@

## Fit the curve

```python
@@EST@@
```

@@READ_EST@@

## The factors over time

```python
@@FAC@@
```

@@READ_FAC@@

## Forecast the curve

```python
@@FC@@
```

@@READ_FC@@

## Robustness: the decay parameter and the horizon

```python
@@ROB@@
```

@@READ_ROB@@

## Diagnostics: does the fit recover a known curve?

```python
@@SIM@@
```

@@READ_SIM@@

## Write the results

@@WRITE@@

**References.** Nelson and Siegel (1987), *Journal of Business*; Diebold and Li (2006), *Journal of Econometrics*; Svensson (1994), NBER Working Paper 4871; Litterman and Scheinkman (1991), *Journal of Fixed Income*; Diebold, Rudebusch and Aruoba (2006), *Journal of Econometrics*; McCulloch and Kwon (1993), Ohio State University Working Paper 93-6.
