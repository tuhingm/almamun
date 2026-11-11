# Lecture 11 — DSGE Estimation: Likelihood, Bayesian Inference, and Model Evaluation

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Connect a solved DSGE model to observables; construct a likelihood; estimate parameters; assess identification and posterior fit.

## 1. Measurement system
A solved linear model can be written
\[
s_t=Ts_{t-1}+R\varepsilon_t,\qquad
y_t^o=Zs_t+D\varepsilon_t+\eta_t,
\]
where \(s_t\) is the state, \(y_t^o\) observed data, and \(\eta_t\) measurement error. Observables require explicit definitions, units, transformations, and sample alignment.

## 2. Kalman filter likelihood
For linear Gaussian systems, the Kalman filter recursively computes the one-step forecast error and covariance. The log likelihood is the sum of Gaussian forecast-error contributions:
\[
\ell(\theta)=-\frac12\sum_t\left[m\log(2\pi)+\log|F_t|+v_t'F_t^{-1}v_t\right],
\]
where \(v_t\) and \(F_t\) are the innovation and its covariance.

## 3. Bayesian estimation
Bayes' rule gives
\[
p(\theta\mid Y)\propto p(Y\mid\theta)p(\theta).
\]
The prior must respect parameter support and be justified. Posterior simulation commonly uses Metropolis–Hastings or other MCMC methods. Report acceptance rates, trace plots, effective sample sizes, convergence diagnostics, and Monte Carlo standard errors where available.

## 4. Identification and sensitivity
A concentrated likelihood or posterior correlation may indicate weak identification. Examine prior-to-posterior learning, likelihood profiles, local sensitivity, and alternative observable sets. Posterior predictive checks evaluate whether simulated observables reproduce salient empirical features.

## 5. Model comparison
Marginal likelihoods and information criteria answer different questions and depend on assumptions. State the criterion, prior sensitivity, and model parameterization. Do not equate a better in-sample fit with a better causal or policy model.

## 6. Computational lab
Estimate an RBC or NK model using simulated data first, then real data. Store data transformations, parameter priors, random seeds, chain settings, and posterior draws. Compare posterior predictive moments with observed moments.

## Problem set
1. Derive the Gaussian innovation log likelihood.
2. Explain the role of the measurement equation.
3. Distinguish prior, likelihood, and posterior.
4. Propose diagnostics for weak identification and MCMC convergence.

## Coding exercises

These exercises combine **real U.S. macroeconomic data** (the quarterly `macrodata` series in `statsmodels`, 1959–2009,
and annual series from the `wooldridge` package) with **numerical solutions of the models** in this lecture. Install once
with `pip install numpy scipy pandas statsmodels wooldridge`. Try the starter code first, then open the solution. Every
printed result below comes from running the solution code exactly as shown.

### Exercise 11.1 — Estimating the TFP process that drives an RBC model

*Data:* `macrodata` — U.S. Solow residual constructed as in Lecture 2.

1. Rebuild the quarterly log Solow residual, and estimate an AR(1) after removing a linear trend, by OLS and by exact Gaussian maximum likelihood (`statsmodels` ARIMA).
2. Report \(\rho\) and \(\sigma_\varepsilon\) with standard errors, and the half-life of a TFP shock.
3. Compare with the standard RBC calibration \(\rho=0.95\), \(\sigma=0.007\) (Cooley and Prescott, 1995). How sensitive is \(\rho\) to the detrending method (linear trend versus HP filter)?

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
# TODO: rebuild log z; detrend; ARIMA(z, order=(1,0,0), trend="n").fit()
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
m = sm.datasets.macrodata.load_pandas().data
Y, I = m["realgdp"].to_numpy(), m["realinv"].to_numpy(); dl, a = 0.025, 1/3
K = np.empty_like(Y); K[0] = I[0] / (np.mean(np.diff(np.log(I))) + dl)
for t in range(len(Y) - 1): K[t + 1] = (1 - dl) * K[t] + I[t]
N = m["pop"].to_numpy() * (1 - m["unemp"].to_numpy() / 100)
lz = (np.log(Y) - a * np.log(K) - (1 - a) * np.log(N))[20:]              # drop the capital-initialization years
t = np.arange(len(lz))
for lab, z in (("linear trend", lz - np.polyval(np.polyfit(t, lz, 1), t)), ("HP filter", hpfilter(lz, 1600)[0])):
    fit = ARIMA(z, order=(1, 0, 0), trend="n").fit()
    rho, sig2 = fit.params[0], fit.params[1]
    ols_rho = np.sum(z[1:] * z[:-1]) / np.sum(z[:-1]**2)
    print(f"{lab:12s}: MLE rho = {rho:.3f} (se {fit.bse[0]:.3f}), sigma = {100*np.sqrt(sig2):.3f}%  | OLS rho = {ols_rho:.3f} | half-life {np.log(0.5)/np.log(rho):.1f} quarters")
```

Output:

```text
linear trend: MLE rho = 0.940 (se 0.023), sigma = 0.722%  | OLS rho = 0.940 | half-life 11.3 quarters
HP filter   : MLE rho = 0.800 (se 0.041), sigma = 0.663%  | OLS rho = 0.800 | half-life 3.1 quarters
```

With a linear trend, U.S. TFP is highly persistent (\(\rho\approx0.94\), a half-life of almost three years), with innovations of
about 0.7% — close to the standard RBC calibration of 0.95 and 0.007. With the HP filter, persistence drops to 0.80 and the
half-life to about three quarters, because the filter removes low-frequency movements. Since the model's implied volatility
and persistence of output depend heavily on \(\rho\), the choice of detrending method is effectively a model assumption;
estimating the model and its shocks jointly (full-information Bayesian estimation) makes that assumption explicit.

</details>

### Exercise 11.2 — A Kalman filter by hand: trend inflation as a latent state

*Data:* `macrodata` — U.S. quarterly CPI inflation (annualized), 1959–2009.

State-space model: observed inflation \(\pi_t=\tau_t+e_t\), with latent persistent component \(\tau_t=\mu+\rho(\tau_{t-1}-\mu)+u_t\), \(e_t\sim N(0,\sigma_e^2)\), \(u_t\sim N(0,\sigma_u^2)\).

1. Write the Kalman filter recursions and the prediction-error log-likelihood by hand.
2. Maximize the likelihood over \((\mu,\rho,\log\sigma_u,\log\sigma_e)\) with `scipy.optimize`.
3. Check against `statsmodels` (`SARIMAX` with `measurement_error=True`).
4. Report the signal-to-noise ratio and the filtered estimate of the persistent component in 1980Q1 and 2008Q4. This is exactly how DSGE likelihoods are evaluated, with a larger state vector.

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
y = sm.datasets.macrodata.load_pandas().data["infl"].to_numpy()
def loglik(th):
    mu, rho, lsu, lse = th; su2, se2 = np.exp(2 * lsu), np.exp(2 * lse)
    # TODO: initialize x = mu, P = su2/(1-rho^2); loop: predict, update, accumulate -0.5*(log 2 pi F + v^2/F)
    ...
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
m = sm.datasets.macrodata.load_pandas().data; y = m["infl"].to_numpy()
def kfilter(th, keep=False):
    mu, rho, lsu, lse = th; su2, se2 = np.exp(2 * lsu), np.exp(2 * lse)
    x, P = mu, su2 / max(1 - rho**2, 1e-6); ll = 0.0; filt = []
    for yt in y:
        F = P + se2; v = yt - x
        ll += -0.5 * (np.log(2 * np.pi * F) + v**2 / F)
        K = P / F; x, P = x + K * v, P * (1 - K)             # update
        filt.append(x)
        x, P = mu + rho * (x - mu), rho**2 * P + su2         # predict
    return (ll, np.array(filt)) if keep else ll
res = optimize.minimize(lambda th: -kfilter(th), x0=[3.0, 0.9, 0.0, 0.5], method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
mu, rho, lsu, lse = res.x
print(f"hand-coded Kalman MLE: mu = {mu:.3f}, rho = {rho:.4f}, sigma_u = {np.exp(lsu):.3f}, sigma_e = {np.exp(lse):.3f}, logL = {-res.fun:.3f}")
sm_fit = SARIMAX(y, order=(1, 0, 0), trend="c", measurement_error=True).fit(disp=0)
print(f"statsmodels SARIMAX logL = {sm_fit.llf:.3f};  AR coefficient {sm_fit.params[1]:.4f}")
print(f"signal-to-noise ratio sigma_u^2 / sigma_e^2 = {np.exp(2*lsu)/np.exp(2*lse):.3f}")
_, filt = kfilter(res.x, keep=True)
for yr, q in ((1980, 1), (2008, 4)):
    i = int(m.index[(m.year == yr) & (m.quarter == q)][0])
    print(f"  {yr}Q{q}: observed inflation {y[i]:6.2f}%, filtered persistent component {filt[i]:6.2f}%")
```

Output:

```text
hand-coded Kalman MLE: mu = 3.693, rho = 0.9340, sigma_u = 0.962, sigma_e = 1.792, logL = -456.076
statsmodels SARIMAX logL = -456.076;  AR coefficient 0.9340
signal-to-noise ratio sigma_u^2 / sigma_e^2 = 0.288
  1980Q1: observed inflation  14.60%, filtered persistent component  12.67%
  2008Q4: observed inflation  -8.79%, filtered persistent component  -1.98%
```

The hand-coded Kalman likelihood reproduces the `statsmodels` log-likelihood to three decimals, which validates the code.
Quarterly CPI inflation is mostly noise: the signal-to-noise ratio is about 0.3. The persistent component is highly persistent
(\(\rho\approx0.93\)). In 1980Q1, the filter attributes most of the 14.6% observed inflation to the persistent component
(12.7%); in 2008Q4, it treats the −8.8% reading mostly as noise, with the persistent component near −2%. A DSGE likelihood is
the same calculation with a larger state vector generated by the model's solution.

</details>

## Further reading
An and Schorfheide (2007); Fernández-Villaverde (2010); Herbst and Schorfheide, *Bayesian Estimation of DSGE Models*.
