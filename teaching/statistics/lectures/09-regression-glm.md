# Lecture 09 — Linear Regression and Generalized Linear Models

## Learning objectives
Derive the linear model; interpret conditional mean parameters; specify GLMs; interpret logistic and Poisson regression; assess overdispersion and diagnostics.

## 1. Linear regression
Write \(Y_i=x_i'\beta+\varepsilon_i\), with \(E(\varepsilon_i\mid x_i)=0\). Then \(E(Y_i\mid x_i)=x_i'\beta\). Under full column rank, OLS is \(\widehat\beta=(X'X)^{-1}X'Y\). Regression association is not automatically causal; causal interpretation requires an identification argument.

## 2. GLM structure
A generalized linear model specifies a response distribution from the exponential family, linear predictor \(\eta_i=x_i'\beta\), and link \(g(\mu_i)=\eta_i\), where \(\mu_i=E(Y_i\mid x_i)\). The variance function links conditional variance to the mean.

## 3. Logistic regression
For binary \(Y_i\),
\[
\log\frac{p_i}{1-p_i}=x_i'\beta.
\]
A one-unit increase in \(x_j\) changes log odds by \(\beta_j\); \(e^{\beta_j}\) is the conditional odds ratio. The probability marginal effect is \(\beta_jp_i(1-p_i)\), which depends on covariates.

## 4. Poisson regression
For counts, \(Y_i\mid x_i\sim Poisson(\mu_i)\), with \(\log\mu_i=x_i'\beta\). Then \(e^{\beta_j}\) is the multiplicative change in the conditional mean for a one-unit change in \(x_j\). An exposure variable enters as an offset \(\log t_i\). Assess the Poisson variance restriction; robust covariance does not repair a misspecified mean or dependence structure.

## 5. Estimation and inference
GLM coefficients are usually estimated by maximum likelihood, often using iteratively reweighted least squares. Sandwich covariance can protect inference against some variance misspecification, subject to suitable independence or clustering assumptions.

## 6. Diagnostics
Assess residual patterns, leverage, influence, calibration, discrimination, overdispersion, excess zeros, and separation. Match diagnostics to the outcome distribution and scientific objective.

## Worked example
If a logistic model gives \(\widehat\beta_j=0.4\), the odds ratio is \(e^{0.4}\approx1.49\). This is not a 49-percentage-point probability increase. Transform updated odds to obtain the probability change at a specified baseline risk.

## Problem set
1. Derive OLS and state its rank condition.
2. Derive the logistic probability from the logit link.
3. Interpret a Poisson coefficient and exposure offset.
4. Explain why an odds ratio is not a risk difference.
5. Identify diagnostics for overdispersion and separation.

## Computational lab
Fit OLS, logistic, and Poisson models. Report coefficient scales and marginal effects; compare model-based and sandwich standard errors; assess calibration and residual diagnostics.

## Coding exercises

These exercises use **real data** in Python. Install once with
`pip install wooldridge statsmodels scipy numpy pandas`. The `wooldridge` package holds survey, experimental and
administrative datasets; `scipy.stats` and `statsmodels` provide the inference tools. Try the starter code first, then
open the solution. Every printed result below comes from running the solution code exactly as shown.

### Exercise 9.1 — Logistic regression: married women's labor force participation

*Data:* `mroz` — 753 married women, PSID 1975; `inlf` = in the labor force.

1. Fit a logit of `inlf` on `nwifeinc`, `educ`, `exper`, `expersq`, `age`, `kidslt6`, `kidsge6`.
2. Report odds ratios with 95% intervals.
3. Compute average marginal effects, and compare with a linear probability model.
4. What is the predicted probability of participation for a 30-year-old woman with 12 years of education, 10 years of experience, no young children, one older child and non-wife income of 20?

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("mroz")
f = "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
lg = smf.logit(f, data=d).fit()
# TODO: np.exp(lg.conf_int()); lg.get_margeff().summary(); smf.ols(f, d).fit(cov_type="HC1")
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("mroz")
f = "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
lg = smf.logit(f, data=d).fit(disp=0); lpm = smf.ols(f, data=d).fit(cov_type="HC1")
ci = np.exp(lg.conf_int())
me = lg.get_margeff().summary_frame()
print(pd.DataFrame({"odds ratio": np.exp(lg.params), "OR 2.5%": ci[0], "OR 97.5%": ci[1],
                    "logit AME": me["dy/dx"], "LPM coef": lpm.params}).drop("Intercept").round(4).to_string())
new = pd.DataFrame({"nwifeinc": [20], "educ": [12], "exper": [10], "expersq": [100], "age": [30], "kidslt6": [0], "kidsge6": [1]})
print(f"\npredicted participation: logit {lg.predict(new).iloc[0]:.3f}, LPM {lpm.predict(new).iloc[0]:.3f}")
print(f"LPM predictions outside [0,1]: {np.mean((lpm.fittedvalues < 0) | (lpm.fittedvalues > 1)):.3f};  pseudo R2 (McFadden) {lg.prsquared:.3f}")
```

Output:

```text
odds ratio  OR 2.5%  OR 97.5%  logit AME  LPM coef
age           0.9157   0.8900    0.9423    -0.0157   -0.0161
educ          1.2475   1.1457    1.3584     0.0395    0.0380
exper         1.2286   1.1538    1.3083     0.0368    0.0395
expersq       0.9969   0.9949    0.9988    -0.0006   -0.0006
kidsge6       1.0620   0.9172    1.2296     0.0107    0.0130
kidslt6       0.2361   0.1584    0.3519    -0.2578   -0.2618
nwifeinc      0.9789   0.9629    0.9952    -0.0038   -0.0034

predicted participation: logit 0.860, LPM 0.839
LPM predictions outside [0,1]: 0.044;  pseudo R2 (McFadden) 0.220
```

A child under six reduces the odds of participation by about 76% (odds ratio 0.24) and the probability by about 26
percentage points on average. Logit average marginal effects and LPM coefficients are very close — typical when
probabilities are not near 0 or 1. The LPM predicts probabilities outside \([0,1]\) for 4% of women, which the logit avoids.
The predicted participation probability for the described woman is about 0.86.

</details>

### Exercise 9.2 — Count data: Poisson, overdispersion and negative binomial for arrests

*Data:* `crime1` — 2,725 young men; `narr86` = arrests in 1986.

1. Fit a Poisson regression of `narr86` on `pcnv` (proportion of prior arrests leading to conviction), `avgsen`, `tottime`, `ptime86`, `qemp86`, `inc86`, `black`, `hispan`, `born60`.
2. Test for overdispersion (Cameron–Trivedi auxiliary regression).
3. Fit a negative binomial model and compare coefficients and standard errors. Report Poisson standard errors that are robust to overdispersion as well.
4. Interpret the coefficient on `black` as an incidence-rate ratio.

```python
import numpy as np, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("crime1")
f = "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60"
po = smf.poisson(f, data=d).fit()
# TODO: overdispersion test: regress ((y-mu)^2 - y)/mu on mu without constant; smf.negativebinomial(f, d)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("crime1")
f = "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60"
po = smf.poisson(f, data=d).fit(disp=0); por = smf.poisson(f, data=d).fit(disp=0, cov_type="HC0")
mu = po.predict(); y = d["narr86"]
ct = smf.ols("aux ~ mu - 1", data=pd.DataFrame({"aux": ((y - mu)**2 - y) / mu, "mu": mu})).fit()
print(f"overdispersion test: alpha = {ct.params['mu']:.3f} (t = {ct.tvalues['mu']:.2f});  var/mean of y = {y.var()/y.mean():.2f}")
nb = smf.negativebinomial(f, data=d).fit(disp=0)
cols = ["pcnv", "ptime86", "qemp86", "inc86", "black", "hispan"]
print(pd.DataFrame({"Poisson": po.params[cols], "se": po.bse[cols], "robust se": por.bse[cols],
                    "NegBin": nb.params[cols], "NB se": nb.bse[cols]}).round(4).to_string())
print(f"NB dispersion alpha = {nb.params['alpha']:.3f};  IRR for black (Poisson) = {np.exp(po.params['black']):.3f}")
```

Output:

```text
overdispersion test: alpha = 1.246 (t = 4.82);  var/mean of y = 1.82
         Poisson      se  robust se  NegBin   NB se
pcnv     -0.4016  0.0850     0.1011 -0.4771  0.1033
ptime86  -0.0986  0.0207     0.0223 -0.1074  0.0251
qemp86   -0.0380  0.0290     0.0341 -0.0505  0.0352
inc86    -0.0081  0.0010     0.0012 -0.0077  0.0011
black     0.6608  0.0738     0.0994  0.6560  0.0924
hispan    0.4998  0.0739     0.0924  0.5049  0.0896
NB dispersion alpha = 0.929;  IRR for black (Poisson) = 1.936
```

The overdispersion test rejects strongly. The Poisson point estimates remain consistent if the conditional mean is
correctly specified, but the conventional Poisson standard errors are too small; the robust (sandwich) standard errors are
15–35% larger and close to the negative binomial ones. NB coefficients are similar to Poisson. The incidence-rate ratio
for `black` is about 1.9: black men in this sample had nearly twice the expected number of arrests, other covariates
held fixed — a descriptive association that reflects policing as well as behavior.

</details>

## Reading
McCullagh and Nelder, *Generalized Linear Models*; Agresti, *Foundations of Linear and Generalized Linear Models*; Dobson and Barnett, *An Introduction to Generalized Linear Models*.