# Lecture 11 — VAR and Structural VAR: Dynamic Interactions and Shock Identification

## Learning objectives
By the end of this lecture you should be able to:

1. specify and estimate a reduced-form VAR, check stability, choose lag length, and test Granger non-causality;
2. compute impulse responses and forecast-error variance decompositions from the VAR's moving-average representation;
3. state the structural identification problem precisely and count the restrictions needed;
4. implement and critique recursive (Cholesky), long-run, sign-restriction and external-instrument (proxy) identification;
5. compare VAR impulse responses with local projections, and construct valid confidence bands.

## Motivation
Macroeconomists want answers to questions such as: what happens to output and inflation, quarter by quarter, after an unexpected monetary tightening? Sims (1980) proposed letting a small system of variables speak with minimal theoretical restrictions: the vector autoregression. A VAR summarizes the joint dynamics of the data, but its residuals are forecast errors, not economic shocks. Getting from one to the other requires identifying assumptions, and the credibility of every structural VAR result rests on them. This lecture is the time-series counterpart of the simultaneous-equations problem in Lecture 6.

## 1. The reduced-form VAR
For an \(n\times1\) vector \(y_t\), a VAR(\(p\)) is
\[
y_t=c+A_1y_{t-1}+\dots+A_py_{t-p}+u_t,\qquad E[u_t]=0,\quad E[u_tu_t']=\Sigma_u,\quad E[u_tu_s']=0\;(s\neq t).
\]
**Estimation.** Each equation has the same regressors, so equation-by-equation OLS is identical to SUR/GLS and is efficient (Zellner's result). Under Gaussian errors, it is also the conditional MLE.

**Stability.** Stack \(Y_t=(y_t',\dots,y_{t-p+1}')'\) to get the **companion form** \(Y_t=\mathbf c+\mathbf AY_{t-1}+U_t\). The VAR is stable (stationary) if all eigenvalues of \(\mathbf A\) have modulus less than one.

**Lag length.** Use AIC, BIC or Hannan–Quinn computed on a common sample, check residual autocorrelation (multivariate portmanteau or LM tests), and be aware of the parameter count: \(n(np+1)\) coefficients. A VAR with 7 variables and 4 lags has 203. **Bayesian VARs** with Minnesota-type shrinkage priors handle larger systems (Bańbura, Giannone and Reichlin, 2010).

**Granger causality.** \(y_2\) does not Granger-cause \(y_1\) if lags of \(y_2\) do not help predict \(y_1\) given lags of \(y_1\): a Wald test that the relevant blocks of \(A_1,\dots,A_p\) are zero. It is a statement about **incremental predictability**, not causation. Forward-looking variables such as asset prices Granger-cause the variables they anticipate.

## 2. Moving-average representation, impulse responses and variance decompositions
A stable VAR has the Wold representation
\[
y_t=\mu+\sum_{h=0}^\infty\Psi_hu_{t-h},\qquad \Psi_0=I,\quad \Psi_h=\sum_{j=1}^{\min(h,p)}A_j\Psi_{h-j}.
\]
\(\Psi_h\) gives the response of \(y_{t+h}\) to a reduced-form innovation at \(t\). But reduced-form innovations are correlated (\(\Sigma_u\) is not diagonal), so "a shock to \(u_{1t}\) holding \(u_{2t}\) fixed" is not meaningful.

**Structural shocks.** Suppose \(u_t=B_0^{-1}\varepsilon_t\), with structural shocks \(\varepsilon_t\) mutually uncorrelated and normalized to \(E[\varepsilon_t\varepsilon_t']=I_n\). Then the **structural impulse responses** are
\[
\Theta_h=\frac{\partial y_{t+h}}{\partial\varepsilon_t'}=\Psi_hB_0^{-1},
\]
and the **forecast-error variance decomposition** gives the share of the \(h\)-step forecast error variance of variable \(i\) due to shock \(j\):
\[
\omega_{ij}(h)=\frac{\sum_{s=0}^{h-1}[\Theta_s]_{ij}^2}{\sum_{s=0}^{h-1}\sum_{k=1}^n[\Theta_s]_{ik}^2}.
\]

## 3. The identification problem
The data identify \(\Sigma_u\), which has \(n(n+1)/2\) distinct elements. The structure requires the \(n^2\) elements of \(B_0^{-1}\), linked by
\[
\Sigma_u=B_0^{-1}B_0^{-1\prime}.
\]
If \(B_0^{-1}\) satisfies this, so does \(B_0^{-1}Q\) for **any** orthogonal matrix \(Q\). We therefore need at least \(n(n-1)/2\) additional restrictions for exact identification: 1 for \(n=2\), 3 for \(n=3\), 21 for \(n=7\). Identification can also target a single shock of interest, which needs fewer restrictions.

## 4. Identification strategies
**(a) Short-run (recursive) restrictions.** Set \(B_0^{-1}=P\), the lower-triangular Cholesky factor of \(\Sigma_u\). Variable 1 then responds on impact only to shock 1, variable 2 to shocks 1 and 2, and so on. The **ordering is the identifying assumption**. Christiano, Eichenbaum and Evans (1999) order output and prices before the federal funds rate: the policy rate responds within the period to output and prices, but they respond to policy only with a lag. The assumption is plausible for slow-moving real variables at monthly or quarterly frequency, and implausible for financial variables that react instantly.

**(b) Long-run restrictions (Blanchard and Quah, 1989).** Restrict the cumulative effect \(\Theta(1)=\Psi(1)B_0^{-1}\), where \(\Psi(1)=(I-A_1-\dots-A_p)^{-1}\). For example, "demand shocks have no long-run effect on the level of output", with output entering in differences. Since \(\Theta(1)\Theta(1)'=\Psi(1)\Sigma_u\Psi(1)'\), a lower-triangular Cholesky factor of the right-hand side delivers \(\Theta(1)\), and \(B_0^{-1}=\Psi(1)^{-1}\Theta(1)\). Long-run restrictions depend on estimates of \(\Psi(1)\), which are imprecise in finite samples (Faust and Leeper, 1997).

**(c) Sign restrictions (Faust, 1998; Uhlig, 2005).** Require responses to have particular signs for some horizons: a monetary tightening raises the policy rate and does not raise prices for six months. Draw orthogonal \(Q\) at random, keep the rotations \(PQ\) that satisfy the signs, and report the set of accepted responses. **The result is set identification.** The usual "median and percentile bands" mix sampling uncertainty with uniform-prior weighting over the identified set. That prior is informative about the responses themselves (Baumeister and Hamilton, 2015). Report the bounds of the identified set, or state explicit priors.

**(d) External instruments / proxy SVAR (Stock and Watson, 2012, 2018; Mertens and Ravn, 2013).** Suppose an external series \(z_t\) is correlated with the shock of interest and uncorrelated with all other structural shocks:
\[
E[z_t\varepsilon_{1t}]=\alpha\neq0,\qquad E[z_t\varepsilon_{jt}]=0\;(j\neq1).
\]
Then \(E[z_tu_t]=b_1\alpha\), where \(b_1\) is the first column of \(B_0^{-1}\). The impact responses are therefore identified up to scale by regressing each \(u_{it}\) on \(z_t\), normalized so the shock raises \(y_{1t}\) by one unit. High-frequency surprises in interest-rate futures around FOMC announcements (Gertler and Karadi, 2015) and narrative series (Romer and Romer, 2004) are leading proxies. The weak-instrument issues of Lecture 5 apply in full: report the first-stage \(F\) of the proxy on \(u_{1t}\).

**(e) Other routes.** Identification through heteroskedasticity (Rigobon, 2003; Lecture 6) and through non-Gaussianity (independent component analysis) use higher moments of the data.

## 5. Local projections and inference
**Local projections (Jordà, 2005).** Estimate impulse responses directly, horizon by horizon:
\[
y_{i,t+h}=\alpha_h+\theta_{h}\,s_t+\gamma_h'w_t+e_{t+h},\qquad h=0,1,\dots,H,
\]
where \(s_t\) is an identified shock (or a proxy, estimated by IV) and \(w_t\) contains lagged controls. In population, with the same information set, LPs and VARs estimate the **same** impulse responses (Plagborg-Møller and Wolf, 2021). In finite samples, LPs have lower bias at long horizons but much higher variance. VARs are more efficient but can be badly biased if the lag length is too short (Li, Plagborg-Møller and Wolf, 2024). LP residuals are serially correlated, so use HAC standard errors or include lags of the shock and outcome ("lag-augmented" LP; Montiel Olea and Plagborg-Møller, 2021).

**Inference for impulse responses.** Delta-method bands can be poor because responses are highly nonlinear functions of the VAR coefficients. Bootstrap methods (residual-based, wild for heteroskedasticity, or Kilian's 1998 bias-corrected bootstrap-after-bootstrap) are standard. Pointwise bands do not describe the uncertainty about the whole response path. Joint (simultaneous) bands are wider. With persistent or near-unit-root data, VAR-based bands at long horizons undercover.

## Worked example
**A monetary-policy VAR.** Take quarterly U.S. data on log real GDP, log GDP deflator, a commodity price index, and the federal funds rate (in that order), with 4 lags in levels, recursive identification, and the policy shock as the last shock. The classic findings (Christiano, Eichenbaum and Evans, 1999) are:

- after a 100 bp tightening, output falls with a hump-shaped response, reaching a trough after 6–8 quarters, and then recovers;
- prices respond sluggishly. In specifications **without** commodity prices, the price level often *rises* for several quarters: the **price puzzle**. Including commodity prices, which proxy for information the Fed has about future inflation, largely removes it. This illustrates that a recursive VAR can confound policy shocks with the Fed's endogenous response to anticipated inflation.

Later work using high-frequency proxies (Gertler and Karadi, 2015) finds similar output effects without the price puzzle, at the cost of weaker instruments in some samples. The lesson for practice: report results under several identification schemes, and explain which economic assumption each one embodies.

## Problem set
1. **Counting restrictions.** In a 4-variable VAR, how many restrictions does exact identification of all shocks require? How many are imposed by a recursive ordering?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(n(n-1)/2=6\). A lower-triangular \(B_0^{-1}\) has 6 zeros above the diagonal, so a recursive ordering exactly identifies all four shocks.
    </details>

2. **Bivariate Cholesky.** With \(\Sigma_u=\begin{pmatrix}4&2\\2&5\end{pmatrix}\), compute \(P\) and the impact responses to each structural shock. Reverse the ordering and recompute. What changes, and why?

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(P=\begin{pmatrix}2&0\\1&2\end{pmatrix}\): a shock 1 raises \(y_1\) by 2 and \(y_2\) by 1; shock 2 moves only \(y_2\), by 2. Reversed ordering: the Cholesky factor of \(\begin{pmatrix}5&2\\2&4\end{pmatrix}\) is \(\begin{pmatrix}\sqrt5&0\\2/\sqrt5&\sqrt{16/5}\end{pmatrix}\). Now the shock that moves \(y_2\) first also moves \(y_1\) by \(2/\sqrt5\approx0.89\) on impact. The covariance \(2\) is attributed to whichever shock is ordered first.
    </details>

3. **Granger causality is not causality.** Construct an example in which stock prices Granger-cause GDP although stock prices have no causal effect on GDP.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Let GDP be driven by a news shock \(n_t\): \(g_{t+1}=n_t+e_{t+1}\). The stock price is the expected discounted value of future GDP, so \(s_t\) moves with \(n_t\) at time \(t\). Lagged \(s\) then predicts \(g\), but intervening on \(s\) (for example, by a transaction tax) would not change \(g\).
    </details>

4. **Proxy SVAR.** Show that with a valid proxy \(z_t\), \(E[z_tu_{it}]/E[z_tu_{1t}]=b_{i1}/b_{11}\), where \(b_{ij}\) are elements of \(B_0^{-1}\).

    <details class="solution" markdown="1"><summary>Solution</summary>

    \(u_t=B_0^{-1}\varepsilon_t\), so \(E[z_tu_t]=B_0^{-1}E[z_t\varepsilon_t]=B_0^{-1}(\alpha,0,\dots,0)'=\alpha b_1\). Taking ratios removes \(\alpha\).
    </details>

5. **Sign restrictions.** Explain why the median of accepted impulse responses across rotations is not a point estimate of a structural response.

    <details class="solution" markdown="1"><summary>Solution</summary>

    Under sign restrictions, every rotation in the identified set is equally consistent with the data. The distribution across rotations reflects the (implicit Haar) prior over \(Q\), not sampling uncertainty. Moreover, the pointwise median at each horizon need not correspond to any single admissible model (Fry and Pagan, 2011).
    </details>

## Computational lab
**Goal:** estimate a small VAR, impose recursive identification, and compare with local projections.

```python
import numpy as np
from statsmodels.tsa.api import VAR

rng = np.random.default_rng(11)
T = 400
A1 = np.array([[0.5, 0.0, -0.1],      # output gap
               [0.1, 0.6,  0.0],      # inflation
               [0.2, 0.3,  0.7]])     # policy rate (Taylor-type response)
B0inv = np.array([[1.0, 0.0, 0.0],
                  [0.3, 0.8, 0.0],
                  [0.4, 0.5, 0.6]])   # true recursive impact matrix
y = np.zeros((T, 3))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + B0inv @ rng.normal(size=3)

res = VAR(y).fit(maxlags=8, ic="bic")
irf = res.irf(12)
print("lags chosen:", res.k_ar)
print("estimated impact matrix (orthogonalized):\n", irf.orth_irfs[0].round(2))

# local projection of output on the identified policy shock (third Cholesky shock)
P = np.linalg.cholesky(res.sigma_u)
eps = np.linalg.solve(P, res.resid.T).T[:, 2]
yy = y[res.k_ar:]
lp = []
for h in range(9):
    X = np.column_stack([np.ones(len(eps) - h - 1), eps[1:len(eps) - h], yy[:len(yy) - h - 1]])
    lp.append(np.linalg.lstsq(X, yy[h + 1:, 0], rcond=None)[0][1])
print("LP response of output to policy shock:", np.round(lp, 3))
print("VAR response:                         ", irf.orth_irfs[:9, 0, 2].round(3))
```

Tasks: (i) compare the estimated impact matrix with `B0inv`; (ii) reorder the variables and show how the impact responses change; (iii) plot VAR responses with bootstrap bands (`irf.plot(orth=True)`) alongside LP responses with HAC bands. In R, use `vars::VAR`, `vars::irf` and `lpirfs`. Real-data extension: FRED-QD variables GDPC1, GDPDEF, PPIACO and FEDFUNDS for the worked example.

## Coding exercises

These exercises use **real data** and run in Python. Install the packages once with
`pip install wooldridge statsmodels linearmodels scipy`. The `wooldridge` package contains the datasets
from Wooldridge's *Introductory Econometrics*; `statsmodels` ships U.S. and Danish macroeconomic series.
Try the starter code first, then open the solution. Every printed result below was produced by running the solution code exactly as shown.

### Exercise 11.1 — A three-variable monetary VAR for the U.S.

*Data:* `macrodata` — U.S. quarterly data, 1959Q1–2009Q3: real GDP, CPI inflation, 3-month T-bill rate.

1. Build a VAR in GDP growth (annualized, %), inflation, and the T-bill rate. Choose the lag by AIC and BIC (max 8).
2. Check stability (eigenvalues of the companion matrix).
3. Test whether the T-bill rate Granger-causes GDP growth.
4. Compute recursive (Cholesky) impulse responses to a one-standard-deviation interest-rate shock, ordering growth, inflation, rate. Report the response of GDP growth and inflation over 12 quarters.
5. Report the 12-quarter forecast-error variance decomposition of GDP growth.

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
model = VAR(df)
# TODO: model.select_order(8); res = model.fit(p); res.is_stable(); res.test_causality(...); res.irf(12); res.fevd(12)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
model = VAR(df.to_numpy())
sel = model.select_order(8)
print("lag order by AIC:", sel.aic, " by BIC:", sel.bic)
res = model.fit(4)
companion = np.vstack([np.hstack(res.coefs), np.eye(3 * 3, 3 * 4)])      # [A1 A2 A3 A4; I 0]
print("stable:", res.is_stable(), " max |eigenvalue| of companion:", round(max(abs(np.linalg.eigvals(companion))), 3))
gc = res.test_causality(0, [2], kind="wald")
print(f"Granger: rate -> growth  chi2 = {gc.test_statistic:.2f}, p = {gc.pvalue:.4f}")
irf = res.irf(12).orth_irfs
print("response to a 1-s.d. rate shock (Cholesky: growth, infl, rate):")
print(pd.DataFrame({"growth": irf[:, 0, 2], "infl": irf[:, 1, 2], "rate": irf[:, 2, 2]}).iloc[[0, 1, 2, 4, 6, 8, 12]].round(3).to_string())
fe = res.fevd(12).decomp[0, -1]
print("FEVD of growth at h=12 (shares of growth, infl, rate shocks):", fe.round(3))
```

Output:

```text
lag order by AIC: 6  by BIC: 1
stable: True  max |eigenvalue| of companion: 0.902
Granger: rate -> growth  chi2 = 15.16, p = 0.0044
response to a 1-s.d. rate shock (Cholesky: growth, infl, rate):
    growth   infl   rate
0    0.000  0.000  0.731
1    0.458  0.488  0.713
2   -0.498  0.177  0.492
4    0.060  0.376  0.582
6   -0.116  0.144  0.463
8   -0.036  0.079  0.388
12   0.006  0.016  0.271
FEVD of growth at h=12 (shares of growth, infl, rate shocks): [0.867 0.09  0.043]
```

AIC and BIC disagree on the lag length, a typical outcome; we use four lags (one year), a common compromise for quarterly
data. The system is stable. Interest rates Granger-cause GDP growth strongly — but as Problem 3 explains, that may reflect
the Fed reacting to information about future growth, not a causal effect of policy. After a recursive interest-rate
shock, output growth first *rises* and then falls in the second quarter, and **inflation rises** for several quarters: the
**price puzzle** of Lecture 11. The recursive ordering attributes to "policy shocks" some of the Fed's response to anticipated inflation. The
FEVD shows that interest-rate shocks explain only about 4% of the variance of growth at a 3-year horizon; almost 90% is
due to growth's own shocks.

</details>

### Exercise 11.2 — How much does the ordering matter?

*Data:* `macrodata`, same VAR as above.

1. Re-estimate the Cholesky responses with the rate ordered **first** (the policy rate cannot respond within the quarter to output and inflation).
2. Compare the impact responses and the 8-quarter responses of growth to the rate shock under the two orderings.
3. Compute the correlation of the reduced-form residuals. Why does the ordering matter more when this correlation is large?

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
# TODO: fit VAR(4) on df and on df[["rate", "growth", "infl"]]; compare orth_irfs
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.api import VAR
m = sm.datasets.macrodata.load_pandas().data
df = pd.DataFrame({"growth": 400 * np.log(m["realgdp"]).diff(), "infl": m["infl"], "rate": m["tbilrate"]}).dropna()
a = VAR(df).fit(4); b = VAR(df[["rate", "growth", "infl"]]).fit(4)
ia, ib = a.irf(12).orth_irfs, b.irf(12).orth_irfs
print("response of growth to the rate shock")
print(pd.DataFrame({"rate ordered last": ia[:, 0, 2], "rate ordered first": ib[:, 1, 0]}).iloc[[0, 1, 2, 4, 8]].round(3).to_string())
print("\nreduced-form residual correlations:")
print(pd.DataFrame(a.resid).corr().round(3).to_string())
```

Output:

```text
response of growth to the rate shock
   rate ordered last  rate ordered first
0              0.000               0.951
1              0.458               0.742
2             -0.498              -0.297
4              0.060              -0.049
8             -0.036              -0.160

reduced-form residual correlations:
        growth   infl   rate
growth   1.000  0.145  0.304
infl     0.145  1.000  0.350
rate     0.304  0.350  1.000
```

With the rate ordered last, growth cannot respond on impact by construction. With the rate ordered first, the impact
response is whatever the residual correlation implies. That correlation is positive: in the data, unexpected rate rises
come with unexpectedly strong growth, because the Fed responds within the quarter. Ordering the rate first therefore
attributes that co-movement to the policy shock, and growth appears to **rise** on impact after a tightening — an
implausible result. Identification is an economic assumption about timing, and the data alone cannot settle it. External
instruments (high-frequency surprises) are the modern response.

</details>

## Reading
**Required:** Kilian and Lütkepohl, *Structural Vector Autoregressive Analysis* (2017), chapters 2, 8–9 and 12–13; Stock and Watson (2016), "Dynamic factor models, factor-augmented VARs, and SVARs in macroeconomics", *Handbook of Macroeconomics* vol. 2.
**Further:** Sims (1980), "Macroeconomics and reality", *Econometrica* 48(1); Blanchard and Quah (1989), *AER* 79(4); Christiano, Eichenbaum and Evans (1999), *Handbook of Macroeconomics* vol. 1A; Uhlig (2005), *Journal of Monetary Economics* 52(2); Baumeister and Hamilton (2015), *Econometrica* 83(5); Mertens and Ravn (2013), *AER* 103(4); Gertler and Karadi (2015), *AEJ: Macro* 7(1); Jordà (2005), *AER* 95(1); Plagborg-Møller and Wolf (2021), "Local projections and VARs estimate the same impulse responses", *Econometrica* 89(2).
