# Lecture 12 — Multiple Integrals, Change of Variables, and Probability

## Learning objectives
Evaluate double and triple integrals; use Fubini's theorem; change coordinates with Jacobians; derive marginal densities and expectations.

## 1. Double integrals
For an integrable function over a region \(D\subset\mathbb R^2\), the double integral \(\iint_D f(x,y)dA\) aggregates values over area. Iterated integration is valid under Fubini's conditions, including integrability; Tonelli's theorem applies to nonnegative measurable functions.

## 2. Integration over general regions
Describe regions using bounds such as \(a\le x\le b\), \(g_1(x)\le y\le g_2(x)\). Changing the order of integration requires rewriting the same region with the other variable outermost. Sketch the region before changing limits.

## 3. Change of variables
For a differentiable one-to-one transformation \((u,v)\mapsto(x,y)\),
\[
\iint_D f(x,y)dxdy
=\iint_{D^*}f(x(u,v),y(u,v))\left|\frac{\partial(x,y)}{\partial(u,v)}\right|dudv.
\]
The absolute Jacobian determinant adjusts local area. Polar coordinates use \(x=r\cos\theta\), \(y=r\sin\theta\), with Jacobian \(r\).

## 4. Probability applications
For joint density \(f_{X,Y}\), marginals integrate out other variables. Expectations are integrals of functions against the joint density:
\[
E[g(X,Y)]=\iint g(x,y)f_{X,Y}(x,y)dxdy.
\]
Covariance is \(E[XY]-E[X]E[Y]\) when the moments exist. Transformations of random vectors use the same Jacobian principle.

## 5. Applications
Multiple integrals calculate volumes, expected utilities, aggregate welfare, and probability over regions. In economics, they can represent expected payoffs under joint uncertainty and aggregation over heterogeneous agents.

## Worked example
For a uniform point on the unit disk, the density is \(1/\pi\). Polar coordinates give \(P(R\le r)=\pi r^2/\pi=r^2\), for \(0\le r\le1\), and hence radial density \(f_R(r)=2r\).

## Problem set
1. Evaluate a double integral over a rectangular region.
2. Reverse the order of integration over a triangular region.
3. Derive the polar-coordinate Jacobian.
4. Obtain a marginal density from a joint density.
5. Compute an expectation under a bivariate distribution.

## Computational lab
Evaluate integrals symbolically and numerically. Compare Cartesian and polar integration for a circular region and verify numerical marginal densities.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 12.1 — A bivariate normal probability by double integration

*Data:* `econmath` — ACT math scores and college GPA of 814 students.

1. Fit a bivariate normal to `(actmth, colgpa)`.
2. Compute \(P(\text{ACT math}\ge28,\ GPA\ge3.5)\) by numerical double integration of the density (`scipy.integrate.dblquad`).
3. Check it with the bivariate normal CDF and with Monte Carlo integration (one million draws).
4. Compare with the empirical frequency in the data. What does any gap say about the normal model?

```python
import numpy as np, wooldridge as wd
from scipy import stats, integrate
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mvn = stats.multivariate_normal(d.mean(), d.cov())
# TODO: integrate.dblquad(lambda g, a: mvn.pdf([a, g]), 28, inf, 3.5, inf)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats, integrate
d = wd.data("econmath")[["actmth", "colgpa"]].dropna()
mu, S = d.mean().to_numpy(), d.cov().to_numpy(); mvn = stats.multivariate_normal(mu, S)
dbl, err = integrate.dblquad(lambda g, a: mvn.pdf([a, g]), 28, mu[0] + 10 * np.sqrt(S[0, 0]), 3.5, mu[1] + 10 * np.sqrt(S[1, 1]))
cdf = 1 - stats.norm.cdf(28, mu[0], np.sqrt(S[0, 0])) - stats.norm.cdf(3.5, mu[1], np.sqrt(S[1, 1])) + mvn.cdf([28, 3.5])
mc = np.mean(np.all(np.random.default_rng(0).multivariate_normal(mu, S, 1_000_000) >= [28, 3.5], axis=1))
print(f"double integral {dbl:.5f} (error estimate {err:.1e});  via CDF {cdf:.5f};  Monte Carlo {mc:.5f}")
emp = np.mean((d["actmth"] >= 28) & (d["colgpa"] >= 3.5))
print(f"empirical frequency: {emp:.5f} ({int(emp*len(d))} of {len(d)} students)")
```

Output:

```text
double integral 0.02212 (error estimate 4.5e-09);  via CDF 0.02212;  Monte Carlo 0.02208
empirical frequency: 0.03194 (26 of 814 students)
```

Double integration, the CDF formula and Monte Carlo all give about 2.2% for the joint event. In the data, 3.2% of students
(26 of 814) have both an ACT math score of 28 or more and a GPA of 3.5 or more — about 45% more than the normal model
predicts. The bivariate normal understates the joint upper tail: strong students are more concentrated at the top than a
normal distribution with correlation 0.3 allows (and GPA is capped at 4). Joint tail probabilities are exactly where normal
approximations tend to fail.

</details>

### Exercise 12.2 — Change of variables: from log wages to wages

*Data:* `wage1` — hourly wages.

If \(Z=\log W\sim N(\mu,\sigma^2)\), the density of \(W=e^Z\) follows from the change-of-variables formula \(f_W(w)=f_Z(\log w)\,|dz/dw|=f_Z(\log w)/w\).

1. Estimate \(\mu,\sigma\) from log wages, and derive \(f_W\) with `sympy` using the Jacobian.
2. Check numerically that \(f_W\) integrates to one, and that \(\int w f_W(w)\,dw=e^{\mu+\sigma^2/2}\).
3. Compare the implied mean wage with the sample mean, and the implied probability of earning less than \$3/hour with the sample share.

```python
import numpy as np, sympy as sp, wooldridge as wd
z = np.log(wd.data("wage1")["wage"].to_numpy())
w, mu, s = sp.symbols("w mu sigma", positive=True)
fz = lambda x: sp.exp(-(x - mu)**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
# TODO: fw = fz(sp.log(w)) * sp.diff(sp.log(w), w)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd
from scipy import integrate
wage = wd.data("wage1")["wage"].to_numpy(); z = np.log(wage)
w, mu, s = sp.symbols("w mu sigma", positive=True)
fz = lambda x: sp.exp(-(x - mu)**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
fw = fz(sp.log(w)) * sp.diff(sp.log(w), w)
print("f_W(w) =", sp.simplify(fw))
vals = {mu: z.mean(), s: z.std(ddof=1)}
f = sp.lambdify(w, fw.subs(vals))
tot = integrate.quad(f, 0, np.inf)[0]; mean = integrate.quad(lambda x: x * f(x), 0, np.inf)[0]
print(f"mu = {vals[mu]:.4f}, sigma = {vals[s]:.4f}")
print(f"integral of f_W = {tot:.8f};  E[W] by integration = {mean:.4f}, formula e^(mu+s^2/2) = {np.exp(vals[mu] + vals[s]**2/2):.4f}, sample mean = {wage.mean():.4f}")
print(f"P(W < 3): lognormal {integrate.quad(f, 0, 3)[0]:.4f}, sample share {np.mean(wage < 3):.4f}")
```

Output:

```text
f_W(w) = sqrt(2)*exp(-(mu - log(w))**2/(2*sigma**2))/(2*sqrt(pi)*sigma*w)
mu = 1.6233, sigma = 0.5315
integral of f_W = 1.00000000;  E[W] by integration = 5.8389, formula e^(mu+s^2/2) = 5.8389, sample mean = 5.8961
P(W < 3): lognormal 0.1618, sample share 0.1160
```

`sympy` produces the lognormal density through the Jacobian \(1/w\). Numerically it integrates to one, and its mean
equals \(e^{\mu+\sigma^2/2}\approx\$5.84\), close to the sample mean of \$5.90. The lognormal overstates the share of very
low wages (16% below \$3 against 12% in the data), because the 1976 minimum wage of \$2.30 truncates the lower tail of
actual wages. The change of variables is exact; the model's fit is the empirical question.

</details>

## Reading
Apostol, *Calculus*, Vol. 2; Casella and Berger, *Statistical Inference*; Simon and Blume, *Mathematics for Economists*.