# Lecture 05 — Integration, Riemann Sums, and the Fundamental Theorem

## Learning objectives
Define the Riemann integral; connect antiderivatives and accumulated area; apply substitution and integration by parts; evaluate improper integrals and interpret economic accumulation.

## 1. Riemann sums
Partition \([a,b]\) into subintervals and choose sample points \(x_i^*\). The Riemann sum is \(\sum_i f(x_i^*)\Delta x_i\). If the limit exists as the maximum partition width tends to zero, it defines the Riemann integral \(\int_a^b f(x)dx\). Continuous functions on compact intervals are Riemann integrable.

## 2. Antiderivatives
An antiderivative \(F\) satisfies \(F'=f\). Indefinite integration describes a family \(F(x)+C\). Definite integration is a signed accumulation: regions below the axis contribute negatively.

## 3. Fundamental theorem of calculus
If \(f\) is continuous on \([a,b]\), then
\[
\frac{d}{dx}\int_a^x f(t)dt=f(x),\qquad
\int_a^b f(x)dx=F(b)-F(a).
\]
The first statement connects accumulation to marginal change; the second evaluates a definite integral through an antiderivative.

## 4. Integration techniques
Substitution reverses the chain rule. Integration by parts follows from the product rule:
\[
\int u\,dv=uv-\int v\,du.
\]
Partial fractions decompose rational functions when the denominator factorization permits. Select a method by inspecting the integrand structure.

## 5. Improper integrals
An infinite-interval integral is defined by a limit, e.g. \(\int_a^\infty f(x)dx=\lim_{b\to\infty}\int_a^b f(x)dx\). Integrals with singularities are similarly defined by one-sided limits. Convergence must be established before assigning a finite value.

## 6. Economic applications
Integrals measure accumulated flows, consumer and producer surplus, present value under continuous discounting, and expected values. If a flow is \(r(t)\), accumulated quantity over \([0,T]\) is \(\int_0^T r(t)dt\).

## Worked example
\[
\int_0^1 xe^{x^2}dx=\frac12\int_0^1e^u du=\frac{e-1}{2},
\]
using \(u=x^2\).

## Problem set
1. Interpret a definite integral as a limit of sums.
2. Evaluate \(\int (3x^2+2x)dx\).
3. Evaluate \(\int_0^1 xe^{x^2}dx\).
4. Use integration by parts to evaluate \(\int x e^x dx\).
5. Determine convergence of \(\int_1^\infty x^{-p}dx\).

## Computational lab
Approximate definite integrals with left, right, and midpoint sums; compare errors with exact antiderivatives. Explore numerical integration near an integrable singularity.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 5.1 — Consumer surplus as an integral under an estimated demand curve

*Data:* `apple` — randomized prices and demand for eco-labeled apples.

1. With the estimated linear demand (regular price at its mean), find the choke price where demand reaches zero.
2. Compute consumer surplus at the mean eco price, \(CS=\int_{p_0}^{p_{choke}}q(p)\,dp\), symbolically and numerically (`scipy.integrate.quad`).
3. Compute the change in consumer surplus if the eco price rises by 20 cents, using the fundamental theorem of calculus, and compare with the approximation \(-q(p_0)\Delta p\).

```python
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
from scipy import integrate
a = wd.data("apple"); r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
p = sp.symbols("p")
q = r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean()
# TODO: choke = sp.solve(q, p)[0]; CS = sp.integrate(q, (p, p0, choke))
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
from scipy import integrate
a = wd.data("apple"); r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit()
p = sp.symbols("p")
q = r.params["Intercept"] + r.params["ecoprc"] * p + r.params["regprc"] * a["regprc"].mean()
p0 = a["ecoprc"].mean(); choke = float(sp.solve(q, p)[0])
cs_sym = float(sp.integrate(q, (p, p0, choke)))
cs_num = integrate.quad(sp.lambdify(p, q), p0, choke)[0]
print(f"choke price ${choke:.3f}; demand at mean price ${p0:.3f}: {float(q.subs(p, p0)):.3f} lbs")
print(f"consumer surplus: symbolic {cs_sym:.5f}, numerical {cs_num:.5f} dollars per household")
dCS = -float(sp.integrate(q, (p, p0, p0 + 0.20)))
print(f"20-cent price rise: exact change in CS {dCS:.5f};  first-order approximation -q(p0)*dp = {-float(q.subs(p, p0))*0.20:.5f}")
```

Output:

```text
choke price $1.585; demand at mean price $1.082: 1.474 lbs
consumer surplus: symbolic 0.37121, numerical 0.37121 dollars per household
20-cent price rise: exact change in CS -0.23627;  first-order approximation -q(p0)*dp = -0.29480
```

The choke price, where predicted demand reaches zero, is about \$1.59 — the top of the survey's price range. Symbolic and
numerical integration agree: consumer surplus at the mean price is about 37 cents per household. A 20-cent price rise
lowers surplus by about 24 cents. The first-order approximation \(-q\Delta p\) overstates the loss (29 cents), because it
ignores that households buy less as the price rises — the triangle term \(\tfrac12\Delta q\Delta p\), which is large here
because demand is elastic.

</details>

### Exercise 5.2 — The Lorenz curve and Gini coefficient by numerical integration

*Data:* `401ksubs` — 9,275 U.S. households (1991 SIPP): family income `inc` (thousands of dollars).

1. Construct the Lorenz curve \(L(p)\): the share of total income received by the poorest fraction \(p\) of households.
2. Compute the Gini coefficient \(G=1-2\int_0^1L(p)\,dp\) with the trapezoidal rule.
3. Check it against the mean-difference formula \(G=\frac{E|X-Y|}{2E[X]}\) computed from the sorted data.
4. What income shares do the bottom 50% and the top 10% receive?

```python
import numpy as np, wooldridge as wd
x = np.sort(wd.data("401ksubs")["inc"].to_numpy()); n = len(x)
L = np.concatenate([[0], np.cumsum(x) / x.sum()]); P = np.arange(n + 1) / n
# TODO: gini = 1 - 2 * np.trapezoid(L, P)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
x = np.sort(wd.data("401ksubs")["inc"].to_numpy()); n = len(x)
L = np.concatenate([[0], np.cumsum(x) / x.sum()]); P = np.arange(n + 1) / n
g_int = 1 - 2 * np.trapezoid(L, P)
i = np.arange(1, n + 1)
g_md = np.sum((2 * i - n - 1) * x) / (n * x.sum())          # mean-difference formula on sorted data
print(f"Gini by integrating the Lorenz curve: {g_int:.5f};  by the mean-difference formula: {g_md:.5f}")
print(f"income share of bottom 50%: {L[n//2]:.3f};  top 10%: {1 - L[int(0.9*n)]:.3f}")
print(f"mean income ${x.mean():.1f}k, median ${np.median(x):.1f}k")
```

Output:

```text
Gini by integrating the Lorenz curve: 0.31946;  by the mean-difference formula: 0.31946
income share of bottom 50%: 0.277;  top 10%: 0.238
mean income $39.3k, median $33.3k
```

Integrating the Lorenz curve with the trapezoidal rule gives a Gini coefficient of about 0.32, identical to the
mean-difference formula — the two are algebraically the same for a step-function Lorenz curve. The poorest half of
households receive 28% of income, and the richest tenth 24%. These are 1991 SIPP data on households **eligible for 401(k)
plans**, with incomes capped at \$200k, so inequality is lower than in the full U.S. income distribution.

</details>

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*.