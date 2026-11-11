# Lecture 04 — Taylor Approximation, Convexity, and Single-Variable Optimization

## Learning objectives
Derive Taylor approximations; classify stationary points; use convexity; solve economic optimization problems and interpret first- and second-order conditions.

## 1. Taylor's theorem
\[
f(a+h)=f(a)+f'(a)h+\frac12f''(a)h^2+o(h^2).
\]
The first-order approximation is local and linear; the second-order term captures curvature. Taylor remainders determine approximation error.

## 2. Necessary conditions
For an interior local optimum \(x^*\) of a differentiable function, \(f'(x^*)=0\). This is necessary, not sufficient. Boundary points and points where the derivative does not exist must also be considered.

## 3. Second-order conditions
If \(f'(x^*)=0\) and \(f''(x^*)<0\), then \(x^*\) is a strict local maximum; if \(f''(x^*)>0\), it is a strict local minimum. If \(f''(x^*)=0\), the test is inconclusive. Global optimality follows under suitable concavity or convexity on the full feasible interval.

## 4. Convexity
A differentiable function is convex if its derivative is nondecreasing. Twice differentiability and \(f''\ge0\) provide a sufficient characterization. Strict convexity implies at most one minimizer. Jensen's inequality links convexity to expectations.

## 5. Economic optimization
For profit \(\pi(q)=pq-C(q)\), an interior optimum satisfies \(p=C'(q)\); if \(C''(q)>0\), profit is locally concave. Interpret the FOC as marginal benefit equaling marginal cost.

## Worked example
Let \(\pi(q)=20q-q^2-16\), with \(q\ge0\). The FOC gives \(q^*=10\); the SOC is \(-2<0\), and this is the global maximum on the nonnegative domain.

## Problem set
1. Derive the second-order Taylor approximation for \(e^x\) around zero.
2. Classify stationary points using the second derivative.
3. Explain why a zero second derivative is inconclusive.
4. Prove a differentiable convex function lies above each tangent.
5. Solve profit maximization with a capacity constraint.

## Computational lab
Plot objectives and tangent/quadratic approximations. Numerically optimize convex and non-convex functions; compare local optimization with grid search.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 4.1 — The Arrow–Pratt approximation with real stock returns

*Data:* `volat` — monthly S&P 500 returns (`rsp500`, % per year) and 3-month T-bill yields (`i3`), 1947–1993.

An investor with CRRA utility \(u(w)=w^{1-\gamma}/(1-\gamma)\) puts all wealth in the S&P 500 for one month.

1. Using the empirical distribution of monthly gross returns \(R\), compute the **exact** certainty equivalent \(CE=u^{-1}(E[u(R)])\) and risk premium \(E[R]-CE\) for \(\gamma=2,5,10\).
2. Compare with the second-order Taylor (Arrow–Pratt) approximation \(\pi\approx\tfrac12\gamma\operatorname{Var}(R)/E[R]\) (relative risk aversion form).
3. When does the approximation break down, and why (think of the third-order term and skewness)?

```python
import numpy as np, wooldridge as wd
v = wd.data("volat").dropna(subset=["rsp500"])
R = 1 + v["rsp500"].to_numpy() / 1200         # monthly gross return
u = lambda w, g: w**(1 - g) / (1 - g)
# TODO: CE = ((1-g) * mean(u(R)))**(1/(1-g)); compare with 0.5*g*var/mean
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, wooldridge as wd
from scipy import stats
v = wd.data("volat").dropna(subset=["rsp500"])
R = 1 + v["rsp500"].to_numpy() / 1200
m, var = R.mean(), R.var()
print(f"monthly gross return: mean {m:.5f}, sd {np.sqrt(var):.5f}, skewness {stats.skew(R):.3f}")
for g in (2, 5, 10, 50):
    EU = np.mean(R**(1 - g) / (1 - g))
    ce = ((1 - g) * EU)**(1 / (1 - g))
    exact, approx = m - ce, 0.5 * g * var / m
    print(f"gamma = {g:2d}: exact premium {1e4*exact:7.3f} bp/month, Arrow-Pratt {1e4*approx:7.3f} bp, relative error {approx/exact - 1:+.2%}")
```

Output:

```text
monthly gross return: mean 1.01012, sd 0.03355, skewness -0.452
gamma =  2: exact premium  11.356 bp/month, Arrow-Pratt  11.144 bp, relative error -1.87%
gamma =  5: exact premium  29.010 bp/month, Arrow-Pratt  27.861 bp, relative error -3.96%
gamma = 10: exact premium  60.488 bp/month, Arrow-Pratt  55.721 bp, relative error -7.88%
gamma = 50: exact premium 452.166 bp/month, Arrow-Pratt 278.605 bp, relative error -38.38%
```

For moderate risk aversion, the Arrow–Pratt approximation is excellent: the error is under 2% at \(\gamma=2\) and under
8% at \(\gamma=10\). It always understates the premium here, because monthly stock returns are **negatively skewed**:
the third-order Taylor term, which a CRRA investor dislikes, adds to the premium. At \(\gamma=50\) the approximation
misses almost 40%, because higher-order terms grow with \(\gamma\) and the distribution's tails dominate. Second-order
approximations are a local tool.

</details>

### Exercise 4.2 — A revenue-maximizing price from an estimated demand curve

*Data:* `apple` — randomized prices and demand for eco-labeled apples.

1. Using the linear demand from Lecture 1's exercise, write revenue \(TR(p)=p\,q(p)\) at the mean regular-apple price.
2. Find the revenue-maximizing price from the first-order condition, and verify the second-order condition (concavity).
3. Propagate estimation uncertainty: compute a delta-method standard error for the optimal price.
4. Is the optimal price inside the range of prices used in the survey?

```python
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit(cov_type="HC1")
p, b0, b1, b2, pr = sp.symbols("p b0 b1 b2 pr")
TR = p * (b0 + b1 * p + b2 * pr)
# TODO: solve dTR/dp = 0 for p; second derivative; gradient of p* w.r.t. (b0, b1, b2)
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
a = wd.data("apple")
r = smf.ols("ecolbs ~ ecoprc + regprc", data=a).fit(cov_type="HC1")
p, b0, b1, b2, pr = sp.symbols("p b0 b1 b2 pr")
TR = p * (b0 + b1 * p + b2 * pr)
pstar = sp.solve(sp.diff(TR, p), p)[0]
print("p* =", pstar, "   d2TR/dp2 =", sp.diff(TR, p, 2))
vals = {b0: r.params["Intercept"], b1: r.params["ecoprc"], b2: r.params["regprc"], pr: a["regprc"].mean()}
ps = float(pstar.subs(vals))
grad = np.array([float(sp.diff(pstar, s).subs(vals)) for s in (b0, b1, b2)])
se = np.sqrt(grad @ r.cov_params().to_numpy() @ grad)
print(f"revenue-maximizing eco price: ${ps:.3f} (delta-method se {se:.3f}); second derivative = 2*b1 = {2*vals[b1]:.3f} < 0")
print(f"revenue at p*: {float(TR.subs(vals).subs(p, ps)):.3f} $/household;  survey prices ranged ${a['ecoprc'].min():.2f}-${a['ecoprc'].max():.2f}")
```

Output:

```text
p* = (-b0 - b2*pr)/(2*b1)    d2TR/dp2 = 2*b1
revenue-maximizing eco price: $0.793 (delta-method se 0.042); second derivative = 2*b1 = -5.853 < 0
revenue at p*: 1.838 $/household;  survey prices ranged $0.59-$1.59
```

With linear demand, revenue is a concave quadratic in price (second derivative \(2\beta_1<0\)), so the first-order
condition gives the unique maximum: about 79 cents per pound, where the demand elasticity equals \(-1\) (compare
Exercise 1.2). The delta-method standard error of about 4 cents reflects uncertainty in the estimated demand curve. The
optimal price is inside the range of prices the survey randomized, so we are not extrapolating.

</details>

## Reading
Apostol, *Calculus*, Vol. 1; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.