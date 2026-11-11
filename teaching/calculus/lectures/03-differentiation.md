# Lecture 03 — Differentiation, Mean Value Theorems, and Comparative Rates

## Learning objectives
Derive derivatives from first principles; apply differentiation rules; use mean value theorems; interpret marginal effects, elasticities, and comparative rates.

## 1. Derivative definition
\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},
\]
when the limit exists. It is the local linear approximation slope. Differentiability implies continuity, but continuity does not imply differentiability.

## 2. Differentiation rules
Derive sum, product, quotient, and chain rules from the limit definition. For compositions, \((f\circ g)'(x)=f'(g(x))g'(x)\). Exponential and logarithmic derivatives follow from their defining properties. Implicit differentiation applies the chain rule to equations defining variables jointly.

## 3. Higher derivatives and curvature
The second derivative measures the change in slope. If \(f''>0\) on an interval, the function is strictly convex there; if \(f''<0\), it is strictly concave. An inflection point requires a change in concavity, not merely a zero second derivative.

## 4. Mean value theorems
If \(f\) is continuous on \([a,b]\) and differentiable on \((a,b)\), there exists \(c\in(a,b)\) such that
\[
f'(c)=\frac{f(b)-f(a)}{b-a}.
\]
Rolle's theorem is the special case \(f(a)=f(b)\). These results establish monotonicity tests and uniqueness arguments.

## 5. Economic derivatives
Marginal cost is \(C'(q)\), marginal revenue is \(R'(q)\), and elasticity is \(\varepsilon_{y,x}=(dy/dx)(x/y)\). For positive variables, \(d\log y/d\log x\) equals elasticity. Interpret derivatives only over the relevant domain.

## Worked example
For \(C(q)=F+cq+dq^2\), marginal cost is \(MC=c+2dq\), increasing when \(d>0\). Average cost is \(F/q+c+dq\), which can have an interior minimum.

## Problem set
1. Derive the derivative of \(x^2\) from first principles.
2. Differentiate \(x^2e^{3x}\) and \(\log(1+x^2)\).
3. Prove that a differentiable function with zero derivative on an interval is constant.
4. Apply the mean value theorem to establish monotonicity.
5. Derive elasticity for a power function.

## Computational lab
Use symbolic differentiation and finite differences. Plot approximation error as step size changes and discuss round-off versus truncation error.

## Coding exercises

These exercises apply the calculus of this lecture to **real economic data** in Python, mixing symbolic (`sympy`),
numerical (`scipy`) and statistical (`statsmodels`) tools. Install once with
`pip install sympy scipy numpy pandas statsmodels wooldridge`. Try the starter code first, then open the solution.
Every printed result below comes from running the solution code exactly as shown.

### Exercise 3.1 — Marginal effects and elasticities from a fitted R&D model

*Data:* `rdchem` — 32 U.S. chemical firms: R&D spending `rd` and `sales` (millions of dollars).

1. Fit \(rd=\beta_0+\beta_1 sales+\beta_2 sales^2+u\).
2. Differentiate the fitted function symbolically with `sympy` to get the marginal effect \(d\,rd/d\,sales\) and the elasticity \(\frac{d\,rd}{d\,sales}\frac{sales}{rd}\).
3. Evaluate both at the median firm and at the largest firm. Where does the marginal effect turn negative, and is that inside the data?
4. Check the symbolic derivative against a central finite difference.

```python
import sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("rdchem")
r = smf.ols("rd ~ sales + I(sales**2)", data=d).fit()
S = sp.symbols("S")
f = r.params["Intercept"] + r.params["sales"] * S + r.params["I(sales ** 2)"] * S**2
# TODO: sp.diff(f, S); elasticity; solve df/dS = 0
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, sympy as sp, wooldridge as wd, statsmodels.formula.api as smf
d = wd.data("rdchem")
r = smf.ols("rd ~ sales + I(sales**2)", data=d).fit()
S = sp.symbols("S", positive=True)
f = r.params["Intercept"] + r.params["sales"] * S + r.params["I(sales ** 2)"] * S**2
fp = sp.diff(f, S); el = sp.simplify(fp * S / f)
print("fitted rd(S) =", sp.N(f, 4)); print("rd'(S) =", sp.N(fp, 4))
for lab, s in (("median firm", d["sales"].median()), ("largest firm", d["sales"].max())):
    print(f"  {lab:12s} (sales {s:,.0f}): marginal effect {float(fp.subs(S, s)):.4f}, elasticity {float(el.subs(S, s)):.3f}")
peak = float(sp.solve(sp.Eq(fp, 0), S)[0])
print(f"marginal effect turns negative at sales = {peak:,.0f}; firms above it: {(d['sales'] > peak).sum()} of {len(d)}")
fn = sp.lambdify(S, f); h = 1e-3; s0 = d["sales"].median()
print(f"finite difference at median: {(fn(s0 + h) - fn(s0 - h)) / (2*h):.6f}  vs symbolic {float(fp.subs(S, s0)):.6f}")
```

Output:

```text
fitted rd(S) = -6.538e-7*S**2 + 0.06391*S - 43.1
rd'(S) = 0.06391 - 1.308e-6*S
  median firm  (sales 1,332): marginal effect 0.0622, elasticity 2.025
  largest firm (sales 39,709): marginal effect 0.0120, elasticity 0.325
marginal effect turns negative at sales = 48,879; firms above it: 0 of 32
finite difference at median: 0.062169  vs symbolic 0.062169
```

The fitted R&D function is slightly concave. The marginal effect of \$1 of sales on R&D falls from about 6 cents at the
median firm to about 1 cent at the largest firm, and the elasticity falls from about 2 to 0.3. The marginal effect would
turn negative only at sales of about \$49 billion, beyond every firm in the data — so the "turning point" is an artefact of
the quadratic, not a finding. The symbolic derivative agrees with the finite difference to six decimals. The elasticity
above 1 at the median is driven by the negative intercept and should not be over-interpreted with 32 firms.

</details>

### Exercise 3.2 — Newton's method for an internal rate of return

*Data:* `macrodata` — annual averages of U.S. real GDP, 2000–2008, used to give a stylized project realistic cash flows that grow with the economy.

A project costs \$1,000 today and returns cash flows \(C_1,\dots,C_9\) proportional to annual U.S. real GDP growth paths. The IRR solves \(NPV(r)=-1000+\sum_t C_t/(1+r)^t=0\).

1. Derive \(NPV'(r)\) analytically.
2. Implement Newton's method \(r_{k+1}=r_k-NPV(r_k)/NPV'(r_k)\) from \(r_0=0\). Print each iteration and show quadratic convergence (the number of correct digits roughly doubles).
3. Check with `scipy.optimize.brentq`.

```python
import numpy as np, statsmodels.api as sm
m = sm.datasets.macrodata.load_pandas().data
y = m.groupby("year")["realgdp"].mean().loc[2000:2008].to_numpy()
C = 150 * y / y[0]                       # cash flows that grow with GDP
npv = lambda r: -1000 + sum(c / (1 + r)**t for t, c in enumerate(C, 1))
# TODO: dnpv = lambda r: -sum(t * c / (1 + r)**(t+1) ...); Newton loop
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

```python
import numpy as np, statsmodels.api as sm
from scipy import optimize
m = sm.datasets.macrodata.load_pandas().data
y = m.groupby("year")["realgdp"].mean().loc[2000:2008].to_numpy()
C = 150 * y / y[0]
npv = lambda r: -1000 + sum(c / (1 + r)**t for t, c in enumerate(C, 1))
dnpv = lambda r: -sum(t * c / (1 + r)**(t + 1) for t, c in enumerate(C, 1))
print("cash flows:", C.round(1))
root = optimize.brentq(npv, -0.5, 1.0, xtol=1e-15)
rk = 0.0
for k in range(6):
    print(f"iteration {k}: r = {rk:.15f}   error = {abs(rk - root):.2e}")
    rk = rk - npv(rk) / dnpv(rk)
print(f"brentq IRR = {root:.15f}")
```

Output:

```text
cash flows: [150.  151.6 154.4 158.2 163.9 168.9 173.4 177.1 177.9]
iteration 0: r = 0.000000000000000   error = 8.27e-02
iteration 1: r = 0.062431867426432   error = 2.02e-02
iteration 2: r = 0.081358309440441   error = 1.30e-03
iteration 3: r = 0.082656305240854   error = 5.53e-06
iteration 4: r = 0.082661833017955   error = 9.96e-11
iteration 5: r = 0.082661833117509   error = 8.33e-17
brentq IRR = 0.082661833117509
```

Newton's method converges to the IRR of about 8.27% in five iterations. The errors fall from \(10^{-2}\) to \(10^{-3}\),
\(10^{-6}\), \(10^{-11}\) and machine precision: the number of correct digits roughly doubles each step, which is
quadratic convergence. It works so well here because \(NPV(r)\) is smooth, decreasing and convex. With cash flows that
change sign more than once, NPV can have several roots and Newton's method can converge to any of them, or fail;
bracketing methods such as Brent's are safer.

</details>

## Reading
Apostol, *Calculus*, Vol. 1; Chiang and Wainwright, *Fundamental Methods of Mathematical Economics*; Simon and Blume, *Mathematics for Economists*.