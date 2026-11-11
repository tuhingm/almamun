# Lecture 10 — The Zero Lower Bound and Unconventional Policy

**Level:** Graduate | **Field:** Monetary Economics

## Learning objectives
Analyse liquidity traps in the New Keynesian model, understand why the lower bound makes recessions deeper, and evaluate forward guidance, quantitative easing, negative rates and fiscal policy at the lower bound. Explain the forward-guidance puzzle and its resolutions.

## 1. The effective lower bound
Because cash pays a zero nominal return, the nominal interest rate cannot fall much below zero: \(i_t\ge i_{\text{ELB}}\approx0\). The bound binds when the natural real rate becomes very negative, for instance after a financial crisis or a collapse in demand. U.S. policy rates were at the bound in 2008–2015 and 2020–2022; the euro area, Japan, Switzerland and Sweden pushed policy rates slightly below zero. The secular decline in the neutral real rate (Lecture 7) makes the bound bind more often.

## 2. A liquidity trap in the New Keynesian model
Following Eggertsson and Woodford (2003), suppose the natural rate falls to \(r^n_L<0\) and stays there with probability \(\mu\) each period, returning permanently to \(r^*>0\) with probability \(1-\mu\). After the shock ends, policy delivers \(\pi=\tilde y=0\). While the shock lasts, \(i=0\) and the equilibrium is constant:
\[
\tilde y_L=\mu\tilde y_L+\tfrac1\sigma(\mu\pi_L+r^n_L),\qquad \pi_L=\beta\mu\pi_L+\kappa\tilde y_L ,
\]
which gives
\[
\tilde y_L=\frac{r^n_L/\sigma}{(1-\mu)-\dfrac{\mu\kappa}{\sigma(1-\beta\mu)}},\qquad \pi_L=\frac{\kappa}{1-\beta\mu}\tilde y_L .
\]
The denominator is positive but can be small: falling output lowers inflation, which raises the real rate, which lowers output further. Small changes in persistence or price flexibility generate very large recessions — a **deflationary spiral**. Paradoxically, more price flexibility (a larger \(\kappa\)) makes things worse (Eggertsson and Krugman, 2012). Fiscal multipliers are large at the bound because government spending raises expected inflation and lowers real rates (Christiano, Eichenbaum and Rebelo, 2011).

## 3. Forward guidance
The optimal commitment policy keeps rates at zero *after* the natural rate recovers and lets inflation overshoot. The promise raises expected inflation and output in the trap, lowering real rates today (Krugman, 1998; Eggertsson and Woodford, 2003). The policy is time inconsistent: when the trap ends, the central bank would prefer to raise rates. Central banks used **Delphic** guidance (forecasts of future policy) and **Odyssean** guidance (commitments), with calendar-based and threshold-based formulations (Campbell et al., 2012).

## 4. The forward-guidance puzzle
In the standard model, iterating the Euler equation forward gives \(\tilde y_t=-\frac1\sigma\sum_{k\ge0}E_t r_{t+k}\): a one-time cut in the real rate at any horizon \(T\) raises output today by the same amount, and through the Phillips curve the effect on today's inflation *grows* with \(T\). Promises about the distant future are implausibly powerful (Del Negro, Giannoni and Patterson, 2012). Resolutions include:

- **Discounted Euler equations** from incomplete markets and borrowing constraints, \(\tilde y_t=ME_t\tilde y_{t+1}-\frac1\sigma r_t\) with \(M<1\) (McKay, Nakamura and Steinsson, 2016);
- **Bounded rationality and cognitive discounting** (Gabaix, 2020);
- **Finite planning horizons** and level-\(k\) thinking (Woodford, 2019; Farhi and Werning, 2019);
- **Imperfect credibility** of promises.

## 5. Quantitative easing and balance-sheet policy
Large-scale asset purchases replace long-term bonds held by the private sector with reserves. In a frictionless world this is irrelevant (Wallace, 1981). With segmented markets or preferred habitats (Vayanos and Vila, 2021), purchases lower term premia; with constrained intermediaries (Gertler and Karadi, 2011), purchases of private assets ease credit spreads. Purchases also signal that rates will stay low (Bauer and Rudebusch, 2014). Event studies of the 2008–09 programmes find declines in 10-year yields of tens of basis points per programme (Gagnon et al., 2011), with smaller effects later. In 2022 the Fed began **quantitative tightening**, shrinking its balance sheet.

## 6. Negative rates and other tools
Negative policy rates in Europe and Japan (2014–2024) lowered market rates, but pass-through to deposit rates is limited, which may squeeze bank margins and lending (the "reversal rate", Brunnermeier and Koby, 2018). Other options include yield-curve control (Bank of Japan, 2016–2024; Reserve Bank of Australia, 2020–21), a higher inflation target (Blanchard, Dell'Ariccia and Mauro, 2010) and make-up strategies such as price-level targeting (Lecture 7).

## 7. Computation
The lower bound makes the model nonlinear. Dynare handles occasionally binding constraints with the **OccBin** toolkit (Guerrieri and Iacoviello, 2015), using the `occbin_constraints` block and `occbin_solver`, and solves deterministic paths with `perfect_foresight_setup` and `perfect_foresight_solver`. The two-state model of Section 2 and the perfect-foresight experiments of Section 4 have closed forms and are solved directly in MATLAB in the exercises.

## Problem set
1. Derive the liquidity-trap solution of Section 2 and the condition for a unique bounded equilibrium.
2. Explain why greater price flexibility deepens the recession at the lower bound.
3. Show that with the standard Euler equation, the effect of a promised real-rate cut on today's output is independent of the horizon.
4. Derive the discounted Euler equation in a simple model with idiosyncratic income risk and a borrowing constraint (sketch).
5. Compare the transmission channels of QE in the preferred-habitat model and in the Gertler–Karadi model.

## Coding exercises

These exercises use **MATLAB and Dynare only**. Download the course [data files and helper functions](../index.html#data-and-code)
into your working folder. Every data series is real: U.S. quarterly and monthly macro data (FRED-QD / FRED-MD,
1959–2025), published monetary-policy shock series (Romer–Romer, Jarociński–Karadi, Bu–Rogers–Wu,
Nakamura–Steinsson), World Bank CPI inflation for 190+ countries, and the KRW/USD exchange rate. Dynare models are
`.mod` files run with `dynare file noclearall` (Dynare 6). Try the starter files first, then open the solution. Every
printed result was produced by running the solution files as shown: the `.mod` files were processed by the Dynare
preprocessor and solved at first order, and the MATLAB code was run in GNU Octave.

### Exercise 10.1 — A liquidity trap and the value of forward guidance

*Data:* Model calibration: \(\beta=0.99,\ \sigma=1,\ \kappa=0.02\); the natural rate falls to −1% a quarter and stays there with probability \(\mu=0.8\) each quarter.

Follow Eggertsson and Woodford (2003). The natural real rate is \(r^n_L<0\) while the trap lasts and returns to
\(r^*=1/\beta-1\) with probability \(1-\mu\) each quarter, after which policy achieves \(\pi=\tilde y=0\). While the
trap lasts \(i=0\).

1. Show that in the trap \(\tilde y_L=\frac{r^n_L/\sigma}{(1-\mu)-\mu\kappa/[\sigma(1-\beta\mu)]}\) and
   \(\pi_L=\kappa\tilde y_L/(1-\beta\mu)\), and compute them.
2. **Forward guidance:** the central bank promises to keep \(i=0\) for one extra quarter after the natural rate
   recovers. In that exit quarter \(\tilde y_E=r^*/\sigma\) and \(\pi_E=\kappa\tilde y_E\). Recompute the trap
   outcomes.
3. How much of the recession does one quarter of forward guidance prevent? Why is the effect so large?

*File `ex10_1.m`*

```matlab
beta = 0.99; sigma = 1; kappa = 0.02; mu = 0.8; rL = -0.01; rstar = 1 / beta - 1;
% TODO: solve the two linear equations for (x_L, pi_L), first with x_E = pi_E = 0, then with forward guidance
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex10_1.m`*

```matlab
beta = 0.99; sigma = 1; kappa = 0.02; mu = 0.8; rL = -0.01; rstar = 1 / beta - 1;
% in the trap (i = 0):  x_L = mu x_L + (1-mu) x_E + (1/sigma)(mu pi_L + (1-mu) pi_E + rL)
%                       pi_L = kappa x_L + beta (mu pi_L + (1-mu) pi_E)
A = [1 - mu, -mu / sigma; -kappa, 1 - beta * mu];
fprintf('                         output gap (%%)   inflation (%%, annualised)\n');
for fg = [0 1]
  xE = fg * rstar / sigma; piE = kappa * xE;                % exit quarter with i = 0 instead of r*
  rhs = [(1 - mu) * xE + ((1 - mu) * piE + rL) / sigma; beta * (1 - mu) * piE];
  sol = A \ rhs;
  if fg == 0, nm = 'no forward guidance  '; base = sol; else, nm = '1 qtr of guidance    '; end
  fprintf('%s      %7.2f            %7.2f\n', nm, 100 * sol(1), 400 * sol(2));
end
fprintf('expected trap length: %.1f quarters; guidance removes %.0f%% of the output loss\n', 1 / (1 - mu), ...
        100 * (1 - sol(1) / base(1)));
```

Output:

```text
output gap (%)   inflation (%, annualised)
no forward guidance          -8.13              -3.13
1 qtr of guidance            -6.33              -2.36
expected trap length: 5.0 quarters; guidance removes 22% of the output loss
```

With a natural rate of −4% a year and an expected trap of five quarters, the zero bound turns a moderate shock into
a deep recession with deflation. Falling inflation raises the real rate when it should fall, which lowers output,
which lowers inflation further. This feedback is why the denominator can approach zero (a "deflationary spiral"). A
promise to hold the rate at zero for a single additional quarter, once the shock is over, raises expected output and
inflation in every trap state; higher expected inflation lowers today's real rate, and the improvement compounds
through the same loop. In this calibration that single quarter removes about a fifth of the output loss. The catch is
credibility: when the trap ends, the central bank is tempted to renege (Lecture 6).

</details>

### Exercise 10.2 — The forward-guidance puzzle

*Data:* Model calibration: \(\sigma=1\), \(\kappa=0.1717\), \(\beta=0.99\).

Under perfect foresight, suppose the central bank announces at \(t=0\) that the real rate will be 25 bp lower in
quarter \(T\) only. With the standard Euler equation \(\tilde y_t=\tilde y_{t+1}-\sigma^{-1}r_t\) and Phillips curve
\(\pi_t=\beta\pi_{t+1}+\kappa\tilde y_t\), solve backwards from \(\tilde y=\pi=0\) after \(T\).

1. Compute \(\tilde y_0\) and \(\pi_0\) for \(T\in\{0,4,8,20,40\}\).
2. Repeat with the "discounted" Euler equation of McKay, Nakamura and Steinsson (2016),
   \(\tilde y_t=M\tilde y_{t+1}-\sigma^{-1}r_t\) with \(M=0.97\).
3. Why does the standard model imply that distant promises are so powerful? What resolves the puzzle?

*File `ex10_2.m`*

```matlab
sigma = 1; kappa = 0.1717; beta = 0.99; dr = -0.25;
% TODO: for each T and M in {1, 0.97}: backward recursion over t = T, T-1, ..., 0
```

<details class="solution" markdown="1"><summary>Solution, output and discussion</summary>

*File `ex10_2.m`*

```matlab
sigma = 1; kappa = 0.1717; beta = 0.99; dr = -0.25;
fprintf('   T    standard Euler: x0    pi0 (ann.)   |   discounted Euler (M = 0.97): x0   pi0 (ann.)\n');
for T = [0 4 8 20 40]
  out = zeros(1, 4);
  for j = 1:2
    M = 1 - 0.03 * (j == 2); x = 0; p = 0;
    for t = T:-1:0
      r = dr * (t == T);
      x = M * x - r / sigma; p = beta * p + kappa * x;
    end
    out(2*j-1:2*j) = [x 4 * p];
  end
  fprintf('%4d        %6.3f      %7.3f      |                    %6.3f      %7.3f\n', T, out);
end
```

Output:

```text
T    standard Euler: x0    pi0 (ann.)   |   discounted Euler (M = 0.97): x0   pi0 (ann.)
 0         0.250        0.172      |                     0.250        0.172
 4         0.250        0.842      |                     0.221        0.792
 8         0.250        1.485      |                     0.196        1.316
20         0.250        3.267      |                     0.136        2.423
40         0.250        5.799      |                     0.074        3.223
```

With the standard Euler equation the output gap today rises by the same 0.25% whatever the horizon of the
promise, because today's output depends on the sum of all future real rates. Inflation today rises more the more
distant the promise, because the higher output path accumulates in the forward-looking Phillips curve: a promise
ten years out has a far larger effect on today's inflation than an immediate cut. That is the forward-guidance
puzzle (Del Negro, Giannoni and Patterson, 2012). With the discounted Euler equation, which arises with incomplete
markets and borrowing constraints (McKay, Nakamura and Steinsson, 2016) or bounded rationality, the effect of distant
promises dies out. Heterogeneous-agent models (Lecture 12) give a microfoundation.

</details>

## Further reading
- Eggertsson, G. and M. Woodford (2003), "The Zero Bound on Interest Rates and Optimal Monetary Policy," *Brookings Papers on Economic Activity*.
- Christiano, L., M. Eichenbaum and S. Rebelo (2011), "When Is the Government Spending Multiplier Large?" *Journal of Political Economy*.
- McKay, A., E. Nakamura and J. Steinsson (2016), "The Power of Forward Guidance Revisited," *American Economic Review*.
- Bernanke, B. (2020), "The New Tools of Monetary Policy," *American Economic Review*.
- Guerrieri, L. and M. Iacoviello (2015), "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily," *Journal of Monetary Economics*.
