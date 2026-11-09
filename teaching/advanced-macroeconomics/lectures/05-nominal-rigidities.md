# Lecture 05 — Nominal Rigidities, Calvo Pricing, and the New Keynesian Phillips Curve

**Level:** Graduate | **Field:** Advanced Macroeconomics

## Learning objectives
Derive the firm's reset-price problem under Calvo pricing; interpret the New Keynesian Phillips curve and its slope.

## 1. Price-setting environment
Each period, a fraction \(1-\theta\) of firms can reset prices; fraction \(\theta\) keep prices unchanged. A resetting firm chooses \(P_t^*\) to maximize expected discounted profits while its price may remain fixed for multiple periods.

## 2. Reset-price condition
The optimal reset price is a discounted markup over a weighted average of future nominal marginal costs. In log-linear form around a zero-inflation steady state, the aggregate pricing condition yields
\[
\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t,
\]
where \(\pi_t\) is inflation, \(x_t\) the output gap, and \(u_t\) a cost-push disturbance. Under the standard Calvo model,
\[
\kappa=\frac{(1-\theta)(1-\beta\theta)}{\theta}\times \text{real marginal-cost scaling}.
\]
The scaling depends on preferences and technology; it should not be silently set to one across parameterizations.

## 3. Economic interpretation
A larger \(\theta\) means fewer firms reset prices each period and typically a flatter Phillips curve. Expected future inflation matters because today's reset price is chosen for a potentially long spell. Marginal cost, rather than output itself, is the primitive pricing determinant; the output-gap representation uses a relation between marginal cost and the gap.

## 4. Rotemberg alternative
With quadratic price-adjustment costs, firms choose prices every period but pay a cost proportional to squared inflation. This yields a forward-looking Phillips curve with a slope governed by the adjustment-cost parameter. Calvo and Rotemberg models are alternative microfoundations, not identical mechanisms.

## 5. Worked example
For \(\beta=0.99\) and \(\theta=0.75\), compute the Calvo component \((1-\theta)(1-\beta\theta)/\theta\). Interpret how increasing \(\theta\) changes price flexibility.

## 6. Computational lab
Plot the Phillips curve slope over a grid of \(\theta\). Simulate inflation following a cost-push shock under alternative degrees of price rigidity.

## Problem set
1. Explain the reset-price firm's forward-looking choice.
2. Interpret each term in the NK Phillips curve.
3. Compare Calvo and Rotemberg pricing.
4. Explain why marginal cost is central to price-setting.

## Further reading
Galí, *Monetary Policy, Inflation, and the Business Cycle*; Woodford, *Interest and Prices*.
