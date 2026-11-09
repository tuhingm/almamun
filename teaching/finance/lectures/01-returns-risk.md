# Lecture 01 — Returns, Risk, and the Foundations of Asset Pricing

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Measure simple and log returns; distinguish arithmetic and geometric performance; quantify risk and explain the role of stochastic discounting.

## 1. Return measurement
For price \(P_t\) and cash distribution \(D_t\), the one-period simple total return is
\[
R_{t+1}=\frac{P_{t+1}+D_{t+1}-P_t}{P_t}.
\]
The gross return is \(1+R_{t+1}\), while the log return is \(r_{t+1}=\log(1+R_{t+1})\). Simple returns compound multiplicatively; log returns add over time. The approximation \(r\approx R\) is accurate only for small returns.

## 2. Arithmetic and geometric averages
The arithmetic mean estimates average one-period return. The geometric mean
\[
R_G=\left[\prod_{t=1}^T(1+R_t)\right]^{1/T}-1
\]
is the constant per-period return producing the same cumulative wealth. Volatility generally makes geometric growth lower than arithmetic average return.

## 3. Risk and covariance
For return vector \(R\), expected return is \(\mu=E(R)\), covariance matrix is \(\Sigma=E[(R-\mu)(R-\mu)']\), and portfolio variance is \(w'\Sigma w\). Correlation is scale-free dependence; covariance retains units.

## 4. Stochastic discount factor
A fundamental pricing relation is
\[
P_t=E_t[M_{t+1}X_{t+1}],
\]
where \(M_{t+1}\) is the stochastic discount factor (SDF) and \(X_{t+1}\) the payoff. For a gross return, \(1=E_t[M_{t+1}R_{t+1}]\). Risk premia depend on covariance with the SDF, not simply on standalone volatility.

## 5. Worked example
An asset rises from 100 to 108 and pays a distribution of 2. Its total simple return is 10%. Calculate its log return and compare the two measures.

## 6. Computational lab
Load adjusted price data, compute simple and log returns, annualize means and volatility under an explicitly stated frequency assumption, and compare arithmetic and geometric annualized performance. Check missing dates and corporate-action treatment.

## Problem set
1. Derive the total return formula.
2. Show that multi-period log returns add.
3. Explain why geometric mean is below arithmetic mean in volatile samples.
4. Interpret the SDF pricing equation.

## Further reading
Cochrane, *Asset Pricing*; Bodie, Kane and Marcus, *Investments*.
