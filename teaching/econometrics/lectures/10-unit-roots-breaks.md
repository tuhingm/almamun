# Lecture 10 — Unit Roots, Structural Breaks, and Persistence

## Objectives
Understand unit-root tests; distinguish deterministic components; explain break sensitivity; choose tests based on maintained alternatives.

## 1. Augmented Dickey–Fuller framework
A common test regression is
\[
\Delta y_t=\alpha+\delta t+\gamma y_{t-1}
+\sum_{j=1}^{p}\phi_j\Delta y_{t-j}+e_t.
\]
The null is \(\gamma=0\) (unit root), against a stationary alternative around the deterministic terms included. Critical values are nonstandard; ordinary t critical values do not apply.

## 2. Deterministic specification
The inclusion of an intercept and trend changes the test's alternative and power. A test without a trend is not interchangeable with a test including one. Lag length should absorb serial correlation without unnecessarily consuming degrees of freedom.

## 3. Complementary tests
The KPSS test reverses the null orientation by testing level- or trend-stationarity. Using complementary tests can clarify evidence, but neither test mechanically proves the true process.

## 4. Structural breaks
A break in mean or trend can cause conventional unit-root tests to have low power or misleading outcomes. Break-aware tests allow specified break structures; break dates and deterministic components should be justified and reported.

## Problem set
1. Interpret the ADF null and alternative.
2. Explain why ADF critical values are nonstandard.
3. Compare ADF and KPSS null hypotheses.
4. Explain how a level break can mimic persistence.
5. Specify a defensible unit-root testing protocol.

## Lab
Test macroeconomic series under intercept, trend, and break-aware specifications; report lag selection, deterministic terms, and sensitivity.

## Reading
Dickey and Fuller (1979, 1981); Kwiatkowski et al. (1992); Perron (1989); Hamilton, *Time Series Analysis*.