# Lecture 02 — Sequences, Convergence, and Infinite Series

## Learning objectives
Define convergence rigorously; use monotone convergence and Cauchy criteria; test infinite series; distinguish absolute and conditional convergence; analyze power series.

## 1. Sequences
A sequence \((a_n)\) converges to \(L\) if for every \(\varepsilon>0\), there exists \(N\) such that \(n\ge N\) implies \(|a_n-L|<\varepsilon\). Limits are unique and convergent sequences are bounded. A sequence is Cauchy if its terms become arbitrarily close to each other; completeness of \(\mathbb R\) ensures every Cauchy sequence converges.

## 2. Monotone convergence
Every monotone increasing sequence bounded above converges to its supremum. Every monotone decreasing sequence bounded below converges to its infimum. This result is useful for recursive sequences and iterative economic processes.

## 3. Infinite series
The series \(\sum_{n=1}^\infty a_n\) converges when its partial sums \(S_N=\sum_{n=1}^N a_n\) converge. A necessary condition is \(a_n\to0\), but this is not sufficient. The geometric series has sum \(a/(1-r)\) for \(|r|<1\).

## 4. Convergence tests
The p-series \(\sum n^{-p}\) converges iff \(p>1\). Use comparison and limit comparison for nonnegative terms. The ratio test uses \(L=\lim|a_{n+1}/a_n|\): convergence if \(L<1\), divergence if \(L>1\), and no conclusion if \(L=1\). The root test uses \(\limsup |a_n|^{1/n}\). Alternating-series convergence follows when positive terms decrease to zero.

## 5. Absolute and conditional convergence
Absolute convergence means \(\sum|a_n|\) converges and implies convergence of \(\sum a_n\). Conditional convergence occurs when the original series converges but its absolute-value series diverges. Rearrangement can change the sum of a conditionally convergent real series.

## 6. Power series
A power series \(\sum a_n(x-c)^n\) has radius of convergence \(R\). It converges absolutely for \(|x-c|<R\), diverges for \(|x-c|>R\), and endpoints must be checked separately. Termwise differentiation and integration are valid inside the interval of convergence.

## Worked example
The integral test compares \(\sum n^{-p}\) with \(\int_1^\infty x^{-p}dx\), finite exactly when \(p>1\). Thus the p-series converges iff \(p>1\).

## Problem set
1. Prove uniqueness of a sequence limit.
2. Apply monotone convergence to a bounded recursive sequence.
3. Test \(\sum 1/[n(n+1)]\) and find its sum.
4. Determine convergence of \(\sum(-1)^{n+1}/n\).
5. Find the radius and endpoint behavior of a power series.

## Computational lab
Plot partial sums for geometric, harmonic, p-series, and alternating harmonic series. Compare numerical behavior with analytic tests.

## Reading
Apostol, *Calculus*, Vol. 1; Spivak, *Calculus*; Rudin, *Principles of Mathematical Analysis*.