# Lecture 06 — Fixed-Income Securities, Bond Pricing, and Yield Curves

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Price fixed cash flows; distinguish coupon, spot, forward, and yield-to-maturity rates; understand credit spreads and discount curves.

## 1. Bond valuation
A default-free bond with cash flows \(C_t\) and maturity \(T\) has price
\[
P=\sum_{t=1}^{T}\frac{C_t}{(1+s_t)^t},
\]
where \(s_t\) are maturity-matched spot rates under annual compounding. For a zero-coupon bond, \(P_T=F/(1+s_T)^T\).

## 2. Yield to maturity
YTM is the single internal discount rate solving the bond price equation. It is a summary measure, not necessarily the realized return if coupons are reinvested at different rates or the bond is sold before maturity.

## 3. Spot and forward rates
Spot rates discount cash flows at each maturity. Forward rates are implied future borrowing rates from spot rates. Under annual compounding,
\[
(1+s_n)^n=(1+s_{n-1})^{n-1}(1+f_{n-1,n}).
\]

## 4. Price conventions and credit
Clean price excludes accrued interest; dirty price includes it. Credit spreads compensate for expected default losses, risk premia, liquidity, and other factors. Promised yield is not expected return when default is possible.

## 5. Worked example
A two-year bond pays 5 at year one and 105 at year two. Given spot rates \(s_1\) and \(s_2\), compute price as \(5/(1+s_1)+105/(1+s_2)^2\). Compare with price using a single YTM.

## 6. Computational lab
Build a cash-flow pricing function, bootstrap a spot curve from coupon bonds, calculate forward rates, and quantify sensitivity to curve shifts. Check compounding and day-count conventions.

## Problem set
1. Price a coupon bond from a spot curve.
2. Distinguish YTM from realized holding-period return.
3. Derive a one-period forward rate.
4. Explain clean and dirty prices.

## Further reading
Tuckman and Serrat, *Fixed Income Securities*; Fabozzi, *Bond Markets, Analysis, and Strategies*.
