# Lecture 09 — Corporate Finance: Capital Budgeting and Project Valuation

**Level:** Graduate / Advanced Undergraduate | **Field:** Finance

## Learning objectives
Value investment projects using incremental cash flows; calculate NPV and IRR; apply WACC consistently; evaluate uncertainty and real options.

## 1. Net present value
For project cash flows \(CF_t\) and discount rate \(r\),
\[
NPV=\sum_{t=0}^{T}\frac{CF_t}{(1+r)^t}.
\]
Accept an independent project when NPV is positive, subject to financing and strategic constraints. NPV is value-additive under consistent valuation assumptions.

## 2. Incremental free cash flow
A typical unlevered project cash flow is
\[
FCF=EBIT(1-T)+D\&A-CapEx-\Delta NWC.
\]
Include opportunity costs and externalities within the firm, exclude sunk costs, and account for terminal value and asset disposal taxes.

## 3. IRR and ranking
IRR solves \(NPV(r)=0\). Multiple sign changes can generate multiple IRRs; mutually exclusive projects can be mis-ranked when scale or timing differs. Use incremental NPV or an explicitly justified equivalent method.

## 4. Cost of capital
WACC is
\[
WACC=\frac{E}{D+E}r_E+\frac{D}{D+E}r_D(1-T),
\]
under the conventional corporate-tax treatment of deductible interest. Match project risk, currency, inflation basis, and cash-flow definition to the discount rate.

## 5. Scenario and sensitivity analysis
Vary key drivers individually and jointly. Scenario analysis should preserve economically coherent dependencies; simulation can quantify the distribution of NPV. Real options arise when management can defer, expand, contract, or abandon under uncertainty.

## 6. Worked example
A project costs 100 today and generates 60 in each of the next two years. Calculate NPV at 10% and solve for IRR. Explain the decision.

## 7. Computational lab
Construct a project model with taxes, depreciation, working capital, and terminal value. Perform sensitivity analysis on price, volume, discount rate, and investment cost.

## Problem set
1. Derive project free cash flow.
2. Calculate NPV and IRR.
3. Explain an IRR ranking conflict.
4. State when WACC is an appropriate discount rate.

## Further reading
Berk and DeMarzo, *Corporate Finance*; Brealey, Myers and Allen, *Principles of Corporate Finance*.
