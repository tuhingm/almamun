# Lecture 07 — Experimental Design, Randomization, and Blocking

## Learning objectives
Define experimental units, treatments, outcomes, and estimands; explain randomization, blocking, replication, factorial and split-plot designs; connect assignment mechanisms to design-based inference.

## 1. Potential outcomes and estimands
For unit \(i\), define potential outcomes \(Y_i(1)\) and \(Y_i(0)\). The individual effect is \(\tau_i=Y_i(1)-Y_i(0)\), and the finite-population average effect is \(N^{-1}\sum_i\tau_i\). Only one potential outcome is observed per unit. The design specifies the unit of assignment, treatment, outcome, target population, and assignment mechanism.

## 2. Completely randomized design
Under complete random assignment, treatment status is independent of fixed potential outcomes over the assignment mechanism. The difference in treatment and control means estimates the finite-population average effect. Randomization supports unbiasedness over assignments but does not guarantee exact covariate balance in the realized sample.

## 3. Randomized blocks
When pre-treatment characteristics predict outcomes, randomize within blocks. A model representation is
\[
Y_{ij}=\mu+\tau_i+\beta_j+\varepsilon_{ij}.
\]
Blocking can reduce residual variance when blocks explain outcome variation. Define blocks before observing treatment outcomes.

## 4. Factorial designs
A two-factor design can be represented as
\[
Y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}.
\]
The interaction means the effect of one factor depends on the level of the other. Interpret main effects in light of the interaction and coding constraints.

## 5. Split-plot and nested designs
Difficult-to-randomize factors may be assigned at a whole-plot level, with other factors randomized within plots. Whole-plot and subplot errors differ; treating all observations as independently randomized gives incorrect standard errors. Nested designs likewise require covariance structures consistent with assignment and sampling.

## 6. Design-based inference
A randomization test reassigns treatment labels according to the original design. Under a sharp null of no effect for any unit, this can provide exact inference. For heterogeneous effects and average-effect targets, specify the estimand and variance estimator carefully.

## Worked example
If clinic strongly predicts the outcome, allocate treatment and control units within each clinic rather than randomizing without regard to clinic. The blocked contrast compares units within pre-specified strata and can improve precision.

## Problem set
1. Define the experimental unit and estimand in a cluster trial.
2. Explain why randomization does not ensure exact balance.
3. Derive a treatment contrast in a randomized block design.
4. Interpret a two-factor interaction.
5. Explain why split-plot designs require distinct error terms.

## Computational lab
Simulate complete and blocked randomization under strong block-level outcome heterogeneity. Compare estimator variance and conduct a randomization test under the sharp null.

## Reading
Montgomery, *Design and Analysis of Experiments*; Fisher, *The Design of Experiments*; Imbens and Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*.