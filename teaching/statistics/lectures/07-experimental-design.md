# Lecture 07 — Experimental Design, Randomization, and Blocking

## Learning objectives
Define experimental units and treatments; explain randomization, blocking, replication, factorial designs, and design-based inference.

## 1. Design before analysis
A design specifies units, treatments, assignment mechanism, outcomes, and estimand. Randomization creates a known assignment mechanism and supports design-based causal comparisons. It does not guarantee balance in every realized sample.

## 2. Completely randomized design
For treatment groups, random assignment makes treatment independent of potential outcomes in expectation under the design. Estimate treatment contrasts and use inference consistent with the assignment mechanism.

## 3. Randomized block design
\[
Y_{ij}=\mu+\tau_i+\beta_j+\varepsilon_{ij}.
\]
Blocks group units with similar nuisance characteristics. Blocking can improve precision when within-block variation is lower than across-block variation. Treatment must be randomized within blocks according to the design.

## 4. Factorial designs
A two-factor model includes main effects and interaction:
\[
Y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}.
\]
An interaction means the effect of one factor depends on the level of another. Main effects must be interpreted in light of the interaction and coding scheme.

## 5. Nested and split-plot structures
Nested factors occur when levels of one factor exist only within levels of another. Split-plot designs have different randomization units and error strata; analyzing them as completely randomized designs can produce incorrect standard errors.

## Problem session
1. Explain why randomization identifies average treatment contrasts.
2. Derive the randomized-block ANOVA decomposition.
3. Interpret a two-factor interaction.
4. Distinguish nested and crossed factors.
5. Identify the experimental unit in a split-plot design.

## Reading
Montgomery, *Design and Analysis of Experiments*; Box, Hunter and Hunter, *Statistics for Experimenters*.