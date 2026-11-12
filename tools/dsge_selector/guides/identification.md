title: Identification Before Estimation
short: Identification
when: Estimate with data: check which parameters the data can pin down
data: The An–Schorfheide New Keynesian model of the estimation guide
lead: Use this guide before estimating any DSGE model. A parameter is identified only if changing it changes what the model implies for the observables. If two parameters always enter together, no amount of data can separate them, and a Bayesian posterior will simply return the prior. Dynare's identification command checks this at the prior, before a single draw is taken.
---

## The question and why it comes first

**Which parameters can the data actually tell us about?** In a DSGE model the observables depend on the parameters through the solution of the model, and some parameters may have no separate effect: they enter only in a combination, or their effect is indistinguishable from another parameter's. Such parameters are **not identified**. Estimating them anyway gives posteriors that mirror the priors, and conclusions that are really assumptions.

Iskrev (2010) and Komunjer and Ng (2011) showed how to check identification from the model alone: a parameter is (locally) identified if the moments, or the spectrum, of the observables change when the parameter changes, in a way that no combination of other parameters can replicate. Dynare implements both checks in the `identification` command.

<div class="read" markdown="1">
**Two kinds of problems**

- **Non-identification**: a parameter has no effect on the observables, or two parameters enter only as a product or ratio. The Jacobian of the moments with respect to the parameters is rank-deficient.
- **Weak identification**: the parameter is identified in principle, but its effect is small or nearly collinear with another's, so the data are only weakly informative. The Jacobian is close to rank-deficient.
</div>

## Check the estimation model

```dynare as_id.mod from="REDUCED-FORM" max=22
// The An-Schorfheide model of the estimation guide, checked for identification at the prior mean.
var y pi R g z YGR INFL INT;
varexo e_R e_g e_z;
parameters tau kappa psi1 psi2 rho_R rho_g rho_z r_A;
tau = 2;  kappa = 0.2;  psi1 = 1.5;  psi2 = 0.5;  rho_R = 0.5;  rho_g = 0.8;  rho_z = 0.66;  r_A = 2;
model(linear);
  # beta = 1 / (1 + r_A/400);
  y  = y(+1) - (1/tau) * (R - pi(+1) - z(+1)) + g - g(+1);
  pi = beta * pi(+1) + kappa * (y - g);
  R  = rho_R * R(-1) + (1 - rho_R) * (psi1*pi + psi2*(y - g)) + e_R;
  g  = rho_g * g(-1) + e_g;  z = rho_z * z(-1) + e_z;
  YGR = y - y(-1) + z;  INFL = 4 * pi;  INT = 4 * R;
end;
shocks; var e_R; stderr 0.4; var e_g; stderr 1; var e_z; stderr 0.5; end;
steady;
estimated_params;
  tau, gamma_pdf, 2.00, 0.50;   kappa, gamma_pdf, 0.20, 0.10;
  psi1, gamma_pdf, 1.50, 0.25;  psi2, gamma_pdf, 0.50, 0.25;
  rho_R, beta_pdf, 0.50, 0.20;  rho_g, beta_pdf, 0.80, 0.10;  rho_z, beta_pdf, 0.66, 0.15;
  stderr e_R, inv_gamma_pdf, 0.4, 4;  stderr e_g, inv_gamma_pdf, 1.0, 4;  stderr e_z, inv_gamma_pdf, 0.5, 4;
end;
varobs YGR INFL INT;
identification(parameter_set = prior_mean, nograph);
```

<div class="read" markdown="1">
**How to read it**

Dynare runs three tests, and they do not all agree:

- **Komunjer and Ng (2011)**, the test on the model's minimal state-space system, finds **all parameters identified**.
- **The reduced-form test (Iskrev 2010) and the spectrum test (Qu and Tkachenko 2012)** flag the **policy-rule block** (\(\psi_1\), \(\psi_2\), \(\rho_R\) and the standard deviation of the monetary shock) as collinear with the other parameters: their effects on the observables can be closely mimicked by combinations of other parameters.

This is not a quirk of the code. Cochrane (2011) argued that Taylor-rule coefficients are hard to identify in New Keynesian models precisely because, under determinacy, the equilibrium never shows the off-equilibrium responses those coefficients govern. In practice it means the data will be only weakly informative about the policy rule, and its posterior will lean on the prior: compare the prior and posterior of \(\psi_1\) in the estimation guide before interpreting it. The other parameters are cleanly identified.
</div>

## A model that is not identified

```dynare as_bad.mod norun
// The same model, but the Phillips-curve slope is written in terms of two deep parameters,
// the Calvo probability theta and the inverse Frisch elasticity varphi, and both are estimated.
var y pi R g z YGR INFL INT;
varexo e_R e_g e_z;
parameters tau theta varphi psi1 psi2 rho_R rho_g rho_z r_A;
tau = 2;  theta = 0.75;  varphi = 1;  psi1 = 1.5;  psi2 = 0.5;  rho_R = 0.5;  rho_g = 0.8;  rho_z = 0.66;  r_A = 2;
model(linear);
  # beta = 1 / (1 + r_A/400);
  # kappa = (1 - theta) * (1 - beta*theta) / theta * (tau + varphi);
  y  = y(+1) - (1/tau) * (R - pi(+1) - z(+1)) + g - g(+1);
  pi = beta * pi(+1) + kappa * (y - g);
  R  = rho_R * R(-1) + (1 - rho_R) * (psi1*pi + psi2*(y - g)) + e_R;
  g  = rho_g * g(-1) + e_g;  z = rho_z * z(-1) + e_z;
  YGR = y - y(-1) + z;  INFL = 4 * pi;  INT = 4 * R;
end;
shocks; var e_R; stderr 0.4; var e_g; stderr 1; var e_z; stderr 0.5; end;
steady;
estimated_params;
  tau, gamma_pdf, 2.00, 0.50;   theta, beta_pdf, 0.75, 0.05;   varphi, gamma_pdf, 1.00, 0.25;
  psi1, gamma_pdf, 1.50, 0.25;  psi2, gamma_pdf, 0.50, 0.25;
  rho_R, beta_pdf, 0.50, 0.20;  rho_g, beta_pdf, 0.80, 0.10;  rho_z, beta_pdf, 0.66, 0.15;
  stderr e_R, inv_gamma_pdf, 0.4, 4;  stderr e_g, inv_gamma_pdf, 1.0, 4;  stderr e_z, inv_gamma_pdf, 0.5, 4;
end;
varobs YGR INFL INT;
identification(parameter_set = prior_mean, nograph);
```

```octave from="REDUCED-FORM" max=24
clear functions;
dynare as_bad noclearall nograph nolog console
```

<div class="read" markdown="1">
**How to read it**

The Phillips curve depends on the Calvo parameter \(\theta\) and the labour-supply parameter \(\varphi\) only through the slope \(\kappa\). Many combinations of \(\theta\) and \(\varphi\) give the same \(\kappa\), and therefore exactly the same predictions for output, inflation and interest rates. Now **all three tests** fail, including Komunjer–Ng, and each names the same culprit: **[varphi, theta] are pairwise collinear**. Unlike the weak identification of the policy rule, this is exact non-identification.

This is a typical situation, not a contrived one. Small models usually identify the slope of the Phillips curve but not the structural parameters behind it. The remedies are:

- **estimate the reduced-form coefficient** (\(\kappa\)) and calibrate one of the structural parameters;
- **add data** that the other parameter affects separately (hours or wages identify \(\varphi\));
- **use informative priors** from micro evidence, and say clearly that the posterior for that parameter reflects the prior.
</div>

## Write the paper

<div class="write" markdown="1">
**How to write it: identification**

Before estimation, we verify local identification of the estimated parameters at the prior mean using the rank conditions of Iskrev (2010) and Komunjer and Ng (2011), as implemented in Dynare. The Komunjer–Ng test indicates that all estimated parameters are locally identified, although the reduced-form and spectral tests point to weak identification of the policy-rule parameters, a known feature of New Keynesian models (Cochrane 2011); we therefore interpret their posteriors with reference to the priors. In an alternative parameterisation in which the Calvo probability and the inverse Frisch elasticity are estimated separately, the two are not separately identified, because they enter the model only through the slope of the Phillips curve; we therefore estimate the slope directly.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Estimating first and checking identification only when a posterior looks like its prior.
- Taking a well-behaved posterior as evidence of identification: a tight prior produces a tight posterior even for an unidentified parameter.
- Checking identification at one parameter value only. Dynare can check it across draws from the prior (`prior_mc` option), which reveals regions where it fails.
- Estimating more structural parameters than the observables can inform, instead of estimating reduced-form coefficients.
</div>
