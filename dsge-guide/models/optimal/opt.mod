// Optimal policy in the New Keynesian model. Run with -DPOL=1 (commitment), 2 (discretion) or 3 (optimal simple rule).
@#ifndef POL
  @#define POL = 1
@#endif
var x pi i u rn;
varexo e_u e_rn;
parameters beta sigma theta varphi eps kappa lambda_x phi_pi phi_x rho_u rho_rn;

beta = 0.99;  sigma = 1;  theta = 0.75;  varphi = 1;  eps = 6;
kappa = (1 - theta) * (1 - beta*theta) / theta * (sigma + varphi);
lambda_x = kappa / eps;                                     // welfare weight on the output gap
phi_pi = 1.5;  phi_x = 0.125;  rho_u = 0.5;  rho_rn = 0.8;

model(linear);
  x  = x(+1) - (1/sigma) * (i - pi(+1) - rn);               // IS curve
  pi = beta * pi(+1) + kappa * x + u;                       // Phillips curve
  u  = rho_u * u(-1) + e_u;                                 // cost-push shock
  rn = rho_rn * rn(-1) + e_rn;                              // natural-rate shock
@#if POL == 3
  i  = phi_pi * pi + phi_x * x;                             // simple rule, coefficients optimised below
@#endif
end;

shocks;
  var e_u;  stderr 0.2;
  var e_rn; stderr 0.5;
end;

@#if POL == 1
planner_objective pi^2 + lambda_x * x^2;
ramsey_model(planner_discount = 0.99, instruments = (i));
stoch_simul(order = 1, irf = 16, nograph) x pi i;
@#elseif POL == 2
planner_objective pi^2 + lambda_x * x^2;
discretionary_policy(instruments = (i), irf = 16, planner_discount = 0.99, nograph) x pi i;
@#else
optim_weights; pi 1; x lambda_x; end;
osr_params phi_pi phi_x;
osr_params_bounds; phi_pi, 1.01, 10; phi_x, 0, 5; end;
osr(opt_algo = 9, irf = 16, nograph) x pi i;
@#endif
