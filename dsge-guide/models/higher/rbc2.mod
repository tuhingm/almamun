// RBC model with a welfare recursion, solved at second (or third) order with pruning: dynare rbc2 -DORD=3
@#ifndef ORD
  @#define ORD = 2
@#endif
var y c i k h a W;
varexo e;
parameters beta alpha delta phi psi rho sigma_e h_ss;

beta = 0.99;  alpha = 0.33;  delta = 0.025;  phi = 1;  rho = 0.95;  sigma_e = 0.007;  h_ss = 1/3;

model;
  exp(-c) = beta * exp(-c(+1)) * (alpha * exp(y(+1) - k) + 1 - delta);
  psi * exp(phi*h) * exp(c) = (1 - alpha) * exp(y - h);
  exp(y) = exp(a) * exp(alpha*k(-1)) * exp((1 - alpha)*h);
  exp(k) = (1 - delta) * exp(k(-1)) + exp(i);
  exp(y) = exp(c) + exp(i);
  a = rho * a(-1) + e;
  W = c - psi * exp((1 + phi)*h) / (1 + phi) + beta * W(+1);       // lifetime utility (log C = c)
end;

steady_state_model;
  r = 1/beta - 1 + delta;
  k_h = (alpha / r)^(1/(1 - alpha));
  h = log(h_ss);  k = log(k_h * h_ss);  y = log(k_h^alpha * h_ss);
  i = log(delta * exp(k));  c = log(exp(y) - exp(i));
  psi = (1 - alpha) * exp(y) / h_ss / (h_ss^phi * exp(c));
  a = 0;
  W = (c - psi * h_ss^(1 + phi) / (1 + phi)) / (1 - beta);
end;

shocks;
  var e; stderr sigma_e;
end;

steady;
check;
stoch_simul(order = @{ORD}, pruning, irf = 0, periods = 0) y c k h W;
