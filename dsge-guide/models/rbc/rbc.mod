// Real business cycle model. Variables are in logs, so impulse responses are percentage deviations.
var y c i k h w r a;
varexo e;
parameters beta alpha delta phi psi rho sigma_e h_ss;

beta = 0.99;  alpha = 0.33;  delta = 0.025;  phi = 1;
rho = 0.95;   sigma_e = 0.007;  h_ss = 1/3;

model;
  exp(-c) = beta * exp(-c(+1)) * (alpha * exp(y(+1) - k) + 1 - delta);    // Euler equation (k is K_{t+1})
  psi * exp(phi*h) * exp(c) = exp(w);                                     // labour supply
  exp(w) = (1 - alpha) * exp(y - h);                                      // wage = marginal product of labour
  exp(r) = alpha * exp(y - k(-1));                                        // rental rate = marginal product of capital
  exp(y) = exp(a) * exp(alpha*k(-1)) * exp((1 - alpha)*h);                // production
  exp(k) = (1 - delta) * exp(k(-1)) + exp(i);                             // capital accumulation
  exp(y) = exp(c) + exp(i);                                               // resource constraint
  a = rho * a(-1) + e;                                                    // productivity
end;

steady_state_model;
  // analytical steady state: hours are fixed at h_ss and psi is backed out
  r = log(1/beta - 1 + delta);
  KY = alpha / exp(r);
  k_h = KY^(1/(1 - alpha));
  h = log(h_ss);
  k = log(k_h * h_ss);
  y = log(k_h^alpha * h_ss);
  i = log(delta * exp(k));
  c = log(exp(y) - exp(i));
  w = log((1 - alpha) * exp(y) / h_ss);
  psi = exp(w) / (h_ss^phi * exp(c));
  a = 0;
end;

shocks;
  var e; stderr sigma_e;
end;

steady;
check;
stoch_simul(order=1, irf=40, hp_filter=1600, periods=0) y c i h w;
