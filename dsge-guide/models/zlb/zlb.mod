// New Keynesian model with the zero lower bound, solved with OccBin. Quarterly percent deviations.
@#ifndef ILB
  @#define ILB = -1.0
@#endif
var x pi i rn inot;
varexo e_rn;
parameters beta sigma kappa phi_pi phi_x rho_rn ilb;

beta = 0.995;  sigma = 1;  kappa = 0.1717;  phi_pi = 1.5;  phi_x = 0.125;  rho_rn = 0.8;
ilb = @{ILB};                                   // the bound: minus the steady-state nominal rate (1% a quarter)

model;
  x = x(+1) - (1/sigma) * (i - pi(+1) - rn);    // IS curve
  pi = beta * pi(+1) + kappa * x;               // Phillips curve
  inot = phi_pi * pi + phi_x * x;               // the rate the Taylor rule prescribes (the shadow rate)
  [name = 'policy', relax = 'zlb']
  i = inot;                                     // regime 1: the rule applies
  [name = 'policy', bind = 'zlb']
  i = ilb;                                      // regime 2: the rate is stuck at the bound
  rn = rho_rn * rn(-1) + e_rn;                  // natural-rate (demand) shock
end;

occbin_constraints;
  name 'zlb'; bind inot <= ilb; relax inot > ilb;
end;

steady;
check;

shocks(surprise);
  var e_rn; periods 1; values -1.25;            // a fall in the natural rate of 5 percentage points (annualised)
end;

occbin_setup;
occbin_solver(simul_periods = 40);
