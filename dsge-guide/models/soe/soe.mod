// Galí-Monacelli (2005) small open economy, Cole-Obstfeld case. Quarterly rates in percent.
@#ifndef RULE
  @#define RULE = 1
@#endif
var x y yn pih pi s e i a rrn;
varexo eps_a;
parameters beta alpha theta varphi phi_pi rho_a sigma_a lambda kappa_a Gamma_a;

beta = 0.99;  alpha = 0.4;  theta = 0.75;  varphi = 3;  phi_pi = 1.5;  rho_a = 0.66;
sigma_a = 1;                                            // sigma/((1-alpha)+alpha*omega) = 1 in the Cole-Obstfeld case
lambda  = (1 - theta) * (1 - beta*theta) / theta;
kappa_a = lambda * (sigma_a + varphi);
Gamma_a = (1 + varphi) / (sigma_a + varphi);

model(linear);
  x   = x(+1) - (1/sigma_a) * (i - pih(+1) - rrn);       // IS curve
  pih = beta * pih(+1) + kappa_a * x;                     // domestic Phillips curve
  x   = y - yn;                                           // output gap
  yn  = Gamma_a * a;                                      // natural output
  rrn = -sigma_a * Gamma_a * (1 - rho_a) * a;             // natural real rate
  s   = sigma_a * y;                                      // terms of trade
  pi  = pih + alpha * (s - s(-1));                        // CPI inflation
  e - e(-1) = s - s(-1) + pih;                            // nominal exchange rate
  a   = rho_a * a(-1) + eps_a;                            // domestic productivity
@#if RULE == 0
  pih = 0;                                                // strict domestic-inflation targeting (the optimum)
@#elseif RULE == 1
  i = phi_pi * pih;                                       // DITR
@#elseif RULE == 2
  i = phi_pi * pi;                                        // CITR
@#else
  e = 0;                                                  // PEG
@#endif
end;

shocks;
  var eps_a; stderr 0.712;                                // percent, as in Galí-Monacelli
end;

steady;
check;
stoch_simul(order=1, irf=16, nograph) y pih pi s e i x;
