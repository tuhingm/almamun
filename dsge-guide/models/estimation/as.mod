// An-Schorfheide (2007) New Keynesian model, estimated on demeaned U.S. data.
var y pi R g z YGR INFL INT;
varexo e_R e_g e_z;
parameters tau kappa psi1 psi2 rho_R rho_g rho_z r_A;

tau = 2;  kappa = 0.2;  psi1 = 1.5;  psi2 = 0.5;  rho_R = 0.5;  rho_g = 0.8;  rho_z = 0.66;
r_A = 2;                                                   // annual real rate, fixes beta (not estimated)

model(linear);
  # beta = 1 / (1 + r_A/400);
  y  = y(+1) - (1/tau) * (R - pi(+1) - z(+1)) + g - g(+1);
  pi = beta * pi(+1) + kappa * (y - g);
  R  = rho_R * R(-1) + (1 - rho_R) * (psi1*pi + psi2*(y - g)) + e_R;
  g  = rho_g * g(-1) + e_g;
  z  = rho_z * z(-1) + e_z;
  YGR  = y - y(-1) + z;                                    // measurement equations
  INFL = 4 * pi;
  INT  = 4 * R;
end;

shocks;
  var e_R; stderr 0.4;  var e_g; stderr 1;  var e_z; stderr 0.5;
end;

steady;
check;

estimated_params;
  tau,   gamma_pdf, 2.00, 0.50;
  kappa, gamma_pdf, 0.20, 0.10;
  psi1,  gamma_pdf, 1.50, 0.25;
  psi2,  gamma_pdf, 0.50, 0.25;
  rho_R, beta_pdf,  0.50, 0.20;
  rho_g, beta_pdf,  0.80, 0.10;
  rho_z, beta_pdf,  0.66, 0.15;
  stderr e_R, inv_gamma_pdf, 0.4, 4;
  stderr e_g, inv_gamma_pdf, 1.0, 4;
  stderr e_z, inv_gamma_pdf, 0.5, 4;
end;

varobs YGR INFL INT;

estimation(datafile = 'us_small_demeaned.csv', mode_compute = 5, mh_replic = 10000, mh_nblocks = 2,
           mh_jscale = 0.6, mh_drop = 0.5, bayesian_irf, irf = 16, nograph) y pi R;
shock_decomposition(nograph) INFL;
