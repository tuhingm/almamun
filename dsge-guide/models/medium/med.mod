// Medium-scale New Keynesian model (Smets-Wouters type, flexible wages), estimated on demeaned U.S. data.
var y c i k q rk l w mc pi R a b ei g ep YGR CGR IGR INFL INT;
varexo e_a e_b e_i e_g e_p e_R;
parameters beta delta alpha cy iy gy h phi_i sigma_l iota_p xi_p psi_pi psi_y rho_R rho_a rho_b rho_i rho_g rho_p;

beta = 0.995;  delta = 0.025;  alpha = 0.3;  gy = 0.18;
iy = delta * alpha / (1/beta - 1 + delta);  cy = 1 - iy - gy;      // steady-state expenditure shares
h = 0.7;  phi_i = 5;  sigma_l = 2;  iota_p = 0.3;  xi_p = 0.75;
psi_pi = 1.5;  psi_y = 0.1;  rho_R = 0.8;
rho_a = 0.95;  rho_b = 0.7;  rho_i = 0.7;  rho_g = 0.9;  rho_p = 0.5;

model(linear);
  # kappa = (1 - xi_p) * (1 - beta*xi_p) / xi_p / (1 + beta*iota_p);
  c  = h/(1+h)*c(-1) + 1/(1+h)*c(+1) - (1-h)/(1+h)*(R - pi(+1)) + b;      // consumption Euler with habits
  i  = 1/(1+beta)*i(-1) + beta/(1+beta)*i(+1) + 1/((1+beta)*phi_i)*q + ei; // investment
  q  = beta*(1-delta)*q(+1) + (1 - beta*(1-delta))*rk(+1) - (R - pi(+1)); // Tobin's q
  k  = (1-delta)*k(-1) + delta*i;                                        // capital
  y  = a + alpha*k(-1) + (1-alpha)*l;                                    // production
  rk = w + l - k(-1);                                                    // cost minimisation
  mc = alpha*rk + (1-alpha)*w - a;                                       // real marginal cost
  w  = sigma_l*l + (c - h*c(-1))/(1-h);                                  // labour supply
  pi = iota_p/(1+beta*iota_p)*pi(-1) + beta/(1+beta*iota_p)*pi(+1) + kappa*mc + ep;  // Phillips curve
  y  = cy*c + iy*i + gy*g;                                               // resource constraint
  R  = rho_R*R(-1) + (1-rho_R)*(psi_pi*pi + psi_y*y) + e_R;               // Taylor rule
  a  = rho_a*a(-1) + e_a;   b = rho_b*b(-1) + e_b;   ei = rho_i*ei(-1) + e_i;
  g  = rho_g*g(-1) + e_g;   ep = rho_p*ep(-1) + e_p;
  YGR = y - y(-1);  CGR = c - c(-1);  IGR = i - i(-1);  INFL = 4*pi;  INT = 4*R;   // measurement
end;

shocks;
  var e_a; stderr 0.5;  var e_b; stderr 0.3;  var e_i; stderr 0.5;
  var e_g; stderr 1;    var e_p; stderr 0.2;  var e_R; stderr 0.2;
end;

steady;
check;

estimated_params;
  h,      beta_pdf,  0.70, 0.10;   phi_i,  gamma_pdf, 4.00, 1.50;   sigma_l, gamma_pdf, 2.00, 0.75;
  iota_p, beta_pdf,  0.50, 0.15;   xi_p,   beta_pdf,  0.66, 0.10;
  psi_pi, gamma_pdf, 1.50, 0.25;   psi_y,  gamma_pdf, 0.125, 0.05;  rho_R,  beta_pdf,  0.75, 0.10;
  rho_a,  beta_pdf,  0.50, 0.20;   rho_b,  beta_pdf,  0.50, 0.20;   rho_i,  beta_pdf,  0.50, 0.20;
  rho_g,  beta_pdf,  0.50, 0.20;   rho_p,  beta_pdf,  0.50, 0.20;
  stderr e_a, inv_gamma_pdf, 0.5, 2;  stderr e_b, inv_gamma_pdf, 0.3, 2;  stderr e_i, inv_gamma_pdf, 0.5, 2;
  stderr e_g, inv_gamma_pdf, 0.5, 2;  stderr e_p, inv_gamma_pdf, 0.2, 2;  stderr e_R, inv_gamma_pdf, 0.2, 2;
end;

varobs YGR CGR IGR INFL INT;

estimation(datafile = 'us_medium_demeaned.csv', mode_compute = 5, mh_replic = 4000, mh_nblocks = 2,
           mh_jscale = 0.25, mh_drop = 0.5, moments_varendo, nograph) y c i pi R;
