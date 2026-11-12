// Three-equation New Keynesian model. Inflation and interest rates are quarterly rates in percent;
// the reports below annualise them.
var x pi i rn u v;
varexo e_rn e_u e_v;
parameters beta sigma varphi theta kappa phi_pi phi_x rho_i rho_rn rho_u;

beta = 0.99;  sigma = 1;  varphi = 1;  theta = 0.75;
kappa = (1 - theta) * (1 - beta*theta) / theta * (sigma + varphi);
phi_pi = 1.5;  phi_x = 0.125;  rho_i = 0.8;
rho_rn = 0.8;  rho_u = 0.5;

model(linear);
  x  = x(+1) - (1/sigma) * (i - pi(+1) - rn);                       // dynamic IS curve
  pi = beta * pi(+1) + kappa * x + u;                                // Phillips curve
  i  = rho_i * i(-1) + (1 - rho_i) * (phi_pi*pi + phi_x*x) + v;      // Taylor rule
  rn = rho_rn * rn(-1) + e_rn;                                       // natural-rate (demand) shock
  u  = rho_u * u(-1) + e_u;                                          // cost-push shock
  v  = e_v;                                                          // monetary policy shock
end;

shocks;
  var e_rn; stderr 0.5;
  var e_u;  stderr 0.2;
  var e_v;  stderr 0.25;
end;

steady;
check;
stoch_simul(order=1, irf=20) x pi i;
