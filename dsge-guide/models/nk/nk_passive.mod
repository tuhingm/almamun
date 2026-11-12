var x pi i rn u v;
varexo e_rn e_u e_v;
parameters beta sigma varphi theta kappa phi_pi phi_x rho_i rho_rn rho_u;
beta = 0.99;  sigma = 1;  varphi = 1;  theta = 0.75;
kappa = (1 - theta) * (1 - beta*theta) / theta * (sigma + varphi);
phi_pi = 0.9;  phi_x = 0.125;  rho_i = 0.8;  rho_rn = 0.8;  rho_u = 0.5;     // passive policy: phi_pi < 1
model(linear);
  x  = x(+1) - (1/sigma) * (i - pi(+1) - rn);
  pi = beta * pi(+1) + kappa * x + u;
  i  = rho_i * i(-1) + (1 - rho_i) * (phi_pi*pi + phi_x*x) + v;
  rn = rho_rn * rn(-1) + e_rn;  u = rho_u * u(-1) + e_u;  v = e_v;
end;
steady;
check;
