function [P, Q, info] = solve_lre(A, B, C, D)
% SOLVE_LRE  Solve the linear rational-expectations model
%       A y(t-1) + B y(t) + C E_t y(t+1) + D e(t) = 0
%   for the stable solution y(t) = P y(t-1) + Q e(t) using the generalized Schur (QZ) decomposition.
%   info = 0: unique stable solution; 3: no stable solution; 4: indeterminacy.
n = size(A, 1);
E = [eye(n) zeros(n); zeros(n) C];
F = [zeros(n) eye(n); -A -B];
[AA, BB, Qz, Z] = qz(F, E);          % complex QZ (triangular)
lam = abs(diag(AA) ./ diag(BB));
stable = lam < 1 + 1e-6;
P = []; Q = []; info = 0;
if sum(stable) < n, info = 3; return; end
if sum(stable) > n, info = 4; return; end
[~, ~, ~, Z] = ordqz(AA, BB, Qz, Z, stable);
P = real(Z(n+1:end, 1:n) / Z(1:n, 1:n));
Q = -((B + C*P) \ D);
end
