function ll = kalman_loglik(Y, P, Q, Sigma_e, H, R)
% KALMAN_LOGLIK  Gaussian log-likelihood of observations Y (T x k) for the state-space model
%       s(t) = P s(t-1) + Q e(t),  e ~ N(0, Sigma_e)      (state)
%       Y(t) = H s(t) + u(t),      u ~ N(0, R)            (observation)
%   The filter starts from the unconditional distribution of the state.
n = size(P, 1); [T, k] = size(Y);
Om = Q * Sigma_e * Q';
V = reshape((eye(n^2) - kron(P, P)) \ Om(:), n, n);   % unconditional variance
s = zeros(n, 1); ll = 0;
for t = 1:T
  s = P * s; V = P * V * P' + Om;                       % predict
  v = Y(t, :)' - H * s; F = H * V * H' + R; F = (F + F')/2;
  [cF, p] = chol(F); if p, ll = -Inf; return; end
  Fi_v = cF \ (cF' \ v);
  ll = ll - 0.5 * (k*log(2*pi) + 2*sum(log(diag(cF))) + v' * Fi_v);
  K = V * H' / F; s = s + K * v; V = V - K * H * V;    % update
end
end
