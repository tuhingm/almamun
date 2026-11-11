function [b, se, res] = ols_nw(y, X, L)
% OLS_NW  OLS with Newey-West (Bartlett kernel, L lags) standard errors.
%   [b, se, res] = ols_nw(y, X, L).  Include a column of ones in X for a constant.
b = X \ y; res = y - X*b; T = size(X, 1);
g = X .* res; S = g' * g;
for l = 1:L
  w = 1 - l/(L+1); G = g(l+1:end, :)' * g(1:end-l, :); S = S + w * (G + G');
end
XXi = inv(X' * X); V = XXi * S * XXi; se = sqrt(diag(V));
end
