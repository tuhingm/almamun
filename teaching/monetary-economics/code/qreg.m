function b = qreg(y, X, tau)
% QREG  Linear quantile regression of y on X at quantile tau (0 < tau < 1).
%   Iteratively reweighted least squares on the check function; no toolbox required.
b = X \ y;
for it = 1:500
  r = y - X*b;
  w = (tau * (r >= 0) + (1 - tau) * (r < 0)) ./ max(abs(r), 1e-6);
  Xw = X .* w;
  bn = (X' * Xw) \ (Xw' * y);
  if max(abs(bn - b)) < 1e-9, b = bn; break; end
  b = bn;
end
end
