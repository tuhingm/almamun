function tr = hp_trend(y, lambda)
% HP_TREND  Hodrick-Prescott trend of a column vector (lambda = 1600 for quarterly data).
%   tr = hp_trend(y, lambda).  The cycle is y - tr.  No toolbox required.
y = y(:); T = numel(y);
D = spdiags(repmat([1 -2 1], T-2, 1), 0:2, T-2, T);   % second-difference operator
tr = (speye(T) + lambda * (D' * D)) \ y;
end
