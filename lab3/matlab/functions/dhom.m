function x = dhom(x)
% Normalize a matrix of homogeneous column vectors.
%
% x = DHOM(x)
den = repmat(x(end,:), [size(x,1)-1, 1]);
x = x(1:end-1,:) ./ den;