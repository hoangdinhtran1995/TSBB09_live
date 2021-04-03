function A = get_camera_intrinsics(Hs)
% Estimate the camera intrinsic matrix A from a set of homographies.
%
% A = GET_CAMERA_INTRINSICS(Hs)
%
% param Hs: Cell array of homographies, each a (3,3) matrix
% return: The intrinsic matrix A, a (3,3) matrix
    
nhomographies = length(Hs);
V = zeros(2*nhomographies, 6);
for k = 1 : nhomographies
    H = Hs{k};
    V(2*k-1, :) = vectorize(H, 1, 2);
    V(2*k,:) = vectorize(H, 1, 1) - vectorize(H, 2, 2);
end

[~, ~, v] = svd(V);
b = v(:, end)';

% A1 = decompose_cholesky(b);
A2 = decompose_zhang(b);
A = A2;
end

function v = vectorize(H, i, j)

v = [H(1, i) * H(1, j), ...
     H(1, i) * H(2, j) + H(2, i) * H(1, j), ...
     H(2, i) * H(2, j), ...
     H(3, i) * H(1, j) + H(1, i) * H(3, j), ...
     H(3, i) * H(2, j) + H(2, i) * H(3, j), ...
     H(3, i) * H(3, j)];
end

function A = decompose_zhang(b)
[b11, b12, b22, b13, b23, b33] = column_split(b);

v = (b12*b13 - b11*b23) / (b11*b22 - b12*b12);
L = b33 - (b13*b13 + v*(b12*b13 - b11*b23)) / b11;
a = sqrt(L / b11);
b = sqrt(L * b11 / (b11*b22 - b12*b12));
g = -b12*a*a*b / L;
u = g*v/b - b13*a*a/L;

A = [a, g, u;
     0, b, v;
     0, 0, 1];
end

function A = decompose_cholesky(b)
[b11, b12, b22, b13, b23, b33] = column_split(b);

B = [b11, b12, b13;
     b12, b22, b23;
     b13, b23, b33];
 
if B(1,1) < 0 || B(2,2) < 0 || B(3,3) < 0
    B = -B;
end
L = chol(B);
A = inv(L) * L(3,3);
end
