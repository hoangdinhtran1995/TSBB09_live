function H = find_homography(model_points, image_points)
% Estimate a homography H such that image_points = H @ model_points.
%
% H = FIND_HOMOGRAPHY(model_points, image_points)
%
% param model_points: array of 3D model points, shape (3, N) (3 rows,
%                     N columns) In each column, the coordinates are
%                     ordered x,y,z
% param image_points: array of 2D image pixel coordinates, shape (2, N)
%                      in the same order as the model points.
% return: The estimated homography, array shape (3,3)

A = model_points(1:2,:);
B = image_points;

Na = normalization_matrix(A);
Nb = normalization_matrix(B);

A = dhom(Na * hom(A));
B = dhom(Nb * hom(B));

npoints = size(A,2);
M = zeros(2*npoints, 9);

for k = 1:npoints
    x = A(1, k);
    y = A(2, k);
    u = B(1, k);
    v = B(2, k);
    M(2*k-1,:) = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u];
    M(2*k,:) = [0, 0, 0, x, y, 1, -v*x, -v*y, -v];
end

[~, ~, v] = svd(M);
h = v(:, end);
H = reshape(h, [3,3])';

H = Nb \ H * Na;
H = H / H(3, 3);

