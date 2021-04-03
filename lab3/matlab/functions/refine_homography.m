function H = refine_homography(H, model_points, image_points)
% Refine a homography with nonlinear optimization.
%
% H = REFINE_HOMOGRAPHY(H, model_points, image_points)
%
% param H:             Homography to optimize, np.array shape (3,3)
% param model_points:  Matrix of 3D model points, shape (3, N) (3 rows, N columns)
%                      In each column, the coordinates are ordered x,y,z
% param image_points:  Matrix of 2D image pixel coordinates, shape (2, N)
%                      in the same order as the model points.
% return: The refined homography

model_points = hom(model_points(1:2, :));  % As homogeneous 2D coordinates
fun = @(x)homography_error(x , image_points, model_points);
options = optimoptions(@lsqnonlin,'Algorithm', 'levenberg-marquardt', ...
    'OptimalityTolerance',1e-8, 'FiniteDifferenceType', 'central', 'Display', 'off');
x = lsqnonlin(fun, H(:), [], [], options);
H = reshape(x,[3, 3]);
H = H / H(3,3);

function residuals = homography_error(x, image_points, model_points)
H = reshape(x, [3,3]);
residuals = image_points - dhom(H * model_points);
residuals = residuals(:)';

