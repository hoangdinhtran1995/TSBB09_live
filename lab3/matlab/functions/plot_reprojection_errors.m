function plot_reprojection_errors(im_size, A, d, Rs, ts, model_points, image_points)
% Plot reprojection errors in the current figure.
%
% PLOT_REPROJECTION_ERRORS(im_size, A, d, Rs, ts, model_points, image_points)
%
% param im_size:       Image (height, width)
% param A:             Intrinsic camera matrix, shape (3, 3)
% param d:             Distortion coefficients
% param Rs:            Cell array of camera rotation matrices
% param ts:            Cell array of camera translation vectors, shape (3, 1)
% param model_points:  Cell array of matrices of 3D model points, each shaped (3, N)
%                      In each column, the coordinates are ordered x,y,z
% param image_points:  Cell array of matrices of 2D image pixel coordinates,
%                      each shaped (2, N), in the same order as the model points.

w = im_size(1);
h = im_size(2);
im = ones(h,w);
imshow(im); % Create a properly sized background image
hold on
r = [];
for i = 1:length(Rs)
    xy = image_points{i};
    uv =  project_points(A, d, Rs{i}, ts{i}, model_points{i}) - xy;
    % rpe = sqrt(sum(uv .* uv, 1))
    r = [r; xy(1,:)', xy(2,:)', uv(1,:)', uv(2,:)'];
end
quiver(r(:,1), r(:,2), r(:,3), r(:,4), 'linewidth', 2.0)
% quiver(r(:,1), r(:,2), r(:,3), r(:,4), 'autoscale', 'off', 'linewidth', 2.0)
hold off
