function [A, d, Rs, ts, rpe] = calibrate_zhang(model_points, image_points)

% Calibrate with Zhang's method
% param model_points:   Matrix of 3D model points, shape (3, N) (3 rows, N columns)
%                       In each column, the coordinates are ordered x,y,z
% param image_points:   Matrix of 2D image pixel coordinates, shape (2, N)
%                       in the same order as the model points.
% param im_size:        tuple (width, height) in pixels
% return: A, d, Rs, ts, rpe   camera intrinsics, distortion coefficients,
% camera rotation matrices, camera translation vectors and the average
% reprojection error [pixels]

% Get homography cell array
Hs = cell(size(image_points));
for i = 1:length(image_points)
    H = find_homography(model_points{i}, image_points{i});
    Hs{i} = H;
end

%get cam matrix and distortion coeff
A = get_camera_intrinsics(Hs); 
[Rs, ts] = get_camera_extrinsics(A, Hs); 
[k1, k2] = estimate_radial_lens_distortion(A, Rs, ts, model_points, image_points);

%refine
[A, k1, k2, Rs, ts] = refine_calibration(A, k1 ,k2, Rs, ts, model_points, image_points);

%dist coeff
d = [k1, k2];

%reproj error
rpes = reprojection_errors(A, d, Rs, ts, model_points, image_points);
rpe = sqrt(mean(rpes.^2));