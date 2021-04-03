function rpes = reprojection_errors(A, d, Rs, ts, model_points, image_points)
% Compute reprojection errors of all points in the calibration.
%
% rpes = REPROJECTION_ERRORS(A, d, Rs, ts, model_points, image_points)
%
% param A:             Intrinsic camera matrix, shape (3, 3)
% param d:             Distortion coefficients
% param Rs:            Cell array of camera rotation matrices
% param ts:            Cell array of camera translation vectors, shape (3, 1)
% param model_points:  Cell array of matrices of 3D model points, each shaped (3, N)
%                      In each column, the coordinates are ordered x,y,z
% param image_points:  Cell array of matrices of 2D image pixel coordinates,
%                      each shaped (2, N), in the same order as the model points.
% returns:  All reprojection errors in a vector.
rpes = cell(1,length(Rs));
for i = 1:length(Rs)
    xy = image_points{i};
    uv =  project_points(A, d, Rs{i}, ts{i}, model_points{i}) - xy;
    rpes{i} = sqrt(sum(uv .* uv, 1));
end
rpes = cell2mat(rpes);