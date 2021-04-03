function [A, k1, k2, Rs, ts] = refine_calibration(A, k1 ,k2, Rs, ts, model_points, image_points)
% Refine a calibration with nonlinear optimization.
%
% [A, k1, k2, Rs, ts] = REFINE_CALIBRATION(A, k1 ,k2, Rs, ts, model_points, image_points)
%
% param A:             Intrinsic camera matrix, shape (3, 3)
% param k1:            Second order radial distortion coefficients
% param k2:            Fourth order radial distortion coefficients
% param Rs:            Cell array of camera rotation matrices
% param ts:            Cell array of camera translation vectors, each shaped (3, 1)
% param model_points:  Cell array of matrices of 3D model points, each shaped (3, N)
%                      In each column, the coordinates are ordered x,y,z
% param image_points:  (n,1) cell array of np.array objects of 2D image pixel coordinates,
%                      shape (2, N) in the same order as the model points.
% return:  Refined A, k1, k2, Rs, ts

% Flatten the parameters for the optimizer

fx = A(1,1);
ga = A(1,2);
u0 = A(1,3);
fy = A(2,2);
v0 = A(2,3);

x = [fx, ga, u0, fy, v0, k1, k2];
ncameras = length(Rs);
for k = 1:ncameras
    r = rodrigues(Rs{k});
    t = ts{k}';
    x = [x, r, t];
end

% Optimize

fun = @(x)reprojection_error(x , model_points, image_points);
options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt', 'FunctionTolerance', 1e-8, 'Display', 'off');
x = lsqnonlin(fun, x, [], [], options);

% Recover the parameters

[fx, ga, u0, fy, v0, k1, k2] = column_split(x(1:7));
A = [fx, ga, u0; 0, fy, v0; 0, 0, 1];

Rs = cell(ncameras,1);
ts = cell(ncameras,1);

cameras = reshape(x(8:end), 6, []);  % shape = (6, ncameras)
for k = 1:ncameras
    R = rodrigues(cameras(1:3,k));
    t = cameras(4:end,k)';
    Rs{k} = R;
    ts{k} = t;
end

function residuals = reprojection_error(x, model_points, image_points)

residuals = [];
[fx, ga, u0, fy, v0, k1, k2] = column_split(x(1:7));

cameras = reshape(x(8:end), 6, []);
ncameras = size(cameras,2);
for k = 1:ncameras
    R = rodrigues(cameras(1:3,k));
    t = cameras(4:end,k);

    x = R * model_points{k} + repmat(t, [1, size(model_points{k}, 2)]); % Transform world to camera coordinates
    xn = dhom(x);                %
    r2 = repmat(sum(xn.^2, 1), [2, 1]);                      % Add distortion
    xn = xn .* (1.0 + k1*r2 + k2 * r2.*r2);  %
    u = fx * xn(1,:) + ga * xn(2,:) + u0;  % Transform to pixel coordinates
    v = fy * xn(2,:) + v0;                %
    
    im_pts = image_points{k};
    ru = u - im_pts(1,:);  % Compute residuals
    rv = v - im_pts(2,:);  %
    
    residuals = [residuals, ru, rv];
end
