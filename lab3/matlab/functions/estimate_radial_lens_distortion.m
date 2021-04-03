function [k1, k2] = estimate_radial_lens_distortion(A, Rs, ts, model_points, image_points)
% Estimate radial distortion parameters k1 and k2.
%
% [k1, k2] = ESTIMATE_RADIAL_LENS_DISTORTION(A, Rs, ts, model_points, image_points)
%
% param A:  Camera intrinsic matrix
% param Rs: Cell array of camera rotation matrices (one cell per camera)
% param ts: Cell array of camera translation vectors, each shaped (3,1)
% param model_points: Cell array of model points matrices
% param image_points: Cell array of image points matrices
% return: The radial distortion coefficients k1 and k2

nimages = size(image_points, 1);
npoints = size(image_points{1}, 2);

D = [];
d = [];
u0 = A(1,3);
v0 = A(2,3);

for i = 1:nimages
    
    R = Rs{i};
    t = ts{i};
    X = model_points{i};
    y = image_points{i};

    xn = R * X + repmat(t, [1, size(X,2)]);
    x = dhom(A * xn);
    xn = dhom(xn);

    for j = 1:npoints

        u = x(1,j);
        v = x(2,j);

        r2 = sum(xn(:,j) .^ 2);
        r4 = r2*r2;

        D = [D; (u - u0) * r2, (u - u0) * r4];
        D = [D; (v - v0) * r2, (v - v0) * r4];

        d = [d; y(1, j) - u];
        d = [d; y(2, j) - v];
    end
    
end

x = mldivide(D, d);
k1 = x(1);
k2 = x(2);
disp(k1)
disp(k2)

