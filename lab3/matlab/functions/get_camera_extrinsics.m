function [Rs, ts] = get_camera_extrinsics(A, Hs)
% Decompose a set of homographies into camera rotations and translations.
%
% [Rs, ts] = GET_CAMERA_EXTRINSICS(A, Hs)
%
% param A:  The intrinsic matrix A,
% param Hs: Cell array of homographies H such that image_points = H * model_points
% return:   Rs, ts, where Rs is a cell array of rotation matrices, each of size (3,3)
%           and ts is a cell array of translation vectors, each of size (3,1)

n = length(Hs);
Rs = cell(n,1);
ts = cell(n,1);

for k = 1:n
    [h1, h2, h3] = column_split(Hs{k});
    r1 = inv(A) * h1;
    r2 = inv(A) * h2;
    L = 1 / norm(r1);
    r1 = L * r1;
    r2 = L * r2;
    r3 = cross(r1, r2);
    t = L * inv(A) * h3;
    R = [r1, r2, r3];

    Rs{k} = R;
    ts{k} = t;
end
