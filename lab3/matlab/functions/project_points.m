function x = project_points(A, d, R, t, points)
%  Project a set of 3D points into a camera.
%
% x = PROJECT_POINTS(A, d, R, t, points)
%
% param A:       Intrinsic matrix (3,3)
% param d:       Distortion, coefficients
% param R:       World-to-camera rotation matrix (3,3)
% param t:       World-to-camera translation vector (3,1).
% param points:  N 3D points to project. Numpy array, shape (3,N)
% return: Reprojected (2,N) image 2D points

k1 = d(1);
k2 = d(2);
p1 = 0;
p2 = 0;
k3 = 0;

if length(d) >= 4
    p1 = d(3);
    p2 = d(4);
end

if length(d) >= 5
    k3 = d(5);
end

t = repmat(t(:),[1,size(points,2)]);
xn = dhom(R * points + t);
r2 = sum(xn.^2, 1);
rd = 1.0 + k1 * r2 + k2 * r2.*r2 + k3 * r2.*r2.*r2;
td = [2*p1*xn(1,:).*xn(2,:) + p2*(r2 + 2*xn(1,:).*xn(1,:)); ...
      p1*(r2 + 2*xn(2,:).*xn(2,:)) + 2*p2*xn(1,:).*xn(2,:)];

x = A * hom(xn .* repmat(rd, [2,1]) + td);
x = dhom(x);
