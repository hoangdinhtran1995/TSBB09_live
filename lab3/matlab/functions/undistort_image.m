function im = undistort_image(A, d, im, B)
%  Remove distortion from an image.
%
% im = UNDISTORT_IMAGE(A, d, im, B)
%
% param A:   Intrinsic matrix (3,3) of the input image
% param d:   Distortion, coefficients [k1, k2, p1, p2, k3]
%            p1, p2, k3 are optional, set to zero if not used
% param im:  Image to process, shape (height, width, 3), uint8, intensity range [0,255]
% param B:   Intrinsic matrix to apply to the output image.
%            B can be set equal to A, or some other matrix.
% return: Corrected image (converted to doubles, intensity range [0,1])

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

[h,w,~] = size(im);
[X,Y] = meshgrid(0:w-1, 0:h-1);
n = numel(X);
x = [reshape(X,1,n); reshape(Y,1,n); ones(1, numel(X))];
xn = dhom(B \ x);

r2 = sum(xn.^2, 1);
rd = 1.0 + k1 * r2 + k2 * r2.*r2 + k3 * r2.*r2.*r2;
td = [2*p1*xn(1,:).*xn(2,:) + p2*(r2 + 2*xn(1,:).*xn(1,:)); ...
      p1*(r2 + 2*xn(2,:).*xn(2,:)) + 2*p2*xn(1,:).*xn(2,:)];

x2 = A * hom(xn .* repmat(rd, [2,1]) + td);
Xq = reshape(x2(1,:)', h, w);
Yq = reshape(x2(2,:)', h, w);

im1 = im2double(im);
im2 = zeros(h,w,3);
im2(:,:,1) = interp2(X, Y, im1(:,:,1), Xq, Yq);
im2(:,:,2) = interp2(X, Y, im1(:,:,2), Xq, Yq);
im2(:,:,3) = interp2(X, Y, im1(:,:,3), Xq, Yq);

im = im2;
