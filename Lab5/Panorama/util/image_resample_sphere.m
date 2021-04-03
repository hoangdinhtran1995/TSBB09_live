%function img=image_resample_sphere(img0,K,R,range_h,range_v,astep)
%
% Resample image to spherical coordinates using bilinear interpolation
%
% IMG0     Input image (UINT8,MxN or MxNx3)
% K        Camera calibration matrix
% R        Rotation matrix from camera to wcs.
% RANGE_H  Range of horizontal angle in radians (default is [0 pi])
% RANGE_V  Range of vertical angle in radians (default is [-pi/4 pi/4])
% ASTEP    Step size in radians (default is 1e-3)
%
%Per-Erik Forssen, June 2008
function img=image_resample_sphere(img0,K,R,range_h,range_v,astep)

if nargin<6,astep=1e-3;end
if nargin<5,range_v=[-45 45]*pi/180;end
if nargin<4,range_h=[0 180]*pi/180;end

% Generate spherical coordinates
theta_c = range_h(1):astep:range_h(2);
phi_c   = range_v(1):astep:range_v(2);
[THETA,PHI]=meshgrid(theta_c,phi_c);

% Map to Cartesian 3D coordinates with
% singularities at north and south poles
u = cos(PHI).*sin(THETA);
v = sin(PHI);
d = cos(PHI).*cos(THETA);

% Rotate rays (we use R' here, as were going from wcs to ccs)
ur = R(1,1)*u+R(2,1)*v+R(3,1)*d;
vr = R(1,2)*u+R(2,2)*v+R(3,2)*d;
dr = R(1,3)*u+R(2,3)*v+R(3,3)*d;

% Project into image
h = K(3,3)*dr;
x = (K(1,1)*ur+K(1,2)*vr+K(1,3)*dr)./h;
y = (          K(2,2)*vr+K(2,3)*dr)./h;

% Mask for back side of sphere
mask = uint8(h>0);

% Sample from img0
ndim=size(img0,3);
img=zeros(length(phi_c),length(theta_c),ndim,'uint8');
for k=1:ndim,
   img(:,:,k)=uint8(linint(double(img0(:,:,k)),x,y));
end
%img=cvRemap_mex(img0,single(x-1),single(y-1),'CV_INTER_LINEAR',0);
for k=1:size(img,3)
    img(:,:,k)=img(:,:,k).*mask;
end
