%function Im=image_lensdist_inv(dtype,Im0,lgamma,wc)
%
% Undo radial lens-distortion of an image
% by resampling to a new grid.
%
% Distortion models:
%      polynomial:  rn = lgamma*rn^3
%      arctangent:  rn=atan(r*lgamma)/lgamma
%
% DTYPE  Distortion type ('pol' or 'atan')
% IM0    Image to resample (UINT8,MxNx3)
% LGAMMA Lens distortion parameter
% WC     Distortion centre [xc yc] (default [cols rows]/2)
%
% See also LENSDIST, LENSDIST_INV, and LININT
%
%Per-Erik Forssen Aug 2004

function Im=image_lensdist_inv(dtype,Im0,lgamma,wc)

[rows,cols,ndim]=size(Im0);
if nargin<4,wc=[cols rows]/2;end

% Find out coordinate range of new image
xl=[1 cols/2 cols cols/2;rows/2 1 rows/2 rows];
Xx=lensdist_inv(dtype,xl,wc,lgamma);

%Generate grid
xl=[1:cols]/cols*(Xx(1,3)-Xx(1,1))+Xx(1,1);
yl=[1:rows]/rows*(Xx(2,4)-Xx(2,2))+Xx(2,2);
[X,Y]=meshgrid(xl,yl);
%Distort grid
Xc=lensdist(dtype,[X(:) Y(:)]',wc,lgamma);
X=reshape(Xc(1,:),[rows cols]);
Y=reshape(Xc(2,:),[rows cols]);
Im=zeros(rows,cols,ndim,'uint8');
for k=1:ndim,
  Im(:,:,k)=uint8(linint(double(Im0(:,:,k)),X,Y));
end
%Im=cvRemap_mex(Im0,single(X-1),single(Y-1),'CV_INTER_LINEAR',0);
