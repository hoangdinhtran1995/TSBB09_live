%function Y=lensdist(dtype,X,wc,lgamma)
%
% Apply radial lens distortion to a list of image coordinates.
%
% Distortion models:
%      polynomial:  rn = lgamma*rn^3
%      arctangent:  rn=atan(r*lgamma)/lgamma
%
% DTYPE  Distortion type ('pol' or 'atan')
% X      Image coordinate list (2xN)
% WC     Distortion centre (2x1)
% LGAMMA Lens distortion parameter
%
% See also LENSDIST_INV and IMAGE_LENSDIST_INV
%
%Per-Erik Forssen, July 2008

function Y=lensdist(dtype,X,wc,lgamma)

% Switch to polar coordinates
rn=sqrt((X(1,:)-wc(1)).^2+(X(2,:)-wc(2)).^2);
phi=atan2(X(2,:)-wc(2),X(1,:)-wc(1));

% Map radius
switch lower(dtype)
    case 'pol',
        r=rn-lgamma*rn.^3;
    case 'atan',
        r=atan(rn*lgamma)/lgamma;
end

% Switch back to rectangular coordinates
Y=zeros(size(X));
Y(1,:)=wc(1)+r.*cos(phi);
Y(2,:)=wc(2)+r.*sin(phi);