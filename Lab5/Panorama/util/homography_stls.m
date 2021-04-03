%function H=homography_stls(Xt,Xi)
%
% Estimate homography that maps Xt to Xi using scaled TLS.
%
% XT   Target coordinate list (2xN) [x; y]
% XI   Image  coordinate list (2xN) [x; y]
%
%Per-Erik Forssen, March 2003

function H=homography_stls(Xt,Xi)
  
% Solve for coeffs in H where x_h=H*X

npts=size(Xt,2);  % Number of points

X=Xt(1,:)';
Y=Xt(2,:)';
Z=ones(npts,1);
x=Xi(1,:)';
y=Xi(2,:)';
%z=ones(npts,1);

% Generate scaling homography S
Xtm=mean(Xt,2);
l=sqrt(1/2/npts*sum(sum((Xt-Xtm*ones(1,npts)).^2)))+eps;
S=[1/l  0  -Xtm(1)/l;
   0   1/l -Xtm(2)/l;
   0    0         1];

% Map X,Y through homography S

Xh=X*S(1,1)+Y*S(1,2)+S(1,3);
Yh=X*S(2,1)+Y*S(2,2)+S(2,3);
h =X*S(3,1)+Y*S(3,2)+S(3,3);
X=Xh./h;
Y=Yh./h;

% Generate scaling homography T

Xim=mean(Xi,2);
l=sqrt(1/2/npts*sum(sum((Xi-Xim*ones(1,npts)).^2)))+eps;
T=[1/l  0  -Xim(1)/l;
   0   1/l -Xim(2)/l;
   0    0         1];

% Map x,y through homography T

xh=x*T(1,1)+y*T(1,2)+T(1,3);
yh=x*T(2,1)+y*T(2,2)+T(2,3);
h =x*T(3,1)+y*T(3,2)+T(3,3);
x=xh./h;
y=yh./h;

% Generate matrix A for which A*vec(H)=0

A=[zeros(npts,3) -X -Y -Z y.*X y.*Y y; ...
   X Y Z zeros(npts,3) -x.*X -x.*Y -x];
[U,D,V]=svd(A);

Hs=reshape(V(:,9),[3 3])';  % H is row-catenated
H=pinv(T)*Hs*S;
