%function h=correspondences_plot(im1,im2,X1,X2,mflag)
%
%  Visualise correspondences
%
% IM1    Left/Top  image
% IM2    Right/Bottom image
% X1     Coordinate list for image 1 (2xN)
% X2     Coordinate list for image 2 (2xN)
% MFLAG  Mode: 1 for landscape (default), 0 for portrait mode 
%
%Per-Erik Forssen, June 2008
function h=correspondences_plot(im1,im2,X1,X2,mflag)

if nargin<5,mflag=1;end

clf;
if mflag,
    image([im1 im2]);
else
    image([im1;im2]);    
end
axis image
hold on;

[rows,cols,ndim]=size(im1);

Np=size(X1,2);
h=zeros(Np,1);
if mflag,
    for k=1:Np,
        h(k)=plot([X1(1,k) X2(1,k)+cols],[X1(2,k) X2(2,k)],'g');
    end
else
    for k=1:Np,
        h(k)=plot([X1(1,k) X2(1,k)],[X1(2,k) X2(2,k)+rows],'g');
    end
end