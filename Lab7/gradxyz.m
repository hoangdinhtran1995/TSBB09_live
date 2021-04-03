function [grady, gradx, gradz] = gradxyz(in)

% a mirrored 3D-sobelx filter kernel
sobelx = zeros(3,3,3);
sobelx(:,:,1) = [-1 0 1; -2 0 2; -1 0 1];
sobelx(:,:,2) = [-2 0 2; -4 0 4; -2 0 2];
sobelx(:,:,3) = [-1 0 1; -2 0 2; -1 0 1];

gradx = sum(sum(sum(sobelx.*in)));


sobely = zeros(3,3,3);
sobely(:,:,1) = [-1 -2 -1; 0 0 0; 1 2 1];
sobely(:,:,2) = [-2 -4 -2; 0 0 0; 2 4 2];
sobely(:,:,3) = [-1 -2 -1; 0 0 0; 1 2 1];

grady = sum(sum(sum(sobely.*in)));

sobelz = zeros(3,3,3);
sobelz(:,:,1) = [-1 -2 -1; -2 -4 -2; -1 -2 -1];
sobelz(:,:,2) = [0 0 0; 0 0 0; 0 0 0];
sobelz(:,:,3) = [1 2 1; 2 4 2; 1 2 1];

gradz = sum(sum(sum(sobelz.*in)));