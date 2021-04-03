% Read an image volume.
% Make a depth-coded image.
% 2017-11 updated by Maria Magnusson
%------------------------------------

% load volume
%------------
load ctvol.mat;
ctvol = double(ctvol);

% check sizes, note that (row,col,z) corresponds to (y,x,z)
%----------------------------------------------------------
siz = size(ctvol);
sizex = siz(2);
sizey = siz(1);
sizez = siz(3);
zmax = sizez;

% open figure with gray colormap
%-------------------------------
figure;
colormap gray;

% Plot 3 slices from image volume
%--------------------------------
subplot(2,2,1), imshow(ctvol(:,:,5), [0 255]);
title('slice 5'); axis image; colorbar;
subplot(2,2,2), imshow(ctvol(:,:,15), [0 255]);
title('slice 15'); axis image; colorbar;
subplot(2,2,3), imshow(ctvol(:,:,32), [0 255]);
title('slice 32'); axis image; colorbar;




% Make a depth-image in the y-direction 
%--------------------------------------

tmp = zeros(3,3,3);
depth = zeros(sizez,sizex);

for z = 2 : zmax -1
  for x = 2 : sizex -1
    y=1;
    while (y<sizey) && (ctvol(y,x,z)<thresh)
      y=y+1;
    end
    if ctvol(y,x,z)>=thresh
      tmp(:,:,1) = ctvol(y-1:y+1,x-1:x+1,z-1);
      tmp(:,:,2) = ctvol(y-1:y+1,x-1:x+1,z);
      tmp(:,:,3) = ctvol(y-1:y+1,x-1:x+1,z+1);
      
      [grady, gradx, gradz] = gradxyz(tmp);
      grad = [grady, gradx, gradz];
      grad = grad / norm(grad);
      l_vec = [1,0,0];
      v_vec = l_vec;
      
      % find r
      % https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
      d = -l_vec;
      r_vec = d-2*(dot(d,grad)*grad); 
      
      scalarprod = dot(grad,l_vec);
      depth(z,x) = max(scalarprod,0)*kd*Md*Id + ks*Ms*Is*(dot(r_vec,v_vec))^n;
    end
  end
end

subplot(2,2,4);
imshow(flip(depth), [0 1]);
title('depthimage'); axis image; colorbar;
