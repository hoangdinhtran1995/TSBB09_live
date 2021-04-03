function [rgb] = raw2rgb(raw)
%raw in uint8 format with BGGR pattern.
im = im2double(raw);
 
[rows,cols] = size(im);

mask1=zeros(rows,cols);
mask1(2:2:end,2:2:end)=1;
mask2=zeros(rows,cols);
mask2(1:2:end,2:2:end)=1;
mask2(2:2:end,1:2:end)=1;
mask3=zeros(rows,cols);
mask3(1:2:end,1:2:end)=1;

f=[1 2 1]/4;

imgR=conv2(f,f,im.*mask1,'same')./conv2(f,f,mask1,'same');
imgG=conv2(f,f,im.*mask2,'same')./conv2(f,f,mask2,'same');
imgB=conv2(f,f,im.*mask3,'same')./conv2(f,f,mask3,'same');

rgb=reshape([imgR imgG imgB],[size(im) 3]);
end
