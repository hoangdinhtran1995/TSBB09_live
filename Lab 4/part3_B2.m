function OI_c = part3_B2(degree_pol,Scenedata,Refdata1,Refdata2,Refdata3)
% Corrects scenedata using C coefficients from polynomial of degree 1, 2 or 3 
% Uses the functions "calc_C" and "NUC"    
OI_raw=Scenedata(:,:,1);
[row, col, ~]=size(OI_raw);
C = calc_C(degree_pol,row,col,Refdata1,Refdata2,Refdata3);
OI_c = NUC (OI_raw, C);   
NaN_array=isnan(OI_c);OI_c(NaN_array)=0;
%figure
low_high=stretchlim(OI_c/16383);
imagesc(OI_c,[low_high(1) low_high(2)]*16383);
title(['OI c' num2str(degree_pol)]); axis image; colormap gray; colorbar
clear part low_high frame row col degree_pol
end
