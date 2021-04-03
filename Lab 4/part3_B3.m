function OI_c_dp = part3_B2(degree_pol,Scenedata,Refdata1,Refdata2,Refdata3, OI_c)

K=1000;         % K = rawdata pixel-median filtered pixel

% Replaces the dead pixels in corrected image data  
        
[~, col, ~]=size(OI_c);
OI_c_dp=OI_c;
mask = id_dp(Refdata1, col, K);
L=medfilt2(OI_c,[3 7]);  
OI_c_dp(mask)=L(mask);

low_high=stretchlim(OI_c_dp/16383);
imagesc(OI_c_dp,[low_high(1) low_high(2)]*16383)
axis image; colormap gray; colorbar;
title(['OI c' num2str(degree_pol) ' dp']) 
end
