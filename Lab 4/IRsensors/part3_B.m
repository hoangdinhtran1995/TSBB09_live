% Load data: Refdata1, Refdata2, Refdata3, Scenedata

part = 'B_2';  % B_1, B_2 or B_3

%========= B_2 =================
% INPUT:
degree_pol=2;   % degree of polynomial = 0, 1 or 2
% OUTPUT: 
% OI_c and C
%======================================

switch part
    case {'B_3'}  
%======= B_3 ====================
% INPUT:
degree_pol=0;   % degree of polynomial = 0,1 or 2
OI_c=OI_c0;     % OI_cX = corrected scenedata using polynomial of degree X
K=1000;         % K = rawdata pixel-median filtered pixel
% OUTPUT:
% OI_c_dp
%==========================================================
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch part

    case 'B_1'       
        figure
        OI_raw=Scenedata(:,:,1);
        low_high=stretchlim(OI_raw/16383);
        imagesc(OI_raw,[low_high(1) low_high(2)]*16383)
        axis image; colormap gray; title('OI raw'); colorbar
        clear part low_high 
        
    case 'B_2'   
        % Corrects scenedata using C coefficients from polynomial of degree 1, 2 or 3 
        % Uses the functions "calc_C" and "NUC"    
        
        OI_raw=Scenedata(:,:,1);
        [row, col, frame]=size(OI_raw);
        C = calc_C(degree_pol,row,col,Refdata1,Refdata2,Refdata3);
        OI_c = NUC (OI_raw, C);   
        NaN_array=isnan(OI_c);OI_c(NaN_array)=0;
        figure
        low_high=stretchlim(OI_c/16383);
        imagesc(OI_c,[low_high(1) low_high(2)]*16383);
        title(['OI c' num2str(degree_pol)]); axis image; colormap gray; colorbar
        disp(['Corrected scene data using polynomial of degree ' num2str(degree_pol)]) 
        disp('Output: OI_c, C') 
        clear part low_high frame row col degree_pol
  
    case {'B_3'}       
        % Replaces the dead pixels in corrected image data  
        
        [row, col, frame]=size(OI_c);
        OI_c_dp=OI_c;
        mask = id_dp(Refdata1, col, K);
        L=medfilt2(OI_c,[3 7]);  
        OI_c_dp(mask)=L(mask);

        figure
        low_high=stretchlim(OI_c_dp/16383);
        imagesc(OI_c_dp,[low_high(1) low_high(2)]*16383)
        axis image; colormap gray; colorbar;
        title(['OI c' num2str(degree_pol) ' dp']) 

        disp(['Corrected scene data using polynomial of degree ' num2str(degree_pol)]) 
        disp(['Identified and replaced dead pixels using using a K value ' num2str(K)]) 
        disp('Output: OI_c_dp') 
       
        clear part low_high frame row col degree_pol mask K L
        
end






