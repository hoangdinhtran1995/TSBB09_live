
%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%
qualityMetrics = 'uiqi';    % RMSE, UIQI or Roughness
col=350;                    % the  are kept 
OI_c=OI_c2_dp(:,1:col);        % OI_cX = corrected scenedata using polynomial of degree X
dp_replacement='true';     % 'true' performs a dead pixel replacement 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OI_raw=Scenedata(:,1:col,1);
[row,col,frame]=size(OI_raw);

switch lower(dp_replacement)
    
case 'true'
    % replacement of dead pixels
    K=1000;
    mask = id_dp (Refdata1, col, K);
    L=medfilt2(OI_c,[3 7]);
    OI_c(mask)=L(mask);
end

switch lower(qualityMetrics)
    
    case 'rmse'
    %%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RMSE=sqrt(mean2((OI_c-OI_raw).^2))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    case 'uiqi'
    corrcoefxy=corrcoef(reshape(OI_raw,1,row*col),reshape(OI_c,1,row*col));
    corrcoefxy=1/(row*col-1)*sum ( (reshape(OI_raw,row*col,1)-mean2(OI_raw)).*(reshape(OI_c,row*col,1)-mean2(OI_c)) );
    Tmp1 = 4*mean2(OI_c)*mean2(OI_raw)*corrcoefxy;
    Tmp2 = (((std2(OI_c)^2)+(std2(OI_raw)^2)) * ((mean2(OI_raw))^2 + (mean2(OI_c))^2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%
    UIQI= Tmp1/Tmp2  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    case 'roughness' 
    imy_r = diff(OI_raw,1,1);
    imx_r = diff(OI_raw,1,2);
    imy_c = diff(OI_c,1,1);
    imx_c = diff(OI_c,1,2);
    %%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    roughness_raw = (norm(imx_r,1) + norm(imy_r,1))/norm(OI_raw,1)
    roughness_c = (norm(imx_c,1) + norm(imy_c,1))/norm(OI_c,1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end
    