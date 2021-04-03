function roughness_c = part3_C3(OI,Scenedata, Refdata1)

OI_c = OI(:,1:350,1); 
OI_raw=Scenedata(:,1:350,1);

% % true dp replacement
% K=1000;
% mask = id_dp (Refdata1, col, K);
% L=medfilt2(OI_c,[3 7]);
% OI_c(mask)=L(mask);


imy_r = diff(OI_raw,1,1);
imx_r = diff(OI_raw,1,2);
imy_c = diff(OI_c,1,1);
imx_c = diff(OI_c,1,2);

roughness_c = (norm(imx_c,1) + norm(imy_c,1))/norm(OI_c,1);
end

