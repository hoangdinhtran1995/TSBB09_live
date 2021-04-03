function RMSE = part3_C1(OI,Scenedata, Refdata1)

OI_c = OI(:,1:350,1); 
OI_raw=Scenedata(:,1:350,1);
[row,col,frame]=size(OI_raw);

% % true dp replacement
% K=1000;
% mask = id_dp (Refdata1, col, K);
% L=medfilt2(OI_c,[3 7]);
% OI_c(mask)=L(mask);


RMSE=sqrt(mean2((OI_c-OI_raw).^2));
end

