function UIQI = part3_C2(OI,Scenedata, Refdata1)

OI_c = OI(:,1:350,1); 
OI_raw=Scenedata(:,1:350,1);
[row,col,frame]=size(OI_raw);

% % true dp replacement
% K=1000;
% mask = id_dp (Refdata1, col, K);
% L=medfilt2(OI_c,[3 7]);
% OI_c(mask)=L(mask);


corrcoefxy=corrcoef(reshape(OI_raw,1,row*col),reshape(OI_c,1,row*col));
corrcoefxy=1/(row*col-1)*sum ( (reshape(OI_raw,row*col,1)-mean2(OI_raw)).*(reshape(OI_c,row*col,1)-mean2(OI_c)) );
Tmp1 = 4*mean2(OI_c)*mean2(OI_raw)*corrcoefxy;
Tmp2 = (((std2(OI_c)^2)+(std2(OI_raw)^2)) * ((mean2(OI_raw))^2 + (mean2(OI_c))^2));

UIQI= Tmp1/Tmp2;
end

