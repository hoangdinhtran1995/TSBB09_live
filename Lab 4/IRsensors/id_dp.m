function mask = id_dp (OI, col, K);

% creates a dead pixel mask using median filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% OI = one image of uncorrected image data
% 1..col = columns to be included in the calculation 
% diff_rawpix_filtpix = rawdata pixel - median filtered pixel

% OUTPUT:
% mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OI_timemean=mean(OI(:,1:col,:),3);
L = medfilt2(OI_timemean,[3 7]);
mask=abs(L-OI_timemean) > K;

