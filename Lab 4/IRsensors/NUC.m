function OI_c = NUC (OI, C)

% non-uniformity correction of scene data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% OI = one image of uncorrected image data
% C = matrix x*y*n of correction coefficients

% OUTPUT:
% OI_c = corrected image data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OI=OI(:,:,1);  % if nr of frames > 1
[row,col,degree_pol]=size(C);
OI_c=zeros(row,col);
degree_pol=degree_pol-1;

if degree_pol == 0
    OI_c=OI-(C(:,:,1));
    OI_c=abs(OI_c);
    maxvalue=OI_c > 16383;
    OI_c(maxvalue)=16383;
elseif degree_pol == 1
    OI_c=(OI-C(:,:,1))./(1+C(:,:,2));
    OI_c=abs(OI_c);
    maxvalue=OI_c > 16383;
    OI_c(maxvalue)=16383;
else
    for k=1:row
        for l=1:col
            OI_c(k,l)= - (1+C(k,l,2))/(2*C(k,l,3)) - sqrt( (1+C(k,l,2))^2 /(4* (C(k,l,3)) ^2) + (OI(k,l)-C(k,l,1))/C(k,l,3));
            if sign(OI_c(k,l))==-1
                OI_c(k,l)=-(1+C(k,l,2))/(2*C(k,l,3))+ sqrt( (1+C(k,l,2))^2 /(4* (C(k,l,3))^2) + (OI(k,l)-C(k,l,1))/C(k,l,3));
            end       
        end
    end
    OI_c=abs(OI_c);
    maxvalue=OI_c > 16383;
    OI_c(maxvalue)=16383;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
