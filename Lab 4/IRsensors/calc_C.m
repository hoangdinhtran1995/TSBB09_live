function C = calc_C(degree_pol,row,col,Refdata1,Refdata2,Refdata3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates C coefficients from a 0th, 1st or 2nd degree polynomial

% Input data:
% degree_pol = 0, 1 or 2 = degree of the polynom
% 1..row, 1..col = rows and columns to be included in the calculation 
% Refdata1 ... = Reference data, "1" is at the lowest radiance level

% Output data:
% C(:,:,1)=C0 ... = corrections coefficients

if degree_pol==0    
    
	Refdata1_timemean=mean(Refdata1(1:row,1:col,:),3);   
    Refdata1_timemean_allpix=mean2(Refdata1_timemean); 
    delta_Y1=Refdata1_timemean-Refdata1_timemean_allpix;      
    C(:,:,1)=delta_Y1;

elseif degree_pol==1
    
	Refdata1_timemean=mean(Refdata1(:,1:col,:),3);
    Refdata2_timemean=mean(Refdata2(:,1:col,:),3);
    
    Refdata1_timemean_allpix=mean2(Refdata1_timemean); 
    Refdata2_timemean_allpix=mean2(Refdata2_timemean); 
    delta_Y1=Refdata1_timemean-Refdata1_timemean_allpix;
    delta_Y2=Refdata2_timemean-Refdata2_timemean_allpix;
    X=([[Refdata1_timemean_allpix Refdata2_timemean_allpix]' [1 1]']);

    for m=1:row
        for n=1:col  
        p=X\[delta_Y1(m,n) delta_Y2(m,n)]';
        C(m,n,1)=p(2);
        C(m,n,2)=p(1);
        end
    end
      
elseif degree_pol==2
    
    Refdata1_timemean=mean(Refdata1(:,1:col,:),3);
    Refdata2_timemean=mean(Refdata2(:,1:col,:),3);
    Refdata3_timemean=mean(Refdata3(:,1:col,:),3);

    Refdata1_timemean_allpix=mean2(Refdata1_timemean); 
    Refdata2_timemean_allpix=mean2(Refdata2_timemean); 
    Refdata3_timemean_allpix=mean2(Refdata3_timemean);
    
    delta_Y1=Refdata1_timemean-Refdata1_timemean_allpix;
    delta_Y2=Refdata2_timemean-Refdata2_timemean_allpix;
    delta_Y3=Refdata3_timemean-Refdata3_timemean_allpix;
    
    X=([[Refdata1_timemean_allpix Refdata2_timemean_allpix Refdata3_timemean_allpix]'.^2 [Refdata1_timemean_allpix Refdata2_timemean_allpix Refdata3_timemean_allpix]' [1 1 1]']);

    for m=1:row
        for n=1:col  
        p=X\[delta_Y1(m,n) delta_Y2(m,n) delta_Y3(m,n)]';
        C(m,n,1)=p(3);
        C(m,n,2)=p(2);
        C(m,n,3)=p(1);
        end
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%