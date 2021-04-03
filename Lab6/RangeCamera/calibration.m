% Calibration points in the laser plane
%--------------------------------------
Y1 =   0;   R1 =  17;   
Y2 =  36.5; R2 =  36.5; 
Y3 =  53.5; R3 =  73;   
Y4 =  70.5; R4 = 109.5; 
Y5 =  87.5; R5 = 109.5; 
Y6 = 104.5; R6 =  73;   
Y7 = 121.5; R7 =  36.5; 
Y8 = 158;   R8 =  17;   

% Calibration points in the image
%--------------------------------
u1 = 45; v1 = 154;
u2 = 136; v2 = 138;
u3 = 171; v3 = 91;
u4 = 213; v4 = 38;
u5 = 263; v5 = 41;
u6 = 327; v6 = 102;
u7 = 382; v7 = 156;
u8 = 484; v8 = 185;

f = [u1 v1 u2 v2 u3 v3 u4 v4 u5 v5 u6 v6 u7 v7 u8 v8]';

% Calibration matrix
%-------------------
D = [  
    Y1 R1  1   0  0  0 -u1*Y1 -u1*R1;
     0  0  0  Y1 R1  1 -v1*Y1 -v1*R1;
    Y2 R2  1   0  0  0 -u2*Y2 -u2*R2;
     0  0  0  Y2 R2  1 -v2*Y2 -v2*R2;
    Y3 R3  1   0  0  0 -u3*Y3 -u3*R3;
     0  0  0  Y3 R3  1 -v3*Y3 -v3*R3;
    Y4 R4  1   0  0  0 -u4*Y4 -u4*R4;
     0  0  0  Y4 R4  1 -v4*Y4 -v4*R4;
    Y5 R5  1   0  0  0 -u5*Y5 -u5*R5;
     0  0  0  Y5 R5  1 -v5*Y5 -v5*R5;
    Y6 R6  1   0  0  0 -u6*Y6 -u6*R6;
     0  0  0  Y6 R6  1 -v6*Y6 -v6*R6;
    Y7 R7  1   0  0  0 -u7*Y7 -u7*R7;
     0  0  0  Y7 R7  1 -v7*Y7 -v7*R7;
    Y8 R8  1   0  0  0 -u8*Y8 -u8*R8;
     0  0  0  Y8 R8  1 -v8*Y8 -v8*R8];
    
% Calculate C via the pseudo inverse
%-----------------------------------

c = [pinv(D)*f; 1];
C = reshape(c,3,3)'

% Check if the point (0,17) is correctly transformed
%---------------------------------------------------
test = C * [0 17 1]';
test(1) = test(1)/test(3);
test(2) = test(2)/test(3)


