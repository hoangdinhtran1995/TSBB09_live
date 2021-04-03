%% Load data: Refdata1, Refdata2, Refdata3, Scenedata

load 'Sensor1 Multispectral'\Refdata1.mat
load 'Sensor1 Multispectral'\Refdata2.mat
load 'Sensor1 Multispectral'\Refdata3.mat
load 'Sensor1 Multispectral'\Scenedata.mat

%% A_1

OI_raw=Refdata1(:,:,1);
low_high=stretchlim(OI_raw/16383);
figure(1)
imagesc(OI_raw,[low_high(1) low_high(2)]*16383);
axis image; colormap gray;title('OI raw')
colorbar;
clear part low_high 

%% A_2

%% A_3

%% A_4
