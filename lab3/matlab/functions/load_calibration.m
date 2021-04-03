function [A, d, Rs, ts, rpe] = load_calibration(filename)
% Load a set of camera calibration parameters.
%
% [A, d, Rs, ts, rpe] = LOAD_CALIBRATION(filename)
%
% param filename:  .mat file to load
% returns:   A, d, Rs, ts, rpe
%            A is the camera intrinsic matrix
%            d is a list of distortion model coefficients
%            Rs a cell array of camera rotation matrices,
%            ts a cell array of camera translation vectors
%            rpe is the average reprojection error.
data = load(filename);
A = data.A;
d = data.d;
Rs = num2cell(data.Rs, [2,3]);
ts = num2cell(data.ts, 2);
for k = 1:length(Rs)
    Rs{k} = squeeze(Rs{k});
    ts{k} = ts{k}';
end
rpe = data.rpe;

