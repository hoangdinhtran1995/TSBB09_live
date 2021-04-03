function [cb_inner_corners, cb_tile_size, im_size, model_points, image_points] = ...
    load_calibration_input(data_path)
% Load chessboard model points, detected image points and image dimensions
% from all .mat files in the given data_path.
%
% [cb_inner_corners, cb_tile_size, im_size, model_points, image_points] = ...
%    LOAD_CALIBRATION_INPUT(data_path)

files = dir(fullfile(data_path, '*.mat'));

% Filter out calibration*.mat result files

to_remove = [];
for k = 1:length(files)
    if strfind(files(k).name, 'calibration') == 1
        to_remove = [to_remove, k];  
    end
end
files(to_remove) = [];
nfiles = length(files);

% Load data

model_points = cell(nfiles,1);
image_points = cell(nfiles,1);
im_size = 0;

for k = 1:nfiles
    file = load(fullfile(data_path, files(k).name));
    if im_size == 0
        cb_inner_corners = file.cb_inner_corners;
        cb_tile_size = file.cb_tile_size;
        im_size = file.im_size;
    end
    % Cells of matrices (one per image), each shaped (2,N)
    image_points{k} = double(file.image_points');
    % Like image_points but shaped (3,N)
    model_points{k} = double(file.model_points');
end

