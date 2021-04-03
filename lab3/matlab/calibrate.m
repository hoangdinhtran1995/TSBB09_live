addpath 'F:\kurser\master\tsbb09\Lab 3\matlab\functions'

% Path to data from find_points.py
data_path = 'F:\kurser\master\tsbb09\Lab 3\data\output';
[~, ~, im_size, model_points, image_points] = load_calibration_input(data_path);

% TODO: Implement this function
[A, d, Rs, ts, rpe] = calibrate_zhang(model_points, image_points);

% Print results

fprintf('--- Zhang''s calibration ---\n\n')

fprintf('Intrinsic matrix:\n')
disp(A)
fprintf('Distortion coefficients:\n')
disp(d)
fprintf('Mean reprojection error: %.2f pixels\n', rpe)

% Save the calibration for later use

save(fullfile(data_path, 'calibration_zhang.mat'), 'A', 'd', 'im_size', 'Rs')

