function cal_help_all()
% Run 'help' on all the functions in the calibration functions directory.
%
% CAL_HELP_ALL()

path = fileparts(mfilename('fullpath'));
files = dir(fullfile(path, '*.m'));
for k = 1:length(files)
    fprintf('\n')
    help(files(k).name)
    fprintf('--------------------------------------------------------\n');
end