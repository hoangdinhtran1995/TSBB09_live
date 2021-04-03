function N = normalization_matrix(x)
% Generate a Hartley normalization matrix.
%
% N = NORMALIZATION_MATRIX(x)
%
% param x: An array of 2D points shape = (2,N)
% return: The matrix that normalizes the points,
% i.e subtracts the mean and divides by the standard deviation.

m = mean(x, 2);
s = sqrt(2) ./ std(x, 0, 2);

N = [[s(1), 0, -s(1) * m(1)];
     [0, s(2), -s(2) * m(2)];
     [0, 0, 1]];
