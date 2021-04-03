function x = hom(x)
% Convert a matrix of column vectors to homogeneous coordinates.
%
% x = HOM(x)
x = [x; ones(1,size(x,2))];