function [varargout] = column_split(A)
% Split a matrix into columns and distribute them to the output variables.
%
% [a,b,c, ...] = COLUMN_SPLIT(A)

ncols = size(A,2);
for k = 1:ncols
    varargout{k} = A(:,k);
end