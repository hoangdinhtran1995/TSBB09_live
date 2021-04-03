function x = rodrigues(x)
% Convert a Rodrigues rotation vector to a rotation matrix or vice versa.
%
% x = RODRIGUES(x)

if isequal(size(x), [3,3])
    x = rotationMatrixToVector(x);
else
    x = rotationVectorToMatrix(x);
end
