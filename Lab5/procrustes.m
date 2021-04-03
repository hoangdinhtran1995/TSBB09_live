function R = procrustes(p1, p2)
%PROCRUSTES estimates a rotation matrix R between p1 and p2

% homogeneous
p1_hom = [p1; ones(1,size(p1,2))];
p2_hom = [p2; ones(1,size(p2,2))];

p1_us = zeros(size(p1_hom));
p2_us = p1_us;

% unit sphere for each point
for i = 1:size(p1_hom,2)
p1_us(:,i) = p1_hom(:,i)/sqrt(sum(p1_hom(:,i).^2));
p2_us(:,i) = p2_hom(:,i)/sqrt(sum(p2_hom(:,i).^2));
end
p1_us_norm = vecnorm(p1_us)
p2_us_norm = vecnorm(p2_us)

[U, S, V] = svd (p2_us*p1_us'); 

R = U*V';

end

