function x_out = map_points(H,x)
%MAP_POINTS maps coordinates in matrix x through homography H

x_hom = [x; ones(1,size(x,2))];
x_out = H*x_hom;
x_out= [x_out(1,:)./x_out(3,:); x_out(2,:)./x_out(3,:)];

end

