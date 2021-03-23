function [vvar] = compute_var(x)

% Compute the variance of 2D joint distribution
% Input:
%       x: pdf of the discrete distribution
% Output:
%       var: variance of the discrete distribution

[xx, yy] = meshgrid(1:size(x,1), 1:size(x,2));
xx = xx - ceil(size(x,1)/2);
yy = yy - ceil(size(x,2)/2);
dist = xx.^2 + yy.^2;

x = mean(abs(x), 3);
x = x / sum(x(:));
%vvar = mean(dist(:) .* x(:));




mu_x = sum(sum(xx.*x));
mu_y = sum(sum(yy.*x));

v11 = sum(sum( ((xx-mu_x).^2).* x));
v21 = sum(sum( ((xx-mu_x).*(yy-mu_y)).* x));
v22 = sum(sum( ((yy-mu_y).^2).* x));

%A = [ v11 v12; v12 v22];
%var = sum(svd(A));
vvar = v11+v22;
end