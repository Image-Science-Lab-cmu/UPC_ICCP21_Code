function [mtf_x, mtf_y, mtf_z, mtf_radial_avg, mtf_radial_min] = compute_mtf(psf)
% compute vertical and horizontal MTF
% psf: N x N 1 channel PSF

[w, h] = size(psf);
psf = psf ./ sum(psf(:)); % normalize by DC component
mtf = abs(fft2(psf));     % compute 2D MTF

mtf_x = mtf(1: floor(w/2), 1);
mtf_y = mtf(1, 1: floor(h/2));

% radially average / min
mtf = abs(ifftshift(fft2(fftshift(psf))));
[xx, yy] = meshgrid(1:w, 1:h);
xx = floor(abs(xx - (w/2+1)));
yy = floor(abs(yy - (h/2+1)));
dists = round(sqrt(xx.^2 + yy.^2));
mtf_radial_avg = zeros(1, w/2);
mtf_radial_min = zeros(1, w/2);
for dist = 0: w/2-1
    mtf_radial_avg(dist+1) = mean(mtf(dists == dist));
    mtf_radial_min(dist+1) = min(mtf(dists == dist));
end

% rotate mtf by 45
mtf = abs(ifftshift(fft2(fftshift(psf))));
mtf_rot = imrotate(mtf, 45);
[w, h] = size(mtf_rot);
% todo: make sure DC component is 1
mtf_z = mtf_rot(floor(w/2)+1: w, floor(h/2)+1);
end