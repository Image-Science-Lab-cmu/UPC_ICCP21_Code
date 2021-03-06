function [psf, openRatio] = computePSF_3(srcPath, dstDir, option, threshold, unitPatternSize, delta1)
% This function compute PSF of RGB channel
% using only peak wavelength.
% 
%
% optical system parameters
% used in simulation experiments
f = 0.01;                     % camera focal length [m]
D1 = 4e-3;                    % diam of the lens/disp aperture [m]

wvl_R=0.61e-6;                % peak wavelength of R channel
wvl_G=0.53e-6;                % peak wavelength of G channel
wvl_B=0.47e-6;                % peak wavelength of B channel

delta2 = 0.5e-6;              % diam of the lens aperture [m]
M_R = floor(wvl_R*f/delta1/delta2/2)*2; % number of samples
M_G = floor(wvl_G*f/delta1/delta2/2)*2; % number of samples
M_B = floor(wvl_B*f/delta1/delta2/2)*2; % number of samples

D2 = M_R*delta2;              % diam of sensor region of interest[m]

%% init square aperture
x1 = (-M_R/2: M_R/2-1) * delta1;
[X1, Y1] = meshgrid(x1,x1);
% u1 = rect(X1/(D1)) .* rect(Y1/(D1));  % src field
u1 = circ(X1/(D1), Y1/(D1));

% Initialize display pattern
if ~exist(srcPath, 'file')
    error('Pixel pattern does not exist.');
end
display = load_aperture(srcPath, M_R*delta1, D1, M_R, ...
            unitPatternSize, option, threshold);
u1 = u1 .* display;

% height map of phase mask (all zeros)
h1 = zeros(size(u1));

I1 = abs(u1.^2);                           % src irradiance
figure,imshow(I1,[]);

crop_G = floor((M_R - M_G)/2);
crop_B = floor((M_R - M_B)/2);

Phi_R = 2*pi/(wvl_R)*(1.4-1)*h1;
u1_R = u1 .* exp(1i*Phi_R);

Phi_G = 2*pi/wvl_G * (1.4-1)*h1;
u1_G = u1 .* exp(1i*Phi_G);
u1_G = u1_G(crop_G+1:end-crop_G, crop_G+1:end-crop_G);

Phi_B = 2*pi/wvl_B * (1.4-1)*h1;
u1_B = u1 .* exp(1i*Phi_B);
u1_B = u1_B(crop_B+1:end-crop_B, crop_B+1:end-crop_B);

%% lens propagation
[u2_R, ~, ~] = propFF(u1_R,M_R*delta1,wvl_R,f,0);
[u2_G, ~, ~] = propFF(u1_G,M_G*delta1,wvl_G,f,0);
[u2_B, ~, ~] = propFF(u1_B,M_B*delta1,wvl_B,f,0);

crop_R = floor((M_R - M_B)/2);
crop_G = floor((M_G - M_B)/2);

u2 = cat(3, u2_R(crop_R+1:end-crop_R, crop_R+1:end-crop_R), ...
            u2_G(crop_G+1:end-crop_G, crop_G+1:end-crop_G), ...
            u2_B);
psf=abs(u2);
psf=psf.^2;

% downsample psf to fit sensor pitch
r=4;
psf = imfilter(psf, ones(r,r)/r^2);
psf = psf(1:r:end, 1:r:end, :);

% normalize PSF
for cc = 1:3
    psf(:,:,cc) = psf(:,:,cc) / sum(sum(psf(:,:,cc)));
end

%% save tiled display pattern
L = 800e-6;
N = floor(L/delta1);
pattern = display(1:N, 1:N);
imwrite(pattern, sprintf('%s/display_pattern_binary.png', dstDir));

openRatio = mean(round(display(:)));
fprintf('%s: Open ratio=%.4f\n', srcPath, openRatio);

%% visualize MTF
colors=[1,0,0; 0,1,0; 0,0,1];
close all;
% figure(3), clf(figure(3), 'reset');hold on
figure('Renderer', 'painters', 'Position', [10, 10, 600, 500]); hold on;
grid on;
set(gca, 'FontSize', 30);
set(gcf,'Color',[1 1 1], 'InvertHardCopy','off');
ylabel('Contrast'), xlabel('Line pairs per pixel'), ylim([0,1]),xlim([0,0.5]),
% title('Horizontal MTF');
hold off;

% figure(4), clf(figure(4), 'reset'); hold on
figure('Renderer', 'painters', 'Position', [10, 10, 600, 500]); hold on;
grid on;
set(gca, 'FontSize', 30);
set(gcf,'Color',[1 1 1], 'InvertHardCopy','off');
ylabel('Contrast'), xlabel('Line pairs per pixel'), ylim([0,1]),xlim([0,0.5]),
% title('Vertical MTF');
hold off;

figure('Renderer', 'painters', 'Position', [10, 10, 600, 500]); hold on;
grid on;
set(gca, 'FontSize', 30);
set(gcf,'Color',[1 1 1], 'InvertHardCopy','off');
ylabel('Contrast'), xlabel('Line pairs per pixel'), ylim([0,1]),xlim([0,0.5]),
% title('Vertical MTF');
hold off;

for cc = 1: 3
    [mtf_x, mtf_y, mtf_z, mtf_radial_avg, mtf_radial_min] = compute_mtf(psf(:,:,cc));
    
    mtf_radial_mins(:,:,cc)=mtf_radial_min;
    mtf_radial_avgs(:,:,cc)=mtf_radial_avg;
    
    lenN = length(mtf_x);
    figure(1), hold on;
    plot(linspace(0, 1/2, lenN), mtf_x, 'LineWidth', 2, 'Color', colors(cc,:)); hold off;
    
    figure(2), hold on;
    lenN = length(mtf_y);
    plot(linspace(0, 1/2, lenN), mtf_y, 'LineWidth', 2, 'Color', colors(cc,:)); hold off;
    
    figure(3), hold on;
    lenN = length(mtf_z);
    plot(linspace(0, 1/2, lenN), mtf_z, 'LineWidth', 2, 'Color', colors(cc,:)); hold off;

end
% todo
figure(1), hold on; saveas(gcf, sprintf('%s/mtf_vertical.pdf', dstDir)); hold off;
figure(2), hold on; saveas(gcf, sprintf('%s/mtf_horizontal.pdf', dstDir)); hold off;
figure(3), hold on; saveas(gcf, sprintf('%s/mtf_diagonal.pdf', dstDir)); hold off;
save(sprintf('%s/mtf_radial.mat', dstDir), 'mtf_radial_avgs', 'mtf_radial_mins');

