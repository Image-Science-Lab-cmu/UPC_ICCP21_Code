function [psf, openRatio] = computePSF_3smooth(h1, phaseOpt, srcDir, dstDir, option, threshold, unitPatternSize, delta1)

% optical system parameters
% todo: UDC paper
f = 0.01;
D1 = 4e-3;                   % diam of lens aperture [m]
% delta1 = 8e-6;               % spacing of the lens plane [m]
% delta2 = 0.5e-6;             % spacing of the sensor plane [m]
% unitPatternSize = 168e-6;
pixelSize = 2e-6;
% f = 5.56e-3;
% D1 = f/1.6;
% delta1 = 15e-6;
delta2 = 0.5e-6;
% unitPatternSize = 315e-6;

% a set of wavelengths
load('sensorResponseCurve.mat');  % load wavelengths and weights

% wvl_Rs=[0.64e-6];           % wavelength
% wvl_Gs=[0.52e-6];
% wvl_Bs=[0.45e-6];
% weight_Rs = [1];
% weight_Gs = [1];
% weight_Bs = [1];

minM = floor(wvl_Bs(1)*f/delta1/delta2/2)*2;
%todo: Feb 20.
% minM = floor(0.47e-6*f/delta1/delta2/2)*2;  % the same size in computePSF_3
maxM = floor(wvl_Rs(end)*f/delta1/delta2/2)*2;

% k=2*pi/(lambda);             % wavenumber
% todo: 0.01 is current simulation
% f=0.01;                      % focal length (m)
% f=0.006;                     % focal length (m)

D2 = maxM*delta2;              % diam of sensor region of interest[m]

%% init circle aperture
x1 = (-maxM/2: maxM/2-1) * delta1;
[X1, Y1] = meshgrid(x1,x1);
% u1 = rect(X1/(D1)) .* rect(Y1/(D1));  % src field
u1 = circ(X1/(D1), Y1/(D1));

% todo: @Jan 19
if exist(sprintf('fig/%s/display_pattern_refined.png', srcDir))
    path = sprintf('fig/%s/display_pattern_refined.png', srcDir);
elseif exist(sprintf('fig/%s/display_pattern_refined.mat', srcDir))
    path = sprintf('fig/%s/display_pattern_refined.mat', srcDir);
else
    path = sprintf('fig/%s/display_pattern.png', srcDir);
end
display = load_aperture(path, maxM*delta1, D1, maxM, unitPatternSize, option, threshold);

u1 = u1 .* display;
I1 = abs(u1.^2);                           % src irradiance
figure,imshow(I1,[]);

%% lens propagation

u2_R = zeros(minM, minM);
u2_G = zeros(minM, minM);
u2_B = zeros(minM, minM);

% average response for R sensor
% Note: number of samples M_R for each wavelength depends on wvl_Rs and
% thus differs a bit. I initialize the input plane using the largest
% possilbe size maxM, and crop to M_R and then feed to propFF.
% After propFF, I need each u2_R to have the same number of samples,
% so I cropped them to the smallest possible minM.
for wvlId = 1:length(wvl_Rs)
    M_R = floor(wvl_Rs(wvlId)*f/delta1/delta2/2)*2; % number of samples
    crop_R = floor((maxM - M_R) / 2);
    u1_R = u1(crop_R+1:end-crop_R, crop_R+1:end-crop_R);
    [u2_R_curr, ~, ~] = propFF(u1_R,M_R*delta1,wvl_Rs(wvlId),f,0);
    crop_R = floor((M_R - minM)/2);
    test = abs(u2_R_curr(crop_R+1:end-crop_R, crop_R+1:end-crop_R) * weight_Rs(wvlId));
    u2_R = u2_R + test .^2;
end

% average response for G sensor
for wvlId = 1:length(wvl_Gs)
    M_G = floor(wvl_Gs(wvlId)*f/delta1/delta2/2)*2; % number of samples
    crop_G = floor((maxM - M_G) / 2);
    u1_G = u1(crop_G+1:end-crop_G, crop_G+1:end-crop_G);
    [u2_G_curr, ~, ~] = propFF(u1_G,M_G*delta1,wvl_Gs(wvlId),f,0);
    crop_G = floor((M_G - minM)/2);
    test = abs(u2_G_curr(crop_G+1:end-crop_G, crop_G+1:end-crop_G) * weight_Gs(wvlId));
    u2_G = u2_G + test .^2;
end

% average response for B sensor
for wvlId = 1:length(wvl_Bs)
    M_B = floor(wvl_Bs(wvlId)*f/delta1/delta2/2)*2; % number of samples
    crop_B = floor((maxM - M_B) / 2);
    u1_B = u1(crop_B+1:end-crop_B, crop_B+1:end-crop_B);
    [u2_B_curr, ~, ~] = propFF(u1_B,M_B*delta1,wvl_Bs(wvlId),f,0);
    crop_B = floor((M_B - minM)/2);
    test = abs(u2_B_curr(crop_B+1:end-crop_B, crop_B+1:end-crop_B) * weight_Bs(wvlId));
    u2_B = u2_B + test.^2;
end

u2 = cat(3, u2_R, ...
            u2_G, ...
            u2_B);
psf = u2;
% psf=abs(u2);
% psf=psf.^2;

% mid = floor(size(psf, 1)/2);
% figure,
% plot(log(psf(mid, :, 2)));
% ylim([-25, 15]);

% - - - downsample psf to fit sensor pitch - - -
r=floor(pixelSize/delta2);
psf = imfilter(psf, ones(r,r)/r^2);
psf = psf(1:r:end, 1:r:end, :);

% - - - save display pattern - - -
pattern = display(1:100, 1:100);
imwrite(pattern, sprintf('%s/display_pattern_binary.png', dstDir));

% - - - compute the sharpness of psf - - -
m = (size(psf, 1) - 1)/2;
n = (size(psf, 2) - 1)/2;
[X, Y] = meshgrid(-m:m, -n:n);
sigma=2;
gauss = exp(-(X.^2+Y.^2)/2/sigma^2);
gauss = gauss / sum(gauss(:));

% - - - normalize PSF - - - 
for cc = 1:3
    psf(:,:,cc) = psf(:,:,cc) / sum(sum(psf(:,:,cc)));
end

imgpsf=mean(psf, 3);
sharpness=sum(sum((imgpsf .* gauss)));
variance = compute_var(psf);
openRatio = mean(round(display(:)));
fprintf('open area=%.4f, sharpness=%.8f std=%.4f\n', openRatio, sharpness, sqrt(variance));

% - - - draw vertical/horizontal MTFs - - -
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
    [mtf_x, mtf_y, mtf_z, ~, mtf_radial_min] = compute_mtf(psf(:,:,cc));
    
    mtf_radial_mins(:,:,cc)=mtf_radial_min;
    
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
save(sprintf('%s/mtf_radial.mat', dstDir), 'mtf_radial_mins');


% load color display
% if exist(sprintf('fig/%s/display_pattern_rgb.png', srcDir))
%     path = sprintf('fig/%s/display_pattern_rgb.png', srcDir);
%     display = load_display(path, M_R*delta1, D1, M_R, 168e-6, option, threshold);
%     imwrite(display(1:126,1:126,:), sprintf('%s/display_pattern_rgb.png', dstDir));
%     figure,imshow(display(1:126,1:126,:));
%     
%     path = sprintf('fig/%s/display_pattern_rgb_only.png', srcDir);
%     display = load_display(path, M_R*delta1, D1, M_R, 168e-6, option, threshold);
%     imwrite(display(1:126,1:126,:), sprintf('%s/display_pattern_rgb_only.png', dstDir));
%     figure,imshow(display(1:126,1:126,:));
%     
% end

% new way of load color display
% if exist(sprintf('fig/%s/display_pattern_refined_shifted.png', srcDir))
%     path = sprintf('fig/%s/display_pattern_refined_shifted.png', srcDir);
%     display = load_display2(path, M_R*delta1, D1, M_R, 168e-6, option, threshold);
%     imwrite(display(1:126,1:126,:), sprintf('%s/display_pattern_rgb_new.png', dstDir));
%     figure,imshow(display(1:126,1:126,:));
%     
% %     path = sprintf('fig/%s/display_pattern_rgb_only.png', srcDir);
% %     display = load_display(path, M_R*delta1, D1, M_R, 168e-6, option, threshold);
% %     imwrite(display(1:126,1:126,:), sprintf('%s/display_pattern_rgb_only.png', dstDir));
% %     figure,imshow(display(1:126,1:126,:));
    
end
