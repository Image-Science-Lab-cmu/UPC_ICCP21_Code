function [psf, openRatio] = computePSF_3smooth(srcDir, dstDir, option, threshold, unitPatternSize, delta1)

% This function compute PSF for RGB sensor channels.
% For each channel, we compute a weighted sum of 
% mulitple wavelength.


% optical system parameters
% simulation parameters in UDC paper
% f = 0.01;                    % camera focal length [m]
% D1 = 4e-3;                   % diam of lens aperture [m]

% Note: To generate PSF in Figure 5, 
% please use the following f and D1.
% Huawei P30 camera
f = 5.56e-3;                % camera focal length [m]
D1 = f/1.6;                 % diam of lens aperture [m]

% delta1                    % spacing of the lens plane [m]
% delta2                    % spacing of the sensor plane [m]
pixelSize = 2e-6;           % sensor pixel pitch [m]
delta2 = 0.5e-6;            % spacing at sensor plane [m]

% a set of wavelengths
load('sensorResponseCurve.mat');  % load wavelengths and weights

minM = floor(wvl_Bs(1)*f/delta1/delta2/2)*2;
maxM = floor(wvl_Rs(end)*f/delta1/delta2/2)*2;

D2 = maxM*delta2;           % diam of sensor region of interest [m]

%% init circle aperture
x1 = (-maxM/2: maxM/2-1) * delta1;
[X1, Y1] = meshgrid(x1,x1);
% u1 = rect(X1/(D1)) .* rect(Y1/(D1));  % src field
u1 = circ(X1/(D1), Y1/(D1));

% Initialize display pattern
if exist(sprintf('data/pixelPatterns/%s.png', srcDir))
    path = sprintf('data/pixelPatterns/%s.png', srcDir);
else
    print('Pixel pattern does not exist.'); return;
end
display = load_aperture(path, maxM*delta1, D1, maxM, unitPatternSize, option, threshold);
openRatio = mean(display(:));

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

% - - - downsample psf to fit sensor pitch - - -
r=floor(pixelSize/delta2);
psf = imfilter(psf, ones(r,r)/r^2);
psf = psf(1:r:end, 1:r:end, :);

% - - - normalize PSF - - - 
for cc = 1:3
    psf(:,:,cc) = psf(:,:,cc) / sum(sum(psf(:,:,cc)));
end

return;
    
end
