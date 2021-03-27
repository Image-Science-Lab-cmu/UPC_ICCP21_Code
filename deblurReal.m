% This script implements the camera pipeline for RAW images captured under 
% different display layouts, based on
% Anqi Yang, Aswin Sankaranarayanan, "Designing Design Pixel 
% Layouts for Under-Panel Cameras", ICCP 2021.
%
%
% For a specific scene in the "data/images" folder, there are RAW images
% captured under different display layouts. "data/PSFs" folder contains
% pre-measured blur kernels for these displays. For each RAW image, we
% process it by the following steps:
%    1. load RAW image, blur kernel, color correction information;
%    2. downsample demosaicked image to 1k;
%    3. denoise using BM3D [1]
%    4. Wiener deblurring
%    5. color correction, gamma correction
%
% We use BM3D code from http://www.cs.tut.fi/~foi/GCF-BM3D/index.html#ref_software
% [1] Y. M??kinen, L. Azzari, A. Foi, 2019.
% Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
% In IEEE International Conference on Image Processing (ICIP), pp. 185-189
%
% @Anqi Yang, March 18, 2021.


close all; clear; clc;
addpath utils
addpath bm3d_matlab_package
addpath bm3d_matlab_package/bm3d


% TODO: specify scene folders under 'data/images/'
sceneName = 'toy2';
srcDir = sprintf('data/images/%s', sceneName);
fprintf(['Deblurring for \"', sceneName, ...
    '\" under the following display layouts:\n\n']);

% pre-measured PSF names under various displays
psfNames = {'TOLED_Repeat_150dpi', ...
    'TOLED_Random_150dpi', ...
    'L2_Repeat_150dpi', ...
    'L2_Random_150dpi', ...
    'L2Inv_Repeat_150dpi', ...
    'L2Inv_Random_150dpi', ...
    'TOLED_Repeat_300dpi', ...
    'TOLED_Random_300dpi', ...
    'POLED_Repeat_300dpi', ...
    'L2_Random_300dpi', ...
    'L2Inv_Random_300dpi'};


for id = 1: length(psfNames)
    
    
    % - - - load blur image, pre-measured PSF, color correction info - - -
    imgPath = [srcDir, '/', psfNames{id}, '.dng'];
    psfPath = ['data/PSFs/', psfNames{id}, '.mat'];
    
    if exist(imgPath, 'file') == 0; continue; end
    fprintf([' --> ', psfNames{id}, '\n']);
    
    [lin_blur,~,info] = raw_process(imgPath);
    load(psfPath); lin_psf = lin_hdr; clear lin_hdr;
    load([srcDir, '/colorCalibrate.mat']);
    
    
    % - - - spatially downsample 4k to 1k - - -
    scale = 4;
    lin_blur = imfilter(lin_blur, ones(scale,scale)/scale^2);
    lin_blur = lin_blur(1:scale:end, 1:scale:end, :);
    
    
    % - - - denoise by BM3D - - -
    noiseProfile = BM3DProfile();
    noiseProfile.gamma = 0;
    noiseProfile.Nstep = 6;
    noiseProfile.Ns = 25;
    noiseProfile.tau_match = 2500;
    noiseProfile.lambda_thr3D = 2.7;
    noiseProfile.Nstep_wiener = 5;
    noiseProfile.N2_wiener = 16;
    noiseProfile.Ns_wiener = 25;
    noise_type =  'gw';
    noise_var = 0.0001; % Noise variance  # selfie and onePiece
    seed = 0; % seed for pseudorandom noise realization
    [~, PSD, ~] = getExperimentNoise(noise_type, noise_var, seed, size(lin_blur));
    lin_blur = CBM3D(lin_blur, PSD, noiseProfile);
    
    
    % - - - wiener deconvolution - - -
    epsilon = 0.037;  % since PSFs are nomarlized to 1, we fix \epsilon.
    SNR = 35;
    [lin_deblur, ~] = myWienerDeconv(lin_blur, lin_psf, 35);
    
    
    % - - - color correction - - -
    lin_blur_srgb = apply_cmatrix(lin_blur, cam2rgb);
    lin_deblur_srgb = apply_cmatrix(lin_deblur, cam2rgb);
    
    
    % - - - gamma correction - - -    
    nl_blur = gamma_correction(single(lin_blur_srgb), offset, gamma, c);
    nl_deblur = gamma_correction(single(lin_deblur_srgb), offset, gamma, c);
    nl_psf = imread(['data/PSFs/', psfNames{id}, '.png']);
    
    
    % - - - save results - - -
    figure(1), imshow(nl_psf);
    figure(2), imshow(nl_blur, []);
    figure(3), imshow(nl_deblur,[]);

    dstDir = ['results/', sceneName];
    if exist(dstDir, 'dir') == 0; mkdir(dstDir); end
    imwrite(nl_blur, ['results/', sceneName, '/', psfNames{id}, '.png']);
    imwrite(nl_deblur, ['results/', sceneName, '/', psfNames{id}, '_deblurred.png']);
    
end

