function [] = simulate(patternIds)
addpath utils
addpath fig
addpath bm3d_matlab_package/
addpath bm3d_matlab_package/bm3d
% clear;close all;clc;

% \PATH\TO\PIXELPATTERNS
srcDir = 'data/pixelPatterns/';
patternNames = {
    'TOLED',                                    % TOLED Repeat
    'TOLED',                                    % TOLED Random
    
    'L2_Repeat',                                % L2 Repeat
    'L2Inv_Repeat',                             % L2Inv Repeat
    
	'L2_Random',                                % L2 Random
    'L2Inv_Random',                             % L2Inv Random
    };

dstDirs = { 
    
    'TOLED_Repeat',                             % TOLED Repeat
    'TOLED_Random',                             % TOLED Random
    
    'L2_Repeat',                                % L2 Repeat
    'L2Inv_Repeat',                             % L2Inv Repeat
    
	'L2_Random',                                % L2 Random
    'L2Inv_Random',                             % L2Inv Random
      
};
tileOptions = {'', 'randomRot90', ...
    '', '',...
    'randomRot90','randomRot90','oneMorePixel', ...
};
thresholds =   [ 0, 0,...
                 0, 0, ...
                 0, 0];
if ~exist('patternIds', 'var')
    patternIds = 1: length(dstDirs);
end

SNRs = 24:4:40;
Ls = [273, 654, 1608, 4005, 10024];
noise_vars = [0.005, 0.002, 0.002, 0.001, 0.0002];
pixelSize = 2e-6;
ssims = zeros(1, length(SNRs));
psnrs = zeros(1, length(SNRs));


for id = patternIds
     
    fprintf('\n\n- - - - - - - - - -\n');
    dstDir = sprintf('results/simulation/%s/', dstDirs{id});
    mkdir(dstDir);
    srcPath = [srcDir, patternNames{id}, '.png']; % path to display pattern
    
    % in this script, we only test DPI=150
    unitPatternSize = 168e-6;       % Size of unit display pattern [m]
    delta1 = 8e-6;                  % Sample spacing on lens plane [m]
    % delta1 = 4e-6;                % Sample spacing on lens plane [m]

    % Note: 
    % computePSF_3       --- computes PSF of peak wavelengths for RGB channel.
    %                        We use this function in simualtion evaluation.
    % computePSF_3smooth --- computes multi-wavelength PSF for RGB channel
    %                        To reproduce PSFs in Figure 5, please use 
    %                        this version.
    [PSFs, openRatio] = computePSF_3(srcPath, dstDir, ...
        tileOptions{id}, thresholds(id), unitPatternSize, delta1); 
    refRatio = 0.2072;
    kernels = PSFs;
    
    %% visualize PSFs
    maxVal = max(PSFs(:));
    PSFs = PSFs/maxVal;
    PSFs = log(PSFs);    
    
    figure, hold on; axis equal, axis tight;
    PSFs_x = (1:size(PSFs,1))*pixelSize;
    PSFs_y = (1:size(PSFs,2))*pixelSize;
    imagesc(PSFs_x, PSFs_y, PSFs(:,:,2));colormap jet; colorbar;
%     saveas(gcf, sprintf('%s/%s_PSFs.png', dstDir, phaseOpt));   
    hold off
    
%     figure, imshow((kernels ./ max(kernels(:)) * 2000).^(1/2.2),[]);
%     saveas(gcf, sprintf('%s/%s_PSFs_RGB.png', dstDir, phaseOpt));
    
    % plot auto-correlation function (FFT of PSFs)
    kernel = kernels(:,:,2);
    K = ifftshift(fft2(fftshift(kernel)));
    figure, hold on; axis equal, axis tight;
    imagesc(PSFs_x, PSFs_y, log(abs(K)+1));colormap jet; colorbar;
%     saveas(gcf, sprintf('%s/%s_Autocorrelation.png', dstDir, phaseOpt));   
    hold off

    continue;
    
    % Evaluate PSF on images
    for noiseId = 1: length(SNRs)
        

        % noise modeling
        sensor.capacity = 15506;
        sensor.noise_std = 4.87;
        SNR = SNRs(noiseId);
        L = Ls(noiseId);
        sensor.gain = 1/L;

        
        mean_psnr = 0;
        mean_ssim = 0;
        imIds = 1:30;

        for imId = imIds
            fprintf('%s SNR=%d imId=%d...\n', dstDir, SNR, imId);
            
            % - - - read image - - -
	    % TODO: we conduct evaluation on thirty ground-truth 
            % sharp images UDC dataset.
            % https://yzhouas.github.io/projects/UDC/udc.html
            % You can also download the images we used from here:
            % https://drive.google.com/file/d/1BuRzeFdPueT7rGIMLCswftx-UjNMvCWr/view?usp=sharing
            UDC_DATASET_VAL = 'data/simulation_test';
            img = im2double(imread(sprintf('%s/%02d.png', UDC_DATASET_VAL, imId)));
            img = img ./ max(img(:)); % normalize img to [0,1]
            img = img .* (openRatio / refRatio);
          
            % ================================================= %
            %  Simulate capturing an image under display panel  %
            % ================================================= %
            
            % blur sharp image by PSF
            for cc = 1: 3
                imgBlur(:,:,cc) = myConv2(img(:,:,cc), kernels(:,:,cc), 'same');
            end
            
            % add read out and shot noise
            imgBlurNoisy = add_noise(imgBlur * L, 1, sensor);
            clear imgBlur;

            % save degraded image
            if mod(imId-1, 5) == 0
                imwrite(imgBlurNoisy, sprintf('%s/%d_SNR%d_blur_wiener_img.png',dstDir, imId, SNR));
            end
            
            
            % ================================================= %
            %               Reconver a sharp image              %
            % ================================================= %
            % denoise (BM3D)
            noiseProfile = BM3DProfile();
            noiseProfile.gamma = 0; 
            noise_type =  'gw';
            noise_var = noise_vars(noiseId); % Noise variance
            seed = 0; % seed for pseudorandom noise realization
            [~, PSD, ~] = getExperimentNoise(noise_type, noise_var, seed, size(imgBlurNoisy));
            imgBlurDenoised = CBM3D(imgBlurNoisy, PSD, noiseProfile);

            % wiener deconvolution
            [imgSharp, ~] = myWienerDeconv(imgBlurDenoised, kernels, 35);     % deblur in paper
            clear imgBlurDenoised;
            
            % save recovered sharp image
            if mod(imId-1, 5) == 0
                imwrite(imgSharp, sprintf('%s/%d_SNR%d_deblur_wiener_img.png',dstDir, imId, SNR));
            end

            % ================================================= %
            %               Compute PSNR and SSIM               %
            % ================================================= %

            % SNR and SSIM
            psnrVal = psnr(imgSharp, img);
            [ssimVal, ~] = ssim(imgSharp, img, 'Radius', 1.5);

            mean_psnr = mean_psnr + psnrVal;
            mean_ssim = mean_ssim + ssimVal;
            
        end
        

        fprintf('\n%s SNR=%d(dB) mean_psnr=%.2f, ssim=%.4f\n', ...
            dstDirs{id},SNRs(noiseId), mean_psnr/length(imIds), mean_ssim/length(imIds));
        ssims(noiseId) = mean_ssim/length(imIds);
        psnrs(noiseId) = mean_psnr/length(imIds);
        
        % save current results
        curr_ssims = ssims;
        curr_psnrs = psnrs;
        % todo
        save(sprintf('%s/sweep_snr.mat',dstDir), 'SNRs', 'curr_ssims', 'curr_psnrs');
    end
    curr_ssims = ssims;
    curr_psnrs = psnrs;
    % todo
    save(sprintf('%s/sweep_snr.mat',dstDir), 'SNRs', 'curr_ssims', 'curr_psnrs');
end
