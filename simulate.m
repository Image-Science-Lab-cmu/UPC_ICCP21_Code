function [] = mainComputePSF(patternIds, phaseOpt)
addpath utils
addpath fig
addpath bm3d_matlab_package/
addpath bm3d_matlab_package/bm3d
% clear;close all;clc;


srcDirs = {
    'aperture_toled_single',
    'aperture_toled_single',
    
    '20201006-140308_Display_L2_Repeat',  % top10 L2 (thres: 0.45)
    '20201006-153518_Display_L2Inv_Repeat',  % top10 L2 + invertible
    
	'20201125-031442_Display_L2_Random',         % L2 random rotate
    '20201125-145433_Display_L2Inv_Random',         % 8. L2+inv random rotate        
    };

dstDirs = { 
    
    'aperture_toled_single',
    'aperture_toled_randomRot90',
    
    '20201006-140308_Display_L2_Repeat',  % top10 L2 (thres: 0.45)
    '20201006-153518_Display_L2Inv_Repeat',  % top10 L2 + invertible
    
	'20201125-031442_Display_L2_Random',         % L2 random rotate
    '20201125-145433_Display_L2Inv_Random',         % 8. L2+inv random rotate
      
};
tileOptions = {'', 'randomRot90', ...
    '', '',...
    'randomRot90','randomRot90','oneMorePixel', ...
};
thresholds =   [ 0, 0,...
                 0, 0, ...
                 0, 0];
area = [0.20, 0.25, 0.20, 0.33,];

SNRs = 24:4:40;
Ls = [273, 654, 1608, 4005, 10024];
noise_vars = [0.005, 0.002, 0.002, 0.001, 0.0002];
pixelSize = 2e-6;
ssims = zeros(1, length(SNRs));
psnrs = zeros(1, length(SNRs));

ratio=0.3;
scores=[];
for id = patternIds
     
    fprintf('\n\n- - - - - - - - - -\n');
    dstDir = sprintf('results/simulation/%s/', dstDirs{id});
    mkdir(dstDir);
    
    % in this script, we only test of DPI=150
    unitPatternSize = 168e-6;
    delta1 = 8e-6;  % ICCP21
%     todo
%     delta1 = 4e-6;    % @Feb 20   

    % todo:
    [PSFs, openRatio] = computePSF_3(h1, phaseOpt, srcDirs{id}, dstDir, ...
                    tileOptions{id}, thresholds(id), ...
                    unitPatternSize, delta1); 
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
            img = im2double(imread(sprintf('fig/HQ/test/%02d.png', imId)));
            img = img ./ max(img(:)); % normalize img to [0,1]
            img = img .* (openRatio / refRatio);
          

            % ================================================= %
            %  Simulate capturing an image under display panel  %
            % ================================================= %
            
            % blur sharp image by PSF
            imgBlur = myConv2(img, kernels);
            
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
            [~, PSD, ~] = getExperimentNoise(noise_type, noise_var, seed, size(imgBlurnoisy));
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