function [] = mainCompareDPIs(patternIds)
addpath utils
addpath fig
addpath ../RealCaptures/scripts/bm3d_matlab_package/
addpath ../RealCaptures/scripts/bm3d_matlab_package/bm3d
% clear;
% close all;

phaseOpt = 'no';
folder = 'corrected_setting_invertibility';

% close all;
h1 = randn(100, 100) * 10e-6;
for i=1:100
    for j=1:100
        if mod(i,2)==0 && mod(j,2)==0
            h1(i,j)=0;
        end
    end
end
h1 = kron(h1, ones(21,21));


srcDirs = {
    'aperture_toled_single',   
    'aperture_toled_single',   
    'aperture_toled_single',
    
    'aperture_toled_single',   
    'aperture_toled_single',   
    'aperture_toled_single',
    
    'aperture_poled_single',    % new: poled 336
    'aperture_poled_single',    % new: poled 336
    'aperture_poled_single',
    
    'aperture_poled_single',
    'aperture_poled_single',
    'aperture_poled_single',
    
    '20201125-145433',         % L2+inv random rotate
    '20201125-145433',         % L2+inv random rotate
    '20201125-145433',         % L2+inv random rotate
    
%     '20201125-031442',         % L2 random rotate 
%     '20201125-031442',         % L2 random rotate 
%     '20201125-031442',         % L2 random rotate 
%     
%     '20201127-155503'          % L2+inv random rotate
%     '20201127-163420'          % 
};
dstDirs = {
    'aperture_toled_single_336um',
%     'aperture_toled_single_252um',
    'aperture_toled_single_168um',
    'aperture_toled_single_84um',
    
    'aperture_toled_randomRot90_336um',
%     'aperture_toled_randomRot90_252um',
    'aperture_toled_randomRot90_168um',
    'aperture_toled_randomRot90_84um',
    
    'aperture_poled_single_336um',
    'aperture_poled_single_168um',    %8
    'aperture_poled_single_84um',
    
    'aperture_poled_randomRot90_336um',
    'aperture_poled_randomRot90_168um',  %11
    'aperture_poled_randomRot90_84um',
    
    '20201125-145433_randomRot90_336um',         % L2+inv random rotate
%     '20201125-145433_randomRot90_252um',         % L2+inv random rotate
    '20201125-145433_randomRot90_168um',         % L2+inv random rotate
    '20201125-145433_randomRot90_84um',          % L2+inv random rotate
        
%     '20201125-031442_randomRot90_336um',         % L2 random rotate   
%     '20201125-031442_randomRot90_168um',         % L2 random rotate
%     '20201125-031442_randomRot90_84um',          % L2 random rotate
%     
%     '20201127-155503_randomRot90_84um'          % L2+inv random rotate
%     '20201127-163420_randomRot90_84um'
};
tileOptions = {'', '',  '', ...
            'randomRot90','randomRot90', 'randomRot90', ...
            '',  '', '', ...
            'randomRot90', 'randomRot90', 'randomRot90', ...
            'randomRot90', 'randomRot90', 'randomRot90', ...
%             'randomRot90', 'randomRot90', 'randomRot90', 'randomRot90', ...
            };
thresholds =   [0, 0, 0, ...
                0, 0, 0, ...
                0, 0, 0, ...
                0, 0, 0, ...
                0, 0, 0, ...
%                 0, 0, 0, 0, ...
                ];

unitPatternSizes = [336e-6, 168e-6, 84e-6, ...
                    336e-6, 168e-6, 84e-6, ...
                    672e-6, 336e-6, 168e-6, ...
                    672e-6, 336e-6, 168e-6, ...
                    336e-6, 168e-6, 84e-6, ...
%                     336e-6, 252e-6, 168e-6, 84e-6, ...
                    ];
delta1s = 4e-6 * ones(1, length(dstDirs));
% delta1s = 3e-6 * ones(1, length(dstDirs));

% patternIds = 4:4;
SNRs = 24:4:40;
Ls = [273, 654, 1608, 4005, 10024];
noise_vars = [0.005, 0.002, 0.002, 0.001, 0.0002];
ssims = zeros(length(patternIds), length(SNRs));
psnrs = zeros(length(patternIds), length(SNRs));

% todo: pixelSize 
for radius = 10:10
    radius = 10;
    pixelSize = 2e-6;
    ratio=0.3;
    scores=[];
for id = patternIds
     
    % todo
%     dstDir = sprintf('CompareDPI_results/%s/', dstDirs{id});
%     dstDir = sprintf('test_MultiWavelength_results/%s/', dstDirs{id});
    dstDir = sprintf('Sims_image_results_SSIM1.5/%s/', dstDirs{id});
    mkdir(dstDir);
    if id > length(unitPatternSizes)
        unitPatternSize = 168e-6;
        delta1 = 8e-6;
    else
        unitPatternSize = unitPatternSizes(id);
        delta1 = delta1s(id);
    end
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
    imagesc(PSFs_x, PSFs_y, PSFs(:,:,2));colormap jet;c=colorbar;c.FontSize=30;
    saveas(gcf, sprintf('%s/%s_PSFs.png', dstDir, phaseOpt));    
    hold off
%     
% 	figure, imshow((kernels ./ max(kernels(:)) * 2000).^(1/2.2),[]);
%     saveas(gcf, sprintf('%s/%s_PSFs_RGB.png', dstDir, phaseOpt));

    % plot auto-correlation function (FFT of PSFs)
    kernel = kernels(:,:,2);
    K = ifftshift(fft2(fftshift(kernel)));
    figure, hold on; axis equal, axis tight;
    imagesc(PSFs_x, PSFs_y, abs(K));colormap jet; c=colorbar;c.FontSize=30;
    saveas(gcf, sprintf('%s/%s_Autocorrelation.png', dstDir, phaseOpt));   
    hold off
    
        % compute Invertible score (Wiener K .* wienerK --> low-30)
    % move to the front, consistent with Python
    kernel = mean(kernels, 3);
%   kernel = kernels(:,:,3);
    K = ifftshift(fft2(fftshift(kernel)));
    df = conj(K) ./ (abs(K).^2 + 0.015);
    kk = abs(K .* df);
    imgkk = abs(ifftshift(fft2(fftshift(kk))));
    imgkk = log(imgkk / max(imgkk(:)));

    num_kk = prod(size(kk));
    sorted_kk = sort(kk(:), 'ascend');
    wiener_score = mean(sorted_kk(1:ceil(ratio*num_kk)));
    fprintf('%s wiener score %.10f\n ', srcDirs{id}, wiener_score);
 
%     continue;
    
    % apply PSFs to images with different noise level
    % todo
    for noiseId = 1:length(SNRs)
        
        SNR = [];
        if noiseId > 0
            % noise modeling
            sensor.capacity = 15506;
            sensor.noise_std = 4.87;
            SNR = SNRs(noiseId);
            L = Ls(noiseId);
            sensor.gain = 1/L;
        end
        
        mean_loss = 0;
        mean_psnr = 0;
        mean_loss_top10 = 0;
        mean_ssim = 0;
%         todo
        imIds = 1:30;

        for imId = imIds
            fprintf('%s SNR=%d imId=%d...\n', dstDir, SNR, imId);
            % read image
            img = im2double(imread(sprintf('fig/HQ/test/%02d.png', imId)));
            img = img ./ max(img(:)); % normalize img to [0,1]
            img = img .* (openRatio / refRatio);    
%             todo
%             img = img .* 0.8;
%             xstart = 250; ystart = 250;
%             img = img(xstart:xstart+512-1, ystart:ystart+512-1,:);
            % scale img brightness according to open ratio w.r.t TOLED open
           
            Isize = size(img);
            ksize = size(kernels);

            % convolve sharp image with PSF
            imgBlur = myConv2(img, kernels);

            % add noise
            if noiseId == 0
                imgBlurnoisy = imgBlur;
            else
                imgBlurnoisy = add_noise(imgBlur * L, 1, sensor);
            end
%             figure(1), imshow(imgBlurnoisy, []);
            if mod(imId-1, 5) == 0
                imwrite(imgBlurnoisy, sprintf('%s/%d_SNR%d_blur_wiener_img.png',dstDir, imId, SNR));
            end
                
            % --- denoise (BM)---
            % ICCP DPI comparison
%             noiseProfile = BM3DProfile();
%             noiseProfile.gamma = 0; 
%             noiseProfile.Nstep = 6; 
%             noiseProfile.Ns = 25;
%             noiseProfile.tau_match = 2500;
%             noiseProfile.lambda_thr3D = 2.7;
%             noiseProfile.Nstep_wiener = 5;
%             noiseProfile.N2_wiener = 16;
%             noiseProfile.Ns_wiener = 25;
%             noise_type =  'gw';
%             % todo
%             noise_var = noise_vars(noiseId); % Noise variance
%             seed = 0; % seed for pseudorandom noise realization
%             [~, PSD, ~] = getExperimentNoise(noise_type, noise_var, seed, size(imgBlurnoisy));
%             imgBlurnoisy = CBM3D(imgBlurnoisy, PSD, noiseProfile);
%             
            % --- denoise (BM)---
            noiseProfile = BM3DProfile();
            noiseProfile.gamma = 0; 
            noise_type =  'gw';
            noise_var = noise_vars(noiseId); % Noise variance
            seed = 0; % seed for pseudorandom noise realization
            [~, PSD, ~] = getExperimentNoise(noise_type, noise_var, seed, size(imgBlurnoisy));
            imgBlurnoisy = CBM3D(imgBlurnoisy, PSD, noiseProfile);

            % wiener deconvolution
            % todo
            [imgSharp, nsr] = myWienerDeconv(imgBlurnoisy, kernels, 35);
%             figure(2),imshow(imgSharp);
            if mod(imId-1, 5) == 0
                imwrite(imgSharp, sprintf('%s/%d_SNR%d_deblur_wiener_img.png',dstDir, imId, SNR));
            end

            % ================ %
            %       Losses     %
            % ================ %

            % SNR and SSIM
            psnrVal = psnr(imgSharp, img);
%             [ssimVal, ~] = ssim(imgSharp, img, 'Radius', radius); % ICCP
            [ssimVal, ~] = ssim(imgSharp, img, 'Radius', 1.5);

            mean_psnr = mean_psnr + psnrVal;
            mean_ssim = mean_ssim + ssimVal;

            % L2 loss 
            l2 = norm(img(:)-imgSharp(:), 2) / prod(size(img));
            mean_loss = mean_loss + l2;    

            % top10 L2 loss
            residual = abs(img(:)-imgSharp(:));
            sorted = sort(residual, 'descend');
            num_top = ceil(0.1 * prod(size(img)));
            top_residual = sorted(1:num_top);
            mean_loss_top10 = mean_loss_top10 + sum(top_residual .^2 ) / num_top;

%             % Invertible (Wiener K .* wienerK --> top10)
% %             kernel = mean(kernels, 3);
%             kernel = kernels(:,:,2);
%             K = ifftshift(fft2(fftshift(kernel)));
%             df = conj(K) ./ (abs(K).^2 + nsr);
%             kk = abs(K .* df);
%             imgkk = abs(ifftshift(fft2(fftshift(kk))));
%             imgkk = log(imgkk / max(imgkk(:)));
%             
%             num_kk = prod(size(kk));
% %             sorted_kk = sort(kk(:), 'ascend');
% %             wiener_score = mean(sorted_kk(1:ceil(ratio*num_kk)));
%             wiener_scores = sort((1-kk(:)).^2, 'descend');
%             wiener_score=mean(wiener_scores(1:ceil(ratio*num_kk)));
%             figure, hold on
%             subplot(221), plot(kk(147,:)); title('WF kernel in fourier (normal scale)');
%             subplot(222), imshow(kk,[]);
%             subplot(223), plot(imgkk(147,:)); title('WF kernel in spatial (log-scale)');
%             subplot(224), imshow(imgkk,[]);
%             suptitle(sprintf('%s, noiseSNR=%d', dstDirs{id}, SNR));

        end
        if noiseId >  0
            fprintf('\n%s SNR=%d(dB) wiener=%.8f mean_l2=%.8f mean_l2_top10=%.8f, mean_psnr=%.2f, ssim=%.4f\n', ...
                dstDirs{id},SNRs(noiseId), wiener_score, mean_loss/length(imIds), mean_loss_top10/length(imIds), ...
                mean_psnr/length(imIds), mean_ssim/length(imIds));
            ssims(id, noiseId) = mean_ssim/length(imIds);
            psnrs(id, noiseId) = mean_psnr/length(imIds);
        else
            fprintf('\n%s wiener=%.8f mean_l2=%.8f mean_l2_top10=%.8f, mean_psnr=%.2f, ssim=%.4f\n', ...
                    dstDirs{id}, wiener_score, mean_loss/length(imIds), mean_loss_top10/length(imIds), ...
                    mean_psnr/length(imIds), mean_ssim/length(imIds));
        end
        % save current results
        curr_ssims = ssims(id,:);
        curr_psnrs = psnrs(id,:);
        save(sprintf('%s/sweep_snr.mat',dstDir), 'SNRs', 'curr_ssims', 'curr_psnrs');
    end
    curr_ssims = ssims(id,:);
    curr_psnrs = psnrs(id,:);
    save(sprintf('%s/sweep_snr.mat',dstDir), 'SNRs', 'curr_ssims', 'curr_psnrs');
%     save(sprintf('%s/sweepRadius_%d_snr.mat',dstDir, radius), 'SNRs', 'curr_ssims', 'curr_psnrs');
end
% figure, hold on;
% subplot(121), plot(psnrs(patternIds,:)');
% subplot(122), plot(ssims(patternIds,:)');
% title(sprintf('ssim radius=%d', radius));
% legend(dstDirs{patternIds});
% hold off;
end