function nl_img = gamma_correction(lin_rgb, offset, gamma, c)

% % - - - Color Space Conversion - - -
% srgb2xyz = [0.4124564 0.3575761 0.1804375;
%     0.2126729 0.7151522 0.0721750;
%     0.0193339 0.1191920 0.9503041];
% 
% % temp = [2.2100 -1.3320 -0.1416, ... 
% %            -0.1895 1.2275 -0.0547, ...
% %            -0.0234 0.1826  0.6035];
% xyz2cam = [
%     1.0684   -0.2988   -0.1426;
%    -0.4316    1.3555    0.0508;
%    -0.1016    0.2441    0.5859];
% 
% rgb2cam = xyz2cam * srgb2xyz;
% rgb2cam = rgb2cam ./ repmat(sum(rgb2cam,2),1,3);
% cam2rgb = rgb2cam^-1;
% 
% lin_srgb = apply_cmatrix(lin_rgb,cam2rgb);
% lin_srgb = max(0,min(lin_srgb,1));

% - - - Gamma correction - - - 
% linrgb = max(0, lin_rgb-offset);
% grayim = rgb2gray(lin_rgb);
% grayscale = c/mean(grayim(:));
% bright_img = min(1,lin_rgb*grayscale);
% % clear grayim
% 
% nl_img = bright_img.^gamma;

% lin_hsv = rgb2hsv(lin_rgb); 
% lin_hsv(:,:,2)=(lin_hsv(:,:,2)).^(1/2.2); 
% lin_hsv(lin_hsv>1)=1; 
% lin_rgb = hsv2rgb(lin_hsv);
% 
% offset = 0.08;
% gamma = 1/1.5;
nl_img = max(0, c*lin_rgb-offset) .^(gamma);

% nl_img = localtonemap(lin_srgb);
% nl_img = lin_srgb;

% nl_img = lin_srgb .^ (1/2.2); % deblur using WF real kernel


end