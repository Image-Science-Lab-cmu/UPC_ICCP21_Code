profile = BM3DProfile();
profile.gamma = 0;
lin_denoised = zeros(size(lin_blur));

%     v = zeros(5,5);
%     v(3:3)=1;
lin_denoised = BM3D(double(lin_blur), sigma, profile);


% lin_denoised = lin_denoised(50:end-50, 50:end-50,:);
lin_denoised = lin_denoised / max(lin_denoised(:));

figure,
subplot(1, 3, 1);
imshow(lin_blur.^(1/2.2));
title('noisy blur');
subplot(1, 3, 2);
imshow(lin_denoised.^(1/2.2));
title('denoised');