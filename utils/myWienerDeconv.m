function [imgSharp, nsr] = myWienerDeconv(imgBlurnoisy, kernels, SNR)

Isize = size(imgBlurnoisy);
ksize = size(kernels);
imgSharp = zeros(Isize(1)*2, Isize(2)*2, 3);
imgSharpFFT = zeros(Isize(1)*2, Isize(2)*2, 3);
for cc = 1:3
    kernel = kernels(:,:,cc); % normalized done
    K = fft2(kernel, Isize(1)*2, Isize(2)*2);
    
    im = imgBlurnoisy(:,:,cc);
    if isempty(SNR)
%         nsr = 0.0001 / var(im(:));
        nsr = 0.01;
        nsr = 0.015; % Jan 4th
    else
        nsr = min(1, 1 / (SNR / 20) ^ 10);
    end
    df = conj(K) ./ (abs(K).^2 + nsr);
    I = fft2(imgBlurnoisy(:,:,cc), Isize(1)*2, Isize(2)*2);
    g = ifftshift(ifft2(I .* df));
    imgSharp(:,:,cc) = g;
    imgSharpFFT(:,:,cc) = I;
end

xx = floor(Isize(1) - ksize(1) / 2);
yy = floor(Isize(2) - ksize(2) / 2);
imgSharp = imgSharp(xx+1:xx+Isize(1), yy+1:yy+Isize(2), :);
imgSharp(imgSharp < 0) = 0;

end