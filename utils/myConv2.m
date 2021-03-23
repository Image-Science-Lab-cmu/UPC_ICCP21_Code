function imgBlur = myConv2(img, kernels)

Isize = size(img);
ksize = size(kernels);
sz = Isize - ksize + 1;

% implement convolution
imgBlur = zeros(Isize(1)*2, Isize(2)*2, 3);
for cc = 1:3
    kernel = kernels(:,:,cc);   % normalized done
    K = fft2(kernel, Isize(1)*2, Isize(2)*2);
    I = fft2(img(:,:,cc), Isize(1)*2, Isize(2)*2);
    imgBlur(:,:,cc) = ifft2(K.*I);
end

xx = floor(ksize(1)/2);
yy = floor(ksize(2)/2);
imgBlur = imgBlur(xx+1:xx+Isize(1), yy+1:yy+Isize(2), :);

end