function imgBlur = myConv2(img, kernel, condition)

Isize = size(img);
ksize = size(kernel);
sz = Isize + ksize - 1;

% implement convolution
imgBlur = zeros(sz(1), sz(2));
%     kernel = kernels(:,:,cc);   % normalized done
K = fft2(kernel, sz(1), sz(2));
I = fft2(img, sz(1), sz(2));
imgBlur = ifft2(K.*I);

switch condition
    case 'valid'
        outSz = Isize - ksize + 1;
        imgBlur = imgBlur(ksize(1)+1: outSz(1)+ksize(1),ksize(2)+1: outSz(2)+ksize(2));
    case 'same'
        xx = floor(ksize(1)/2);
        yy = floor(ksize(2)/2);
        imgBlur = imgBlur(xx+1:xx+Isize(1), yy+1:yy+Isize(2));
    case 'full'
        imgBlur = imgBlur;            
end

end