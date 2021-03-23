function [u1] = load_display2(name,L1, w, M, s, option, threshold)
% load display patterns.
% name: figure name
% L1: diam of the source plane [m]
%     (display and padding region)
% w:  diam of display [m]
% M:  number of samples
% s:  diam of figure pattern [m]

% read in pattern
pattern=im2double(imread(name));
pattern = cat(3, pattern, pattern, pattern);
% pattern=ones(21,21);

% RGB subpixels
color=im2double(imread('colorRGB.png'));
xmin=5; xmax=17;
ymin=5; ymax=17;
color = round(imresize(color, [xmax-xmin+1, ymax-ymin+1]));

% different ways to repeat pattern
switch option
    case 'randomRot90'
        % randomly rotate and flip
        repNum = ceil(L1/s);
        display = zeros(repNum*21, repNum*21, 3);
        for ix = 1: 21:repNum*21
            for iy = 1:21:repNum*21
                currPattern = randomRotAndFlip(pattern);
                currPattern(xmin:xmax, ymin:ymax, :) = currPattern(xmin:xmax, ymin:ymax, :) + color;
                currPattern(currPattern > 1) = 1;
                display(ix:ix+20, iy:iy+20, :) = currPattern;
            end
        end
    otherwise
        % tile the pattern to display
        pattern(xmin:xmax, ymin:ymax, :) = pattern(xmin:xmax, ymin:ymax, :) + color;
        pattern(pattern > 1) = 1;
        display = repmat(pattern, [ceil(L1/s), ceil(L1/s)]);
end 

% figure out how to resize the pattern
display = display(1:M,1:M, :);

% load hann window
% w = hann(M);
% win = w * w';
% display = display .* win;

% output binary field
% u1 = (display>0.9);%0.55 for optimized pattern
if threshold == 0
    u1 = round(display);
else
    u1 = round(display > threshold);%0.55 for optimized pattern
end

end

function g = randomRotAndFlip(h)
    if rand(1) > 0.5
        h = imrotate(h, 90);
    end
    if rand(1) > 0.5
        h = fliplr(h);
    end
    if rand(1) > 0.5
        h = flipud(h);
    end
    g = h;
end

