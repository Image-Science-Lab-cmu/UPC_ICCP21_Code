function [u1] = load_display(name,L1, w, M, s, option, threshold)
% load display patterns.
% name: figure name
% L1: diam of the source plane [m]
%     (display and padding region)
% w:  diam of display [m]
% M:  number of samples
% s:  diam of figure pattern [m]

% read in pattern
pattern=im2double(imread(name));
% pattern=ones(21,21);

% different ways to repeat pattern
switch option
    case 'rot90'
        I90 = imrotate(pattern,-90);
        I=[pattern,I90;I90,pattern];
        display = repmat(I, [ceil(L1/s/2), ceil(L1/s/2), 1]);
    case 'rot90flip'
        I90 = imrotate(pattern,-90);
        Iflip = fliplr(pattern);
        I90flip = flipud(I90);
        I=[pattern,I90;Iflip, I90flip];
        display = repmat(I, [ceil(L1/s/2), ceil(L1/s/2), 1]);
    case 'randomRot90'
        % randomly rotate and flip
        repNum = ceil(L1/s);
        display = zeros(repNum*21, repNum*21, 3);
        for ix = 1: 21:repNum*21
            for iy = 1:21:repNum*21
                display(ix:ix+20, iy:iy+20, :) = randomRotAndFlip(pattern);
            end
        end
    case 'hexagon'
        load('fig/aperture_hexagon_single/hexagons.mat');
        s = 38*8e-6;
        repNum = ceil(L1/s);
        display = zeros(repNum*38, repNum*38, 3);
        
        ids = -1;
        for ix = 13: 21: repNum*38
            for iy =  13: 38: repNum*38
                %         ids = randi(6);
                ids = ids + 1;
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12, :) = display(ix-12:ix+12, iy-12:iy+12, :) + hexagons(:,:,ids);
            end
        end
        ids = 0;
        for ix = 24: 21: (repNum-1)*38
            for iy =  32: 38: (repNum-1)*38
                %         ids = randi(6);
                ids = ids + 1;
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12) = display(ix-12:ix+12, iy-12:iy+12) + hexagons(:,:,ids);
            end
        end
    otherwise
        % tile the pattern to display
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

