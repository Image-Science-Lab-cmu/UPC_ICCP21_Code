function [u1] = load_aperture(name,L1, w, M, s, option, threshold)
% load display patterns.
% name: figure name
% L1: diam of the source plane [m]
%     (display and padding region)
% w:  diam of aperture [m]
% M:  number of samples
% s:  diam of figure pattern [m]

% read in pattern
if strcmp(option, 'randomPatterns')
    load(name);
    T = size(pattern, 1);
    
    % to compare different DPIs
    delta1 = L1 / M;
    T_expected = ceil(s/delta1);
    dilation = T_expected / T;
    for cc = 1: size(pattern, 3)
        pattern_new(:,:,cc) = kron( pattern(:,:,cc), ones(dilation, dilation) );
    end
    pattern = pattern_new; clear pattern_new;
    T = T_expected;
else
    pattern=im2double(imread(name));
    T = size(pattern, 1);
    
    % to compare different DPIs
    delta1 = L1 / M;
    T_expected = ceil(s/delta1);
    dilation = T_expected / T;
    pattern = kron( pattern, ones(dilation, dilation) );
    T = T_expected;
end



% different ways to repeat pattern
switch option
    case 'oneMorePixel'
        % randomly rotate and flip, remove even row/col pixels
        repNum = ceil(L1/s);
        display = zeros(repNum*(T+1), repNum*(T+1));
        for ix = 1: 1:repNum
            for iy = 1:1:repNum
                ixx = (ix-1)*(T+1);
                iyy = (iy-1)*(T+1);
                display(ixx+1:ixx+T, iyy+1:iyy+T) = randomRotAndFlip(pattern);

            end
        end
    case 'randomPatterns'
        patternNum = size(pattern, 3);
        repNum = ceil(L1/s);
        display = zeros(repNum*T, repNum*T);
        for rr = 1: repNum
            for ss = 1:repNum
                u = randi(patternNum);
                display(1+(rr-1)*T: rr*T, 1+(ss-1)*T: ss*T) = pattern(:,:,u); 
            end
        end
    case 'random'
        % this mode is for TOLED only
        pattern = circshift(pattern, [0,-10]);
        T0 = 15;
        repNum = ceil(L1/s);
        display = zeros(repNum*T, repNum*T);
        
        for ix = 1: T:repNum*T
            for iy = 1:T:repNum*T
                offsetH = randi(T0) - 1;
                display(ix:ix+T-1, iy:iy+T-1) = circshift(pattern, [0, offsetH]);
            end
        end
    case 'randomFlip'
        % this mode is for TOLED only
        pattern = circshift(pattern, [0,-floor(T/2)]);
        repNum = ceil(L1/s);
        display = zeros(repNum*T, repNum*T);
        
        for ix = 1: T:repNum*T
            for iy = 1:T:repNum*T
                display(ix:ix+T-1, iy:iy+T-1) = randomFlip(pattern);
            end
        end
    case 'rot90'
        I90 = imrotate(pattern,-90);
        I=[pattern,I90;I90,pattern];
        display = repmat(I, [ceil(L1/s/2), ceil(L1/s/2)]);
    case 'rot90flip'
        I90 = imrotate(pattern,-90);
        Iflip = fliplr(pattern);
        I90flip = flipud(I90);
        I=[pattern,I90;Iflip, I90flip];
        display = repmat(I, [ceil(L1/s/2), ceil(L1/s/2)]);
    case 'randomRot90'
        % randomly rotate and flip
        repNum = ceil(L1/s);
        display = zeros(repNum*T, repNum*T);
        for ix = 1: T:repNum*T
            for iy = 1:T:repNum*T
                display(ix:ix+T-1, iy:iy+T-1) = randomRotAndFlip(pattern);
            end
        end
    case 'localRandom'
        load 'fig/20201125-014953/random_order.mat';
        localNum = size(order, 1);
        display = zeros(localNum*T, localNum*T);
        for ix = 1: 1:localNum
            for iy = 1:1:localNum
                display((ix-1)*T+1:ix*T, (iy-1)*T+1:iy*T) = rotAndFlip(pattern, order(ix, iy));
            end
        end
        repNum = ceil(L1/(localNum * s));
        display = repmat(display, [repNum, repNum]);
    case 'hexagon'
        load('fig/aperture_hexagon_single/hexagons.mat');
        s = 38*8e-6;
        repNum = ceil(L1/s);
        display = zeros(repNum*38, repNum*38);
        
        ids = -1;
        for ix = 13: 21: repNum*38-12
            for iy =  13: 38: repNum*38-12
                ids = ids + 1;
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12) = display(ix-12:ix+12, iy-12:iy+12) + hexagons(:,:,ids);
            end
        end
        ids = 0;
        for ix = 24: 21: (repNum-1)*38-12
            for iy =  32: 38: (repNum-1)*38-12
                ids = ids + 1;
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12) = display(ix-12:ix+12, iy-12:iy+12) + hexagons(:,:,ids);
            end
        end
    case 'randomHexagon'
        load('fig/aperture_hexagon_single/hexagons.mat');
        s = 38*8e-6;
        repNum = ceil(L1/s);
        display = zeros(repNum*38, repNum*38);
        
        ids = -1;
        for ix = 13: 21: repNum*38-12
            for iy =  13: 38: repNum*38-12
                ids = randi(6);
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12) = display(ix-12:ix+12, iy-12:iy+12) + hexagons(:,:,ids);
            end
        end
        ids = 0;
        for ix = 24: 21: (repNum-1)*38-12
            for iy =  32: 38: (repNum-1)*38-12
                ids = randi(6);
                ids = mod(ids, 6) + 1;
                display(ix-12:ix+12, iy-12:iy+12) = display(ix-12:ix+12, iy-12:iy+12) + hexagons(:,:,ids);
            end
        end
    otherwise
        % tile the pattern to display
        display = repmat(pattern, [ceil(L1/s), ceil(L1/s)]);
end 

% figure out how to resize the pattern
display = display(1:M,1:M);

% load hann window
% w = hann(M);
% win = w * w';
% display = display .* win;

% output binary field
% todo
u1 = display;
% if threshold == 0
%     u1 = round(display);
% else
%     u1 = round(display > threshold);%0.55 for optimized pattern
% end

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

function g = rotAndFlip(h, order)
    switch order
        case 0
            g = h;
        case 1 
            g = h';
        case 2
            g = h(:, end:-1:1);
        case 3
            g = h';
            g = g(end:-1:1, :);
    end
           
            
end

function g = randomFlip(h)
    if rand(1) > 0.5
        h = fliplr(h);
    end
%     if rand(1) > 0.5
%         h = flipud(h);
%     end
    g = h;
end

