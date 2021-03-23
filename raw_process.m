% This script implements the workflow for reading and processing a RAW
% image file in MATLAB that is described in
% http://users.soe.ucsc.edu/~rcsumner/papers/RAWguide.pdf
%
% Variable names and code are taken directly from the PDF for easy
% reference. Requires supplemental functions srgb_gamma(), apply_cmatrix(),
% and wbmask().
%
% Both methods for reading a RAW file (DCRAW and MATLAB) are presented
% below. Note the former requires user inputs from the DCRAW command line
% output/source code. The latter is fully automatic.
%
% Since RAW files tend to be quite large, some memory management is
% included. If you run into OUT OF MEMORY errors, try cropping the initial
% 'raw' array to a smaller size. Make sure you crop from the top left so
% that the Bayer arrangement doesn't change. If you have extra memory and
% want to view the variables along the way, try replacing the clear
% statements with imshow() statements.
%
% This script has been tested with .CR2 and .NEF RAW files. If you run into
% an error, please let me know at robert.c.sumner@gmail.com
%
% Rob Sumner, UC Santa Cruz, Sept 2013

% clear,clc,
% close all
function [lin_rgb, raw, meta_info] = raw_process(filename)
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - Enter the filename and choose reading approach, 'DNG' or 'DCRAW', and
% the Bayer pixel arrangement of camera, 'rggb','bggr','gbrg' or 'grbg' - -
% filename = 'ISO50_Exp00400.dng';
read_app = 'DNG';               % 'DNG' for .dng  or  'DCRAW' for .tiff
bayer_type = 'bggr';
warning('off', 'imageio:tiffmexutils:libtiffWarning');
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Define transformation matrix from sRGB space to XYZ space for later use
srgb2xyz = [0.4124564 0.3575761 0.1804375;
    0.2126729 0.7151522 0.0721750;
    0.0193339 0.1191920 0.9503041];

if strcmpi(read_app,'DCRAW')
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % % % - - - - - Reading TIFF image from DCRAW output - - - - - % % %
    % Following user inputs required! Default values are for accompanying
    % BANANA_SLUG.CR2 file, from DCRAW informational output.
    black = 2047;
    saturation = 13584;
    wb_multipliers = [2.007813 1 1.657227];
    xyz2cam = [ 6444 -904 -893;
        -4563 12308 2535;
        -903 2016 6728]/10000;
    
    % - - - Read image into MATLAB - - -
    raw = single(imread(filename));
    
    % - - - Linearize - - -
    lin_bayer = (raw-black)/(saturation-black);
    lin_bayer = max(0,min(lin_bayer,1));
    clear raw
    
    % - - - White Balance - - -
    mask = wbmask(size(lin_bayer,1),size(lin_bayer,2),...
                            wb_multipliers,bayer_type);
    balanced_bayer = lin_bayer .* mask;
    clear lin_bayer mask
    
    
elseif strcmpi(read_app,'DNG')
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % % % - - - - Reading DNG file from DNG Converter output - - - - % % %
    % MATLAB r2011a or later required.
    
    % - - - Reading file - - -
    warning off MATLAB:tifflib:TIFFReadDirectory:libraryWarning
    t = Tiff(filename,'r');
%     offsets = getTag(t,'SubIFD');
%     setSubDirectory(t,offsets(1));
    raw = read(t);
    close(t);
    meta_info = imfinfo(filename);
    x_origin = meta_info.ActiveArea(2) + 1;
    width = meta_info.DefaultCropSize(1);
    y_origin = meta_info.ActiveArea(1)+1;
    height = meta_info.DefaultCropSize(2);
    raw =double(raw(y_origin:y_origin+height-1,x_origin:x_origin+width-1));


    % - - - Linearize - - -
    if isfield(meta_info,'LinearizationTable')
        ltab=meta_info.LinearizationTable;
        raw = ltab(raw+1);
    end
    black = meta_info.BlackLevel(1);
    saturation = meta_info.WhiteLevel;
%     todo: anqi (1209)
    raw(2:2:end,2:2:end) = min(raw(2:2:end,2:2:end), 1800);    %r
    raw(1:2:end,1:2:end) = min(raw(1:2:end,1:2:end), 1800);    %b
    lin_bayer = (raw-black)/(saturation-black);
    lin_bayer = max(0,min(lin_bayer,1));
%     clear raw
    
    % - - - White Balance - - -
    wb_multipliers = (meta_info.AsShotNeutral).^-1;
    wb_multipliers = wb_multipliers/wb_multipliers(2);
    mask = wbmask(height,width,wb_multipliers,bayer_type);
    balanced_bayer = lin_bayer .* mask;
    clear lin_bayer mask
    
    % - - - Color Correction Matrix from DNG Info - - -
    temp = meta_info.ColorMatrix1;
    xyz2cam = reshape(temp,3,3)';
    
else
    error('Invalid Read Approach.')
end


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% % % - - - - - The rest of the processing chain - - - - -

% - - - Demosaicing - - -
temp = uint16(balanced_bayer/max(balanced_bayer(:))*2^16);
lin_rgb = single(demosaic(temp,bayer_type))/65535;
clear balanced_bayer temp

% % - - - Color Space Conversion - - -
% rgb2cam = xyz2cam * srgb2xyz;
% rgb2cam = rgb2cam ./ repmat(sum(rgb2cam,2),1,3);
% cam2rgb = rgb2cam^-1;
% 
% lin_srgb = apply_cmatrix(lin_rgb,cam2rgb);
% lin_srgb = max(0,min(lin_srgb,1));
% clear lin_rgb
% 
% % - - - Brightness and Gamma - - -
% grayim = rgb2gray(lin_srgb);
% grayscale = 0.25/mean(grayim(:));
% bright_srgb = min(1,lin_srgb*grayscale);
% % clear lin_srgb grayim
% 
% nl_srgb = bright_srgb.^(1/2.2);
% 
% % - - - Display output - - -
% imshow(nl_srgb)
