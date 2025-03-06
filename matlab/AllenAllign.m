function Res = AllenAllign(varargin)
% Convert images and ROIs of an SPSIG file to the position of the 
% AllenBrainAtlasOverlay (ABA) image located in 2Pimage/imageprocessing
% 
% USAGE:
% Using software like Adobe Illustrator, The 2photon SPSIG file 
% field of view (FOV) (BImgAverage) should be registered inside the ABA 
% image. This has to be accomplished by moving, rotating and resizing the
% ABA image (including its coordinates) to fit with the 2photon data.
%
% It is essential the 2photon data remains straight, not rotated.
%
% The (straight rectangular/square) FOV should then be used to cut out the
% ABA coordinates (giving us the ABA coordinates to the corresponding
% 2photon data location). This is done by attaching the the clipper mask to
% the 2photon FOV and making only the 2photon FOV visible. 
% The following layers should then be  exported as pngs: 
%   _coordinatesXY.png, _MaskArea.png,
% Optional ABA images to export: 
%       _original, to check against the SPSIG
%       _pRF
% 
% 
% See also: AllenAllignMultiple, 
% 
% Leander de Kraker
% 2023-7-28
% 

% fnXYZ means filenameXYZ
% pnXYZ means pathnameXYZ

% Get SPSIG file
if exist('varargin', 'var') && nargin >= 1
    pnSPSIG = varargin{1}{1};
    fnSPSIG = varargin{1}{2};
    pnSPSIG = [pnSPSIG '/'];
    suppress_plots = true;
else
    [fnSPSIG, pnSPSIG] = uigetfile({'*_SPSIG.mat;*_CASE.mat', 'Data files'; '*.*', 'All Files'}, 'Get ROI Info');
    suppress_plots = false;
    cd(pnSPSIG)
end

% Get ABA images
if exist('varargin', 'var') && nargin >= 2 && ~isempty(varargin{2})
    pnABA = varargin{2}{1};
    fnABA = varargin{2}{2};
    pnABA = [pnABA '/'];
else % Ask user for Coordinates   
    [fnABA, pnABA] = uigetfile('*_CoordinatesXY.png', 'Select coordinatesXY image');
end

%% Load SPSIG file

% pnSPSIG = '\\vs03\VS03-MVP-2\AudioVisual\dataCollection\normRearing\Dieciceis\20230724\FAM\';
% fnSPSIG = 'Dieciceis_20230724_001_normcorr_SPSIG';

% [fnSPSIG, pnSPSIG] = uigetfile('*SPSIG.mat');
% 
load([pnSPSIG fnSPSIG], 'BImgMax', 'BImgAverage', 'BImg', 'PP', 'Mask')


%% Load ABA png/jpg images

% pnABA = 

% [fnABACor, pnABA] = uigetfile('*_coordinatesXY.png', 'Select coordinatesXY image');
%%
fnABABase = strsplit(fnABA, {'_CoordinatesXY.png', '_coordinatesXY'});
fnABABase = fnABABase{1};
fnABApRF  = [fnABABase, '_pRF.png'];
fnABAOrig = [fnABABase, '_Original.png']; 

ABACor = imread([pnABA, fnABA]);
ABACor = double(ABACor)./255; % Coordinates go from 0(top left) to 1(bottom right)


if exist([pnABA fnABApRF], 'file')
    presentpRF = true;
    ABApRF = imread([pnABA fnABApRF]);
else
    presentpRF = false;
    fprintf('"_pRF" not present\n')
end
if exist([pnABA fnABAOrig], 'file')
    presentOrig = true;
    ABAOrig = imread([pnABA fnABAOrig]);
%     ABAOrig = ABAOrig(:,:,1);
else
    presentOrig = false;
    fprintf('"_original" not present, consider saving the FOV via the ABA to check registration\n')
end
fprintf('loaded images\n')

%% Improve coordinate image

% Fix edges of the images since they probably suck.
ABACor(1,:,:) = ABACor(2,:,:);
ABACor(end,:,:) = ABACor(end-1,:,:);
ABACor(:,1,:) = ABACor(:,2,:);
ABACor(:,end,:) = ABACor(:,end-1,:);

% Gaussian filter because the coordinates are rounded and same values, but
% it is better if they are going more smoothly
ABACor = imgaussfilt(ABACor, 10);


% The values should be going smooth and not be ugly
if ~ suppress_plots
    figure('Renderer', 'painters')
    subplot(3,1,1);
    imagesc(ABACor); box off
    title('red=X coordinates. green=Y coordinates')
    subplot(3,2,3)
    plot(ABACor(1,:,1)); hold on; plot(ABACor(end,:,1),'.')
    legend({'first row x coordinates','last row x coordinates'}, 'Location', 'northoutside')
    ylabel('coordinate value')
    subplot(3,2,4)
    plot(ABACor(:,1,2)); hold on; plot(ABACor(:,end,2),'.')
    legend({'first column y coordinates','last column y coordinates'}, 'Location', 'northoutside')
    subplot(3,2,5)
    plot(ABACor(1,:,2)); hold on; plot(ABACor(end,:,2),'.')
    legend({'first row y coordinates','last row y coordinates'}, 'Location', 'northoutside')
    xlabel('row pixel'); ylabel('coordinate value')
    subplot(3,2,6)
    plot(ABACor(:,1,1)); hold on; plot(ABACor(:,end,1),'.')
    legend({'first column x coordinates','last column x coordinates'}, 'Location', 'northoutside')
    xlabel('column pixel');
    
    % If ABA_original is present, use it to check against BImgAverage from SPSIG
    if presentOrig
        figure('Units', 'normalized', 'Position', [0.35 0.25 0.2 0.6])
        subplot(2,1,1)
        imagesc(BImgAverage)
        title('BImgAverage')
        subplot(2,1,2)
        imagesc(ABAOrig)
        title('image from Allan overlay')
    end
end
%% Transpose?
ABACor = permute(ABACor, [2 1 3]);
if presentOrig
    ABAOrig = permute(ABAOrig, [2 1 3]);
end
if presentpRF
    ABApRF = permute(ABApRF, [2 1 3]);
end
fprintf('transposed\n')


%% Flip upside down??
ABACor = flipud(ABACor);
if presentOrig
    ABAOrig = flipud(ABAOrig);
end
if presentpRF
    ABApRF = flipud(ABApRF);
end
fprintf('Flipped images upside down\n')


%% Flip horizontally???
ABACor = fliplr(ABACor);
if presentOrig
    ABAOrig = fliplr(ABAOrig);
end
if presentpRF
    ABApRF = fliplr(ABApRF);
end
fprintf('Flipped images horizontally\n')



%%
clearvars i c n nani nanid1 counter b

%% Load mask
load('AllenBrainAtlasOverlay.mat')

%% Reshape the ABA images to SPSIG images and coordinates
% the ABA images should represent same location as the SPSIG images because
% you cut them out that way in adobe illustrator.

dimsS = size(Mask); % SPSIG image dimensions

ABACor = imresize(ABACor, dimsS, 'bilinear');
if presentOrig
    ABAOrig = imresize(ABAOrig, dimsS, 'bilinear');
end
if presentpRF
    ABApRF = imresize(ABApRF, dimsS, 'bilinear');
end

%% Transform the coordinates of the SPSIG using ABA images
% The coordinates are determined by the color intensity. 
% X(horizontal,dimension2) = first color channel (red) coordinatesXY(:,:,1)
% Y(vertical,dimension 1)  = second color channel(green)coordinatesXY(:,:,2)

x = round(PP.P(1,:));
y = round(PP.P(2,:));
ABAroi = [];
ABAroi.Cnt = PP.Cnt;
ABAroi.X = zeros(1, PP.Cnt);
ABAroi.Y = zeros(1, PP.Cnt);
ABAroi.region = zeros(1, PP.Cnt);
ABAroi.Con = PP.Con;
for i = 1:PP.Cnt
    ABAroi.X(i) = ABACor(y(i), x(i), 1);
    ABAroi.Y(i) = ABACor(y(i), x(i), 2);
    xcon = round(PP.Con(i).x);
    ycon = round(PP.Con(i).y);
    if any(ycon<1)
        fprintf('ROI %d. original y coordinates for contour were <1.\n', i)
        ycon(ycon<1) = 1;
    end
    if any(ycon>dimsS(1))
        fprintf('ROI %d. Original y coordinates were >%d\n', i, dimsS(1))
        ycon(ycon>dimsS(1)) = dimsS(1);
    end
    if any(xcon<1)
        fprintf('ROI %d. original x coordinates for contour were <1.\n', i)
        xcon(xcon<1) = 1;
    end
    if any(xcon>dimsS(2))
        fprintf('ROI %d. Original x coordinates were >%d\n', i, dimsS(2))
        xcon(xcon>dimsS(2)) = dimsS(2);
    end
    for j = 1:length(xcon) 
        ABAroi.Con(i).x(j) = ABACor(ycon(j), xcon(j), 1);
        ABAroi.Con(i).y(j) = ABACor(ycon(j), xcon(j), 2);
    end
    idxi1 = round(ABAroi.Y(i).*dims(1));
    idxi2 = round(ABAroi.X(i).*dims(2));
    ABAroi.region(i) = mask(idxi1, idxi2);
end

x = ABAroi.X * 1024;
y = ABAroi.Y * 1024;
ABAroi.borderDist = CalcDistFromOther(imresize(mask, [1024 1024], 'nearest'), x, y, ABAroi.region);

% Image of the result
if ~ suppress_plots
    figure
    subplot(2,1,1)
    hold off
    him = imagesc([0 1], [0 1], mask);
    % him.C
    hold on
    % Use surf when plotting the images registered to ABA because they can be rotated/ morphed
    % surf(ABACor(:,:,1), ABACor(:,:,2), zeros(dimsS), ABApRF, 'EdgeColor', 'none')
    surf(ABACor(:,:,1), ABACor(:,:,2), zeros(dimsS), BImg, 'EdgeColor', 'none')
    colors = [1 1 1; jet(length(areaNames))];
    colors = colors(ABAroi.region+1, :);
    PlotCon(ABAroi, 'k')
    dSize = ABAroi.borderDist./max(ABAroi.borderDist).*15 + 2;
    scatter(ABAroi.X, ABAroi.Y, dSize, colors, 'filled')
    
    h = subplot(2,1,2);
    histogram(ABAroi.region, -0.5:length(areaNames)+0.5)
    set(h, 'XTick', 0:length(areaNames), 'XTickLabel', ['none', areaNames],'XTickLabelRotation',90)
    ylabel('ROI (n)')
end
%% Save the results to SPSIG
save([pnSPSIG, fnSPSIG], 'ABACor','ABAroi', '-append')
%%




