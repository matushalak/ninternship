function Res = TrialLock(varargin)
% Extracts responses trial aligned from all exiting time series data
% Leander de Kraker and Chris vander Togt, 2019
% 
%   TrialLock(nameSig, nameSbx, nameLog, stimWindowTimes, chrisEyeExtract, stimDelayms, trialID);
%   Res = TrialLock;
% 
% input (optional):
%   1, nameSig {[2x1] cell or string}: pathname and filename of the SPSIG.mat
%                                    or CASE.mat file
%   2, nameSbx {[2x1] cell or string}: pathname and filename of the sbx.mat
%                                    file, which has the info variable
%   3, nameLog {[2x1] cell or string}: pathname and filename of the log file 
%                                 that holds the stimulus timing variable
%   4, stimWindowTimes {[2x1] double}: time in seconds pre- and post
%   stim.
%   5, chrisEyeExtract (boolean): Do eye tracking calculation? true | false
%   6, stimDelayms ([nTrialType x 1] double): how many miliseconds this
%                               stimulus type is delayed after triggertime.
%   7, trialId ([nTrials x 1] double): Used if different stimulus types have 
%           different ms delays. Contains which stimulus type each trial 
%           belongs to. if left empty, stimDelayms may be scalar value.
% 
% updates:
% 2021 : includes extracting eye and running data
% 2022 : enabled call as function without pop-ups
% 2024 : milisecond delay for specific trials
%           also extract spike_prob & spike_times; output from CASCADE
%      : Changed name from extract to TrialLock
%
%%

clearvars -global info
clearvars info
global info

% Get SPSIG file
if exist('varargin', 'var') && nargin >= 1
    if iscell(varargin{1})
        pnSig = varargin{1}{1};
        fnSig = varargin{1}{2};
    else
        pnSig = '';
        fnSig = varargin{1};
    end
    pnSig = [pnSig '/'];
else
    [fnSig, pnSig] = uigetfile({'*_SPSIG.mat;*_CASE.mat', 'Data files'; '*.*', 'All Files'}, 'Get ROI Info');
end
nameSig = [pnSig, fnSig];
if ~strcmpi(nameSig(end-3:end), '.mat')
    nameSig = [nameSig, '.mat'];
end



% Get sbx.mat file info variable
if exist('varargin', 'var') && nargin >= 2 && ~isempty(varargin{2})
    if iscell(varargin{2})
        pnInfo = varargin{2}{1};
        fnInfo = varargin{2}{2};
    else
        pnInfo = '';
        fnInfo = varargin{2};
    end
    pnInfo = [pnInfo '/'];
else % Try to guess info file name
    pnInfo = pnSig;
    fnInfo = strsplit(fnSig, {'.mat','_SPSIG'});
    fnInfo = [fnInfo{1}, '.mat'];
    if ~exist([pnInfo fnInfo],'file') % Not found: Ask user for info file
        [fnInfo, pnInfo] = uigetfile({'*normcorr.mat', 'Registered stack'; '*.*', 'All Files'}, 'Get 2P Info');
    end
end

nameInfo = [pnInfo, fnInfo];
load(nameInfo, 'info');


% Get log file
if exist('varargin', 'var') && nargin >= 3 && ~isempty(varargin{3})
    if iscell(varargin{3})
        pnLog = varargin{3}{1};
        fnLog = varargin{3}{2};
        pnLog = [pnLog '/'];
        nameLog = [pnLog, fnLog];
    else
        nameLog = varargin{3};
    end
else
    % locate and load stimulus log
    [fnLog, pnLog] = uigetfile([pnSig '*.mat'], 'Get Stimulus log. (or press cancel)');
    nameLog = [pnLog, fnLog];
end
% Put the log into the info
if ~isempty(nameLog) && ischar(nameLog)
    Stim = load(nameLog);
    info.Stim = Stim;
end


% Prepare pre & post stim time
if exist('varargin', 'var') && nargin >= 4 && ~isempty(varargin{4})
    timeWindowStim = varargin{4};
else
    timeWindowStim = [];
end

ci = regexp(fnSig, '\d_\D','once');
fnbase = fnSig(1:ci);

% Chris eye processing
chrisEyeProcess = false;
if exist('varargin', 'var') && nargin >= 5 && ~isempty(varargin{5})
    chrisEyeProcess = varargin{5};
    askquestions = false;
else
    fneye = [pnInfo fnbase '_eye.mat'];
    if exist(fneye, 'file')
        chrisEyeProcess = questdlg('load eye motion data? (Chris function)');
        if strcmpi(chrisEyeProcess, 'Yes')
            chrisEyeProcess = true;
        else
            chrisEyeProcess = false;
        end
    end
    askquestions = true; % did not get so much input. Allowed to ask questions
end


if exist('varargin', 'var') && nargin >= 6 && ~isempty(varargin{6})
    stimDelayms = varargin{6};
else
    stimDelayms = 0;
end


if contains(nameSig, 'Concat')
    load(nameSig, 'PP', 'Mask', 'BImg', 'copyinfo', 'redROIs', 'ABAroi')
    %get time, roi info, and running and eyedata
    sbxgettimeinfo(pnInfo, fnInfo, fnbase, timeWindowStim, chrisEyeProcess, askquestions, copyinfo);
else
    load(nameSig, 'PP', 'Mask', 'BImg', 'redROIs', 'ABAroi')
    sbxgettimeinfo(pnInfo, fnInfo, fnbase, timeWindowStim, chrisEyeProcess, askquestions);
end


nRois = PP.Cnt;
Tframe = info.Tframe;
Tline = info.Tline;
Tfirstline = (info.crop.y(1)-1)*Tline; %first line from top after cropping in ms
info.rois = [];  

%get median positions for each roi and define delay for each roi
%copy roi data
for i = 1:nRois
    my = double(median(PP.Con(i).y));
    info.rois(i).msdelay = (Tline * my + Tfirstline)*1000;
    info.rois(i).x = PP.Con(i).x;
    info.rois(i).y = PP.Con(i).y;
    info.rois(i).px = PP.P(1,i);
    info.rois(i).py = PP.P(2,i);
    % get Allan Brain Atlas stuff into Res file
    info.rois(i).ABAx = ABAroi.X(i);
    info.rois(i).ABAy = ABAroi.Y(i);
    info.rois(i).ABAregion = ABAroi.region(i);
    info.rois(i).ABAcontour = ABAroi.Con(i);
    info.rois(i).ABAborder = ABAroi.borderDist(i); % distance to nearest border
    if exist('redROIs', 'var')
        info.rois(i).red = ismember(i, redROIs);
    end
    if isfield(PP, 'A')
        info.rois(i).Area = PP.A(i);
    end
end

% info.rois = struct2table(info.rois);

info.Mask = Mask;
info.BImg = BImg;
info.stimDelayms = [];

%retrieve the Calcium signals for all rois
if strfind(fnSig, '_CASE') % CASE is for file created with CAIMAN
    load(nameSig, 'S', 'C_df', 'ROIvars')
    decon = real(S');          %deconvolved signal
    den = real(full(C_df)');   %denoised ar fitted Df/F signal  
    fnSig = fieldnames(ROIvars);
    %collecting all roi data in one struct array
    for i = 1:length(fnSig)
        tmp = num2cell(ROIvars.(fnSig{i}));
        tmp = cell2struct(tmp', 'T');
        [info.rois.(fnSig{i})] = tmp.T;
    end
else % ROIs via SpecSeg
    load(nameSig, 'den', 'decon', 'sig', 'sigraw',...
        'sigCorrected', 'sigrawCorrected', 'sigCorrected_Z',...
        'denCorrected', 'deconCorrected',...
        'seal', 'sealBack', 'sealCorrected',...
        'spike_prob', 'spike_times',...
        'facemapTraces')
end


% Check facemap data
if ~exist('facemapTraces', 'var') % was it in the SPSIG already? 
    % Check if facemap data file is present
    fnfacemap = [pnInfo fnbase '_proc.mat'];
    if exist(fnfacemap, 'file')
        answer = questdlg('Facemap file detected. Data not yet in SPSIG. Put into SPSIG and extract?',...
                'Facemap?', 'yes', 'no','yes');
        if strcmp(answer, 'yes')
            EyeFacemap_RetrieveTraces(fnfacemap, nameSig)
            load(nameSig, 'facemapTraces')
        end
    end
else
    % Facemap data was in SPSIG and is loaded already
end


%check if length of frametimes equals length of sig
Frametimes  = info.Frametimes;
if length(Frametimes) ~= size(sig,1)
    warning('Number of frametimes does not equal length of signal!, Frametimes updated')
    if length(Frametimes) > size(sig,1)
        Frametimes = Frametimes(1:size(sig,1));
    else
        Frametimes = (0:size(sig,1)-1)'.*info.Tframe*info.Slices + Frametimes(1);
    end
    info.Frametimes = Frametimes;
end
%intermediate save
info.filename = [nameSig(1:end-4) '_Res.mat'];
save(info.filename, 'info')

%now we can align on stimulus onsets
if isfield(info, 'frame')
    framesBeforStim = info.framesbeforstim;
    framesafterstim = info.framesafterstim;
    info.numofframes = framesBeforStim + framesafterstim;
    StimTimes = info.StimTimes;
    
    nStim = length(StimTimes);
    
    if nargin==7
        trialId = varargin{7};
    else
        trialId = ones(nStim, 1);
    end
    
    nStimFrames = info.numofframes;
    rois = info.rois;
    Res = [];
    if exist('sig','var');            CaSig = zeros(nStimFrames, nStim, nRois); end
    if exist('sigraw','var');         CaSigRaw = zeros(nStimFrames, nStim, nRois); end
    if exist('sigBack','var');        CaSigBack = zeros(nStimFrames, nStim, nRois); end
    if exist('sigCorrected','var');   CaSigCorrected = zeros(nStimFrames, nStim, nRois); end
    if exist('sigrawCorrected','var');CaSigRawCorrected = zeros(nStimFrames, nStim, nRois); end
    if exist('sigCorrected_Z','var'); CaSigCorrected_Z = zeros(nStimFrames, nStim, nRois); end
    if exist('den','var');            CaDen = zeros(nStimFrames, nStim, nRois); end
    if exist('denCorrected','var');   CaDenCorrected = zeros(nStimFrames, nStim, nRois); end
    if exist('decon','var');          CaDec = zeros(nStimFrames, nStim, nRois); end
    if exist('deconCorrected','var'); CaDeconCorrected = zeros(nStimFrames, nStim, nRois); end
    if exist('seal','var');           CaSeal = zeros(nStimFrames, nStim, nRois); end
    if exist('sealBack','var');       CaSealBack = zeros(nStimFrames, nStim, nRois); end
    if exist('sealCorrected','var');  CaSealCorrected = zeros(nStimFrames, nStim, nRois); end
    if exist('spike_prob', 'var');    CaSpike_prob = zeros(nStimFrames, nStim, nRois); end
    if exist('spike_times', 'var');   CaSpike_times = zeros(nStimFrames, nStim, nRois); end

    if isfield(info, 'eye')          
        eyepos = zeros(nStimFrames, nStim, 2); 
        eyesz = zeros(nStimFrames, nStim);
    end
    if isfield(info, 'Run'), speed = zeros(nStimFrames, nStim); end
    
    
    hWb = waitbar(0, ['Get trials for ' num2str(nRois) ' rois and ' num2str(nStim) ' stimuli.']);
    for Ri = 1:nRois  %each ROI has a different delay
        for i = 1:nStim    %each stim has a unique set of frames
            padbefore = [];
            padafter = [];
            Stimonset = StimTimes(i);
            delaysi = stimDelayms(trialId(i))/1000;
            info.stimDelayms(i) = delaysi;
            Spix = find((Frametimes + rois(Ri).msdelay/1000) < (Stimonset+delaysi), framesBeforStim, 'last');
            Stmix = find((Frametimes + rois(Ri).msdelay/1000) > (Stimonset+delaysi), framesafterstim, 'first');
            if length(Spix) < framesBeforStim %too short
                lng = length(Spix);
                padbefore = nan(framesBeforStim-lng, 1);
                disp(['not enough prestim frames in trial:' num2str(i)])
            end
            if length(Stmix) < framesafterstim %too short
                lng = length(Stmix);
                padafter = nan(framesafterstim-lng, 1);
                disp(['not enough poststim frames in trial:' num2str(i) ])
            end
            smpl = [Spix(:); Stmix(:)];
            
            if exist('sig','var')
                CaSig(:,i,Ri) = [padbefore; sig(smpl,Ri); padafter];
            end
            if exist('sigraw','var')
                CaSigRaw(:,i,Ri) = [padbefore; sigraw(smpl,Ri); padafter];
            end
            if exist('sigBack','var')
                CaSigBack(:,i,Ri) = [padbefore; sigBack(smpl,Ri); padafter];
            end
            if exist('sigCorrected','var')
                CaSigCorrected(:,i,Ri) = [padbefore; sigCorrected(smpl,Ri); padafter];
            end
            if exist('sigrawCorrected','var')
                CaSigRawCorrected(:,i,Ri) = [padbefore; sigrawCorrected(smpl,Ri); padafter];
            end
            if exist('sigCorrected_Z','var')
                CaSigCorrected_Z(:,i,Ri) = [padbefore; sigCorrected_Z(smpl,Ri); padafter];
            end
            % if exist('den','var')
            %     CaDen(:,i,Ri) = [padbefore; den(smpl,Ri); padafter];
            % end
            % if exist('denCorrected','var')
            %     CaDenCorrected(:,i,Ri) = [padbefore; denCorrected(smpl,Ri); padafter];
            % end
            % if exist('decon','var') 
            %     CaDec(:,i,Ri) = [padbefore; decon(smpl,Ri); padafter];
            % end
            if exist('deconCorrected','var') && all(size(deconCorrected) == size(sig))
                CaDeconCorrected(:,i,Ri) = [padbefore; deconCorrected(smpl,Ri); padafter];
            end
            % if exist('seal','var')
            %     CaSeal(:,i,Ri) = [padbefore; seal(smpl,Ri); padafter];
            % end
            % if exist('sealBack','var')
            %     CaSealBack(:,i,Ri) = [padbefore; sealBack(smpl,Ri); padafter];
            % end
            % if exist('sealCorrected','var')
            %     CaSealCorrected(:,i,Ri) = [padbefore; sealCorrected(smpl,Ri); padafter];
            % end
            if exist('spike_prob', 'var')
                CaSpike_prob(:,i,Ri) = [padbefore; spike_prob(smpl,Ri); padafter];
            end
            if exist('CaSpike_times', 'var')
                CaSpike_times(:,i,Ri) = [padbefore; spike_times(smpl,Ri); padafter];
            end
            if exist('eyepos','var')
                eyepos(:,i,:) = [[padbefore, padbefore]; info.eye.pos(smpl, :); [padafter, padafter]];
            end
            if exist('eyesz','var')
                eyesz(:,i) = [padbefore; info.eye.area(smpl); padafter];
            end
            if exist('speed','var')
                speed(:,i) = [padbefore; info.Run.Speed(smpl); padafter];
            end
        end
        waitbar(Ri/nRois, hWb);
    end
    close(hWb)
    
    Slices = info.Slices;
    ax = (1:nStimFrames)*(Tframe*Slices) - framesBeforStim*(Tframe*Slices);
    win = ax > 0 & ax < 1;
    spnt = ax < 0;
    %get diversity ratio for each roi/response
    Snr = zeros(nRois,1);
    if exist('CaDec', 'var')
        for j = 1:nRois
            Rtmp = smoothG(CaDec(:,:,j),1);
            %Rtmp = CaDec(:,:,j);
            %Mn = poissfit(mean(Rtmp(spnt,:))); %poisonfit over trials
            Sd = std(mean(Rtmp(spnt,:)), 'omitnan');

            Rm = mean(Rtmp, 2, 'omitnan');
            Mx = max(Rm(win), [], 'omitnan');
            SMn = mean(Rm(spnt));

            %Snr(j) = (Mx-Mn)/Mn;
            Snr(j) = (Mx-SMn)/Sd; 
        end
        Snr(isnan(Snr) | isinf(Snr)) = 0;
    end
    
    if exist('sig', 'var');           Res.CaSig = CaSig; end
    if exist('sigraw','var');         Res.CaSigRaw = CaSigRaw; end
    if exist('sigBack','var');        Res.CaSigBack = CaSigBack; end
    if exist('sigCorrected','var');   Res.CaSigCorrected = CaSigCorrected; end
    if exist('sigrawCorrected','var');Res.CaSigRawCorrected = CaSigRawCorrected; end
    if exist('sigCorrected_Z','var'); Res.CaSigCorrected_Z = CaSigCorrected_Z; end
    if exist('den','var');            Res.CaDen = CaDen; end
    if exist('denCorrected','var');   Res.CaDenCorrected = CaDenCorrected; end
    if exist('decon','var');          Res.CaDec = CaDec; end
    if exist('deconCorrected','var'); Res.CaDeconCorrected = CaDeconCorrected; end
    if exist('seal','var');           Res.CaSeal = CaSeal; end
    if exist('sealBack','var');       Res.CaSealBack = CaSealBack; end
    if exist('sealCorrected','var');  Res.CaSealCorrected = CaSealCorrected; end
    if exist('spike_times', 'var');   Res.CaSpike_times = CaSpike_times; end
    if exist('spike_prob', 'var');    Res.CaSpike_prob = CaSpike_prob; end
    if exist('eyepos','var'),         Res.eye.pos = eyepos; end
    if exist('eyesz','var'),          Res.eye.sz = eyesz; end
    if exist('speed','var'),          Res.speed = speed; end
    if exist('facemapTraces', 'var')
        stimWindow = [framesBeforStim framesafterstim];
        Res.facemap = EyeFacemap_Extract(facemapTraces, stimWindow, Frametimes, StimTimes, askquestions); 
    end
    Res.Snr = Snr;
    Res.ax =  ax;
    
    save(info.filename, 'Res', 'info', '-v7.3')
    
%     % plot
%     Data.Res = Res;
%     Data.info = info;
%    plot_Res(Data)
    fprintf('Done with Res file for %s\n\n', fnbase)
else
	disp('Trial aligned data traces will not be generated!!')
end