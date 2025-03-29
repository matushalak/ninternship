function addSpikeProbToSPSIG(currentFolder, spsigFileName)
% addSpikeProbToSPSIG Adds the spike_prob field from a full_prediction CASCADE file 
% to the SPSIG file and saves the updated file.
% by Matus Halak (@matushalak)
%   addSpikeProbToSPSIG(currentFolder, spsigFileName)
%
%   Inputs:
%       currentFolder   - Path to the folder containing the files.
%       spsigFileName   - The name of the SPSIG .mat file (e.g., 'SPSIG.mat').
%
%   Example:
%       addSpikeProbToSPSIG('C:\data', 'SPSIG.mat');

    % Find .mat file starting with "full_prediction" in the folder
    fullPredFiles = dir(fullfile(currentFolder, 'full_prediction*.mat'));
    if isempty(fullPredFiles)
        warning('No file starting with "full_prediction" found in folder: %s', currentFolder);
        return;
    end
    
    % Use the first matching file
    fullPredFilePath = fullfile(currentFolder, fullPredFiles(1).name);
    fullPredData = load(fullPredFilePath);
    
    % Check if the spike_prob field exists
    if ~isfield(fullPredData, 'spike_prob')
        error('The file %s does not contain the field "spike_prob".', fullPredFiles(1).name);
    end
    spikeProb = fullPredData.spike_prob.'; % transpose to match dimensions for trial locking
    
    % Load the SPSIG file
    spsigFilePath = fullfile(currentFolder, spsigFileName);
    spsigData = load(spsigFilePath);
    
    % Save spikeProb CASCADE output into SPSIG
    spsigData.spike_prob = spikeProb;
    
    % Save the updated structure back to the SPSIG file
    save(spsigFilePath, '-struct', 'spsigData');
    
    fprintf('Updated SPSIG file saved with CASCADE spike_prob field.\n');
end