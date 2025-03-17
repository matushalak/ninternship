function RunAllenTrialLock()
% Matus Halak
% runs Allen Align and Trial lock on all sessions within parent directory
% session must include _SPSIG.mat, _log.mat, _normcorr.mat and _CoordinatesXY.png files
% based on these, the AllenAlign script will run for all sessions and
% subsequently those sessions will be TrialLock-ed

%% Select main folder to search
mainFolder = uigetdir('Select main folder to search');
if mainFolder == 0
    disp('Folder selection cancelled.');
    return;
end

%% Find all *_SPSIG.mat files recursively in the selected folder
spsigFiles = dir(fullfile(mainFolder, '**', '*_SPSIG.mat'));

%% Loop through each found SPSIG file
for k = 1:length(spsigFiles)
    currentFolder = spsigFiles(k).folder;
    % Extract the last folder name from the path
    [~, folderName] = fileparts(currentFolder);
    
    % Check if the folder name starts with "Bar_Tone"
    if startsWith(folderName, 'Bar_Tone')
        % Look for the corresponding _CoordinatesXY.png file in the same folder
        coordFiles = dir(fullfile(currentFolder, '*_CoordinatesXY.png'));
        sbxFiles = dir(fullfile(currentFolder, '*_normcorr.mat'));
        logFiles = dir(fullfile(currentFolder, '*_log.mat'));
        if ~isempty(coordFiles)
            % Get the full paths (or separate directory and filename)
            spsigFileName = spsigFiles(k).name;
            
            if startsWith(spsigFileName, '._')
                continue
            end

            coordFileName = erase(coordFiles(1).name, '._');  % if multiple, taking the first one
            sbx_normcorrFileName = erase(sbxFiles(1).name, '._');
            log_FileName = erase(logFiles(1).name, '._');
            
            % Optionally, display info
            fprintf('Processing folder: %s\n', currentFolder);
            fprintf('SPSIG file: %s\n', spsigFileName);
            fprintf('Coordinates file: %s\n', coordFileName);
            
            % Call the functions with the folder and file names
            
            % Adjust the arguments as required by your function definitions
            fprintf('Starting AllenAllign for %s\n', currentFolder)
            AllenAllign({currentFolder, spsigFileName}, {currentFolder, coordFileName});
            fprintf('Starting TrialLock for %s\n', currentFolder)
            popups = false;
            TrialLock({currentFolder, spsigFileName}, {currentFolder, sbx_normcorrFileName}, {currentFolder, log_FileName}, [1 2], popups);    
        else
            fprintf('No _CoordinatesXY.png found in folder: %s\n', currentFolder);
        end
    end
end
end