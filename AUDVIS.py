#@matushalak
# main class that holds and organizes data for the Audio-visual task of Dark-rearing experiment at Levelt Lab
# requires python > 3.8, preferably newer for := to function
# aggregates all neurons across recording sessions of same group
from utils import group_condition_key, default_neuron_index, default_session_index, load_audvis_files, progress_bar
from collections import defaultdict
from SPSIG import SPSIG, Dict_to_Class
import numpy as np
from numpy import ndarray
from pandas import DataFrame, concat
import pickle
from multiprocessing import Pool, cpu_count
import argparse
import matplotlib.pyplot as plt
import os
from typing import Literal

class Behavior:
    def __init__(self,
                 speed:ndarray, 
                 continuous:SPSIG | None = None, # needed to Z-score everything (not if we do the trial-evoked Z-scoring / baseline correction)
                 facemap:Dict_to_Class|None = None,
                 baseline_frames:int = 16):
        # NOTE: baseline frames (16) hardcoded now!
        run = speed.swapaxes(0,1)[:720,:]
        nans_run = np.any(np.isnan(run), axis = 1)
        run_correct = run[~nans_run]
        self.running =  np.zeros_like(run)
        self.running[:len(run_correct), :] = run_correct - np.nanmean(run_correct[:,:baseline_frames], axis = 1, keepdims=True) #baseline corrected
        self.running[len(run_correct):, :] = run[nans_run]
        if facemap: # some sessions don't have video & facemap
            cont_fm : Dict_to_Class = continuous.facemapTraces
            # take care of NANs
            motion = facemap.motion.swapaxes(0,1)[:720,:]
            nans_motion = np.any(np.isnan(motion), axis = 1)
            motion_correct = motion[~ nans_motion]

            # Z-scored and baseline-corrected whisker movement on trial level, to control for trial-evoked whisker movements
            self.whisker = np.zeros_like(motion)
            self.whisker[:len(motion_correct),:] = (motion_correct - np.nanmean(motion_correct[:,:baseline_frames], axis = 1, keepdims=True)
                                                    ) / np.nanstd(motion_correct[:,:baseline_frames],axis = 1, keepdims=True)
            self.whisker[len(motion_correct):,:] = motion[nans_motion]
            self.blink = (facemap.blink.swapaxes(0,1)[:720,:] - np.nanmean(cont_fm.blink)) / np.nanstd(cont_fm.blink)
            self.pupil = (facemap.eyeArea.swapaxes(0,1)[:720,:] - np.nanmean(cont_fm.eyeArea)) / np.nanstd(cont_fm.eyeArea)


class AUDVIS:
    def __init__(self,
                 neuron_indexing:dict[int:dict[str]],
                 session_indexing:dict[int:dict[str]],
                 ROIs:DataFrame,
                 SIG:ndarray,
                 Z:ndarray,
                 TRIALS_ALL:ndarray,
                 NAME:str,
                 CASCADE : ndarray | None,
                 pre_post_trial_time:tuple[float, float] = (1, 2)):
        # Group-condition name of AUDVIS object
        self.NAME = NAME 
        
        # need to be careful when indexing into this
        # indexes in ABAregion are MATLAB (1-7) and 0 means the neuron is outside the regions of interest!
        self.ABA_regions = ['VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISl', 'VISp']
        
        # Imports
        self.neurons = neuron_indexing # information about each neuron
        self.sessions = session_indexing # information about each session
        # ranges of neurons corresponding to one recording session
        self.session_neurons = self.update_session_index()
        
        # information about all ROI locations
        self.rois = ROIs 
        # neuropil-corrected, trial-locked ∆F/F signal
        self.signal = SIG 
        # z-scored neuropil-corrected, trial-locked ∆F/F signal
        self.zsig = Z 

        # regress out running speed & OR whisker movement
        print(NAME)
        self.zsig_CORR = self.regress_out_behavior(self.zsig, signalname='zsig')
        # self.signal_CORR = self.regress_out_behavior(self.signal, signalname = 'sig')

        # load spike probability estimated using CASCADE algorithm and regress out trial-evoked whisker and running
        if CASCADE is not None:
            self.CASCADE = CASCADE
            self.CASCADE_CORR = self.regress_out_behavior(self.CASCADE, signalname = 'CASCADE')

        # per session, identify of all the presented trials
        self.trials = TRIALS_ALL 
        # trial window in seconds
        self.trial_sec = pre_post_trial_time
        # Sampling Frequency
        self.SF = self.signal.shape[1] / sum(self.trial_sec) 
        # Trial window (pre_trial, post_trial) in frames
        self.trial_frames = (self.SF * np.array(self.trial_sec)).round().astype(int)
        # TRIAL duration between frames hardcoded TODO: fixed based on indicated trial duration
        self.TRIAL = (self.trial_frames[0], 2*self.trial_frames[0]) 

        # Trials
        self.str_to_int_trials_map, self.int_to_str_trials_map = {s:i for i, s in enumerate(np.unique(self.trials))}, {i:s for i, s in enumerate(np.unique(self.trials))}
        # trialIDs as integers 
        self.trials = self.trials_apply_map(self.trials, self.str_to_int_trials_map)
        # create masks to separate different trials
        self.trial_types =  self.get_trial_types_dict(all_trials=self.trials)# this gives the [session (1:9)], [trial (1:720)] indices for each trial type
                            

    # Methods: to use on instance of the class in a script
    def baseline_correct_signal(self, signal:ndarray, baseline_frames:int = None)->ndarray:
        '''assumes (trial, time, neuron, ...) dimensionality'''
        if baseline_frames is None:
            baseline_frames = self.trial_frames[0]
        return signal - signal[:,:baseline_frames,:].mean(axis = 1, keepdims=True)

    def separate_signal_by_trial_types(self,  
                                       signal:ndarray,
                                       **kwargs) -> dict[int:ndarray]:
        '''Only works on trial-locked signal,
        signal is assumed to be in (trials_all_conditions, time, neurons_across sessions) format
        
        returns a dictionary that stores separate arrays for each trial type (trials_one_condition, time, neurons_across_sessions)
        '''
        all_trials : ndarray = self.trials if 'all_trials' not in kwargs else kwargs['all_trials']
        assert isinstance(all_trials, ndarray), 'all_trials must be (session, trials) array'        
        trials_per_TT = round(signal.shape[0] / len(np.unique(all_trials)))
        
        signal_by_trials = dict()

        # split signal from different recordings before joining
        ttypes : dict[int:ndarray] = self.trial_types if 'ttypes' not in kwargs else kwargs['ttypes']
        assert isinstance(ttypes, dict), 'ttypes must be a dictionary with indices for all occurences of each trial type across sessions'

        for tt in list(ttypes): # 0-n_trial types
            signal_tt = []
            for (n_first, n_last) in self.session_neurons:
                _, trials = ttypes[tt]
                trials = trials[tt*trials_per_TT : (tt+1)*trials_per_TT]
                signal_tt.append(signal[trials,:,n_first:n_last])
            
            signal_by_trials[tt] = np.concatenate(signal_tt,
                                                  axis = 2)
            
        return signal_by_trials
    
    def regress_out_behavior(self, signal:ndarray, 
                             MODE : Literal['all', 'whisker', 'running'] = 'all',
                             signalname : str = 'zsig', 
                             PLOT:bool = False
                             )->ndarray:
        ''''
        Regresses whisker movement and running speed out of all neural signals
        '''
        if os.path.exists(res := os.path.join('pydata', f'{self.NAME}_{signalname}_CORRECTED.npy')):
            # once created, loaded immediately
            result = np.load(res)
        
        else:
            sig_by_session, run_by_session, whisk_by_session = [], [], []
            for i_session, (start, end) in enumerate(self.session_neurons):
                sig_by_session.append(signal[:,:,start:end])
                behavior : Behavior = self.sessions[i_session]['behavior']
                
                if behavior is None:
                    print(self.sessions[i_session]['session'], ' has no behavioral information to regress out')
                    run_by_session.append(None)
                    whisk_by_session.append(None)
                    continue
                
                # if there is behavior, there is running
                # impute trials missing running data
                if np.any(np.isnan(behavior.running)): 
                    nan_trials_rs = np.any(np.isnan(behavior.running), axis = 1)
                    print(f'Imputing running speed for {sum(nan_trials_rs)} trials in session {i_session} : {self.sessions[i_session]['session']}')
                    # Compute session means excluding NaNs
                    mean_running = np.nanmean(behavior.running[~nan_trials_rs])
                    behavior.running[nan_trials_rs] = mean_running
                
                run_by_session.append(behavior.running)
                
                if hasattr(behavior, 'whisker'):
                    # impute trials missing whisker data
                    if np.any(np.isnan(behavior.whisker)): 
                        nan_trials_wm = np.any(np.isnan(behavior.whisker), axis = 1)
                        print(f'Imputing whisker movement energy for {sum(nan_trials_wm)} trials in session {i_session} : {self.sessions[i_session]['session']}')
                        # Compute session means excluding NaNs
                        mean_whisker = np.nanmean(behavior.whisker[~nan_trials_wm])
                        behavior.whisker[nan_trials_wm] = mean_whisker
                    
                    whisk_by_session.append(behavior.whisker)
                else:
                    whisk_by_session.append(None)
            
            if PLOT:
                if os.path.exists(plot_dir := os.path.join('/Volumes/my_SSD/NiNdata', 'single_neuron_plots', self.NAME)):
                    pass
                else:
                    os.makedirs(plot_dir)
            
            # regression parameters 
            reg_params = prepare_regression_args(sig_by_session, run_by_session, whisk_by_session, plot_dir=None if not PLOT else plot_dir)

            print('Regressing out running speed and whisker movement energy.')

            # return residual_signals
            with Pool(processes=cpu_count()) as pool:
                residual_signals = pool.starmap(regress_out_neuron, reg_params)
            print('Done!')
            
            result = np.dstack(residual_signals)
            np.save(os.path.join('pydata', f'{self.NAME}_{signalname}_CORRECTED.npy'), result)

        return result

    @staticmethod
    def trials_apply_map(trials_array:ndarray[str|int],
                         tmap:dict[str|int:int|str]) -> ndarray[int|str]:
        dtype = int if isinstance(list(tmap)[0], str) else str # depending on which map used want to map from str->int or from int->str
        
        if len(trials_array.shape) == 1:
            return np.fromiter(map(lambda s: tmap[s], trials_array), dtype = dtype)
        else:
            return np.array([np.fromiter(map(lambda s: tmap[s], trials_array[session, :]), dtype = dtype) 
                            for session in range(trials_array.shape[0])])
        
    def get_trial_types_dict(self, all_trials:ndarray[int]):
         return {tt : np.where(all_trials == tt) # this gives the [session (1:9)], [trial (1:720)] indices for each trial type
                for tt in np.unique(all_trials)}

    # used within __init__ block
    def update_session_index(self)-> list[tuple[int, int]]:
        # Update self.sessions
        iii = 0
        session_neurons = []
        
        for i in list(self.sessions): 
            session_neurons.append(neurons := (iii, iii + self.sessions[i]['n_neurons']))
            iii += self.sessions[i]['n_neurons']
            self.sessions[i]['neurons'] = neurons
        
        return session_neurons



class CreateAUDVIS:
    def __init__(self,
                 files:list[str],
                 name:str,
                 storage_folder:str = 'pydata'):
        self.NAME = name
        # SPSIG or SPSIG_Res files to be combined across sessions in same group and condition
        self.files = sorted(files)
        # Neuron index has the follwing structure:
        # - overall neuronID int (0:n_neurons in given condition across animals)
        #   - 'overall_id'  : int (0:n_neurons in given condition across animals)
        #   - 'specific_id' : tuple(str, int) ({RecordingName}, {neuronID}) (1:n_neurons in given recording in given animal)
        #   - 'region' : int | str (based on AllenAllign.m script gives the area in Visual system in which this neuron resides)
        #   - 'trialIDs' : ndarray (indicates which trials correspond to what condition for this particular neuron)
        #   - 'n_trials': int (indicates number of trials that this particular neuron was involved in)
        #   - 'facemap_corrected' : bool (indicates whether the signals of this neuron have been corrected with facemap information)
        self.neuron_index = defaultdict(default_neuron_index)
        # TODO: explanation for  session index
        self.session_index = defaultdict(default_session_index) 
        
        # main step in CreateAUDVIS class
        self.ROIs, self.SIG, self.Z, self.CASCADE, self.TRIALS_ALL = self.update_index_and_data()

        # save files that we will use when loading AUDVIS
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)
        
        self.ROIs.to_pickle(os.path.join(storage_folder, f'{self.NAME}_rois.pkl'))
        np.save(os.path.join(storage_folder, f'{self.NAME}_sig.npy'), 
             self.SIG)
        np.save(os.path.join(storage_folder, f'{self.NAME}_zsig.npy'), 
             self.Z)
        np.save(os.path.join(storage_folder, f'{self.NAME}_CASCADE.npy'), 
             self.CASCADE)
        np.save(os.path.join(storage_folder, f'{self.NAME}_trials.npy'), 
             self.TRIALS_ALL)
        
        with open(os.path.join(storage_folder, f'{self.NAME}_indexing.pkl'), 'wb') as indx_f:
            pickle.dump({'neuron_index':self.neuron_index,
                         'session_index':self.session_index}, indx_f)

    def update_index_and_data(self):
        roi_info = [] # collect dataframes here
        signal_corrected = [] # collect neuropil corrected ∆F/F signal for all sessions here
        signal_z = [] # collect z-score (over entire session) of neuropil corrected ∆F/F signal for all sessions here
        CASCADE_probs = [] # collect CASCADE spike probability
        trials_ALL = [] # collect trial IDs for each session

        for session_i, file in enumerate(self.files):
            sp = SPSIG(file) # SPSIG_Res.mat file
            continuous = SPSIG(file.replace('_Res', '')) # SPSIG.mat file

            # this is consistent across all neurons in one recording
            recording_name = sp.info.strfp.split('\\')[-1]
            n_neurons = sp.Res.CaSig.shape[-1]
            trials_ALL.append(trialIDs := sp.info.Stim.log.stim[:720]) # hardcoded because one session had more somehow
            n_trials = len(trialIDs)
            behavior = Behavior(sp.Res.speed, continuous ,sp.Res.facemap) if hasattr(sp.Res, 'facemap') else (Behavior(sp.Res.speed) if hasattr(sp.Res, 'speed') else None)
            
            # session index
            self.session_index[session_i]['n_neurons'] = n_neurons
            self.session_index[session_i]['trialIDs'] = trialIDs 
            self.session_index[session_i]['n_trials'] = n_trials
            self.session_index[session_i]['behavior'] = behavior
            self.session_index[session_i]['session'] = recording_name
            self.session_index[session_i]['event_times'] = sp.info.StimTimes
            self.session_index[session_i]['frame_times'] = sp.info.Frametimes
            # ms_delay info is in ROIs DataFrame

            # this is per-neuron
            roi_info.append(roiDF := DataFrame(sp.info.rois))
            signal_corrected.append(sig := sp.Res.CaSigCorrected[:,:720,:])
            
            if hasattr(sp.Res, 'CaSigCorrected_Z'):        
                signal_z.append(sp.Res.CaSigCorrected_Z[:, :720, :])
            else:
                print(f'Calculating z-score for {recording_name} using sigCorrected from {file.replace('_Res', '')}')
                signal_z.append((sig - continuous.sigCorrected.mean(axis = 0))/ continuous.sigCorrected.std(axis = 0)) # Z-score calculation (timepoints, rows organization)

            if hasattr(sp.Res, 'CaSpike_prob'):
                CASCADE_probs.append(sp.Res.CaSpike_prob[:,:720,:])

            for neuron_i_session in range(n_neurons):
                neuron_i_overall = neuron_i_session if session_i == 0 else max(list(self.neuron_index)) + 1
                self.neuron_index[neuron_i_overall]['overall_id'] = neuron_i_overall
                self.neuron_index[neuron_i_overall]['specific_id'] = (recording_name, neuron_i_session)
                self.neuron_index[neuron_i_overall]['region'] = roiDF.ABAregion[neuron_i_session]
                self.neuron_index[neuron_i_overall]['trialIDs'] = trialIDs
                self.neuron_index[neuron_i_overall]['n_trials'] = n_trials
            
            print(f'Session {recording_name}, with {n_neurons} ROIs DONE!')
        
        # create the IMPORTANT DFs & Arrays
        return (concat(roi_info), # DataFrane with ABA info & contours & locations
                np.dstack(signal_corrected).swapaxes(0,1), # ndarray Neuropil corrected ∆F/F
                np.dstack(signal_z).swapaxes(0,1), # ndarray Z-Scored (over entire signal) Neuropil corrected ∆F/F
                np.dstack(CASCADE_probs).swapaxes(0,1), # ndarray of Spike Probabilities for each neuron (obtained with CASCADE algorithm)
                np.array(trials_ALL) # ndarray with trial identities of each trial 
                ) 


        
#-------------------- Little / Parallelizable functions --------------------
# for multiprocessing
def run_CreateAUDVIS(args):
    g, g_name = args
    return CreateAUDVIS(g, g_name)

# regressing out
def regress_out_neuron (neuron_mat : np.ndarray ,X_movement_trials : np.ndarray,
                        plot_dir:str | None = None) -> ndarray:
    '''
    vectorized regressing out using pseudo-inverse for all trials at once in one neuron

    input:
    -------
    args list has 2:
        args[0] => neuron_mat = (ntrials, n_timepoints) 2D ndarray of neuronal signal across trials and timepoints
        args[1] => X_movement_trials = (ntrials, 2 | 3, n_timepoints) 3D ndarray | None design matrix with running, (whisker) and intercept columns for each trial
            if None means no behavior information to be regressed
    plot_dir enables saving single neuron figures into specified directory, None by default

    returns:
    ---------
    residual => (ntrials, ntimepoints) 2D ndarray = signal left after regressing out movement on TRIAL LEVEL
    '''
    if X_movement_trials is None:
        return neuron_mat # nothing to regress out

    assert neuron_mat.shape[0] == X_movement_trials.shape[0], f'Dimensions of neuron ({neuron_mat.shape[0]}) and first axis of design matrix ({X_movement_trials.shape[0]}) do not match, cannot proceed'
    
    # reshape to prepare for multidimensional array operations -> n trials, t timepoints, p predictors
    # this is necassary because batched pseudoinverse accepts stacked arrays in the 1st dimension
    X_movement_trials = X_movement_trials.transpose(0,2,1) 
    # n trials, t timepoints, p predictors
    n, t, p = X_movement_trials.shape
    
    # batched pseudoinverse on (n-trials, TIMEPOINTS x PREDICTRORS) -> n x p x t
    PIV_mat_all = np.linalg.pinv(X_movement_trials) 
    assert PIV_mat_all.shape == (n,p,t), f'Error with batched pseudo-inverse, shape {PIV_mat_all.shape} instead of {(n,p,t)}'
    # n (trials) p (predictors) t (timepoints); performs matrix multiplication of 3D tensor w 2D matrix
    beta_all = np.einsum('npt, nt -> np', PIV_mat_all, neuron_mat) # beta_all has shape (n,p)
    assert beta_all.shape == (n,p), f'Error with beta_all calculation, shape {beta_all.shape} instead of {(n,p)}'
    # ntp @ np -> nt
    predicted_all =  np.einsum('ntp, np -> nt', X_movement_trials, beta_all) # predicted_all has shape (n, t)
    assert predicted_all.shape == neuron_mat.shape, f'Error with predicted_all calculation, shape {predicted_all.shape} instead of {neuron_mat.shape}'
    residual : ndarray = neuron_mat - predicted_all

    # plotting (normally disabled)
    if plot_dir is not None:
        plot_single_neuron_regression(neuron_mat, X_movement_trials, predicted_all, residual,
                                      plot_dir)
    
    return residual.reshape(neuron_mat.shape) # residual signal

def prepare_regression_args(sbs:list[ndarray], rbs:list[ndarray|None], wbs:list[ndarray|None],
                            plot_dir:str|None = None) -> list[tuple[ndarray, ndarray | None]]:
    ''''
    sbs : signals by session, list[ (n_trials, n_timepoints, n_neurons),...] 
    rbs : running by session, list[ (n_trials, n_timepoints), ...] 
        if None, no behavior information was present for mouse
    wbs : whisker movement by session, list[ (n_trials, n_timepoints), ...] 
        if None, facemap video was missing but running wheel information usually still there
    '''
    sig_X_args = []
    add_neurons = 0
    n_trials, n_timepoints = sbs[0].shape[:2]
    for i_sess, args in enumerate(zip(sbs, rbs, wbs)):
        match args:
            case neuron_mat, None, None:
                X_movement_flat = None

            case neuron_mat, running, None:
                X_movement : ndarray = running  # shape: (n_trials, time_points)
                X_movement_flat = X_movement.reshape(n_trials, 1, n_timepoints)  # shape: (n_trials, 1, n_time_points)
                X_movement_flat = np.column_stack((np.ones(X_movement_flat.shape), X_movement_flat))  # add intercept, (n-trials, 2, n_timepoints)

            case neuron_mat, running, whisker:
                X_movement = np.dstack([running, whisker])  # shape: (n_trials, time_points, 2)
                X_movement_flat = X_movement.reshape(n_trials, 2, n_timepoints)  # shape: (n_trials, 2, n_timepoints)
                X_movement_flat = np.column_stack((np.ones(X_movement_flat.shape), X_movement_flat))  # add intercept, (n_trials, 3,n_timepoints )
        
        # (signal and design_matrix)
        if plot_dir is not None: # return filenames for plots
            sig_X_args.extend([(neuron_mat[:,:,neuron], X_movement_flat, os.path.join(plot_dir, f'neuron{neuron + add_neurons}')) for neuron in range(neuron_mat.shape[-1])])
        else:
            sig_X_args.extend([(neuron_mat[:,:,neuron], X_movement_flat) for neuron in range(neuron_mat.shape[-1])])

        add_neurons += neuron_mat.shape[-1]

    return sig_X_args # arguments for parallelized execution

def plot_single_neuron_regression(neuron_mat : np.ndarray, X_movement_flat : np.ndarray, 
                                  predicted : np.ndarray, residual : np.ndarray,
                                  plot_dir : str):
    X_mat = X_movement_flat.reshape((neuron_mat.shape[0], neuron_mat.shape[1], -1))[:,:,1:] # first dimension is just intercept
    run_mat = X_mat[:,:,0]
    whisk_mat = X_mat[:,:,1] if X_mat.shape[2] == 2 else None
    pred_mat = predicted.reshape(neuron_mat.shape)
    res_mat = residual.reshape(neuron_mat.shape) # residual signal
    
    # axes / rows: 
    #   timeseries_raw traces + average raw trace + final corrected average trace + average predicted trace
    #   running traces + average running trace
    #   whiker traces + average whisker trace
    rows = 3 if whisk_mat is not None else 2
    fig, ax = plt.subplots(nrows = rows, ncols = 1, figsize = (8, rows * 5), sharex=True)
    for trial in range(neuron_mat.shape[0]):
        # timeseries
        ax[0].plot(neuron_mat[trial,:], color = 'lavender', alpha = 0.025)#, label = 'z∆F/F0' if trial == neuron_mat.shape[0]-1 else '')
        # ax[0].plot(pred_mat[trial,:], color = 'orange', alpha = 0.025, label = 'predicted z∆F/F0'if trial == neuron_mat.shape[0]-1 else '')
        # running
        ax[1].plot(run_mat[trial,:], color = 'lightgreen', alpha = 0.08)#, label = 'running speed'if trial == neuron_mat.shape[0]-1 else '')
        # whisker energy
        if whisk_mat is not None:
            ax[2].plot(whisk_mat[trial,:], alpha = .08, color = 'lightpink')#), label = 'trial-evoked whisker-energy'if trial == neuron_mat.shape[0]-1 else '')
    
    # averages
    ax[0].plot(neuron_mat.mean(axis = 0), color = 'slateblue', label = 'average z∆F/F0')
    ax[0].plot(pred_mat.mean(axis = 0), color = 'darkorange', label = 'average predicted z∆F/F0')
    ax[0].plot(res_mat.mean(axis = 0), color = 'blue', label = 'average residual z∆F/F0', linewidth = 3)
    ax[0].set_ylabel('z(∆F/F), symlog')
    ax[0].legend(loc = 2)
    ax[0].set_yscale('symlog')
    ax[0].vlines([15,31], ymin = neuron_mat.min(), ymax=neuron_mat.max(), linestyle = '--', color = 'saddlebrown')

    ax[1].plot(run_mat.mean(axis = 0), color = 'limegreen', label = 'average running speed')
    ax[1].set_ylabel('a.u., symlog')
    ax[1].set_yscale('symlog')
    ax[1].legend(loc = 2)
    ax[1].vlines([15,31], ymin = run_mat.min(), ymax=run_mat.max(), linestyle = '--', color = 'saddlebrown')
    
    if whisk_mat is not None:
        ax[2].plot(whisk_mat.mean(axis = 0), color = 'crimson', label = 'average whisker energy')
        ax[2].legend(loc = 2)
        ax[2].set_yscale('symlog')       
        ax[2].set_ylabel('a.u., symlog')
        ax[2].vlines([15,31], ymin = whisk_mat.min(), ymax=whisk_mat.max(), linestyle = '--', color = 'saddlebrown')
    
    ax[-1].set_xticks([0, 15, 31, 46], ['-1', '0', '1', '2']) # NOTE: hardcoded
    ax[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(plot_dir, dpi = 200)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-create_files', type = str, default = 'no')
    return parser.parse_args()

def load_in_data()->tuple[AUDVIS, AUDVIS, AUDVIS, AUDVIS]:
    # Group - condition
    data = {1:{'pre':None,
            'post':None},
            2:{'pre':None,
            'post':None}}

    names = ['g1pre', 'g1post', 'g2pre', 'g2post']

    for group_name in names:
        params, Cascade = load_audvis_files(os.path.join('pydata', group_name))
        AVclass = AUDVIS(*params, NAME = group_name, CASCADE=Cascade)
        # by_trials = AVclass.separate_signal_by_trial_types(AVclass.signal)

        g = int(group_name[1])
        cond = group_name[2:]

        data[g][cond] = AVclass

    av1 = data[1]['pre']
    av2 = data[1]['post']
    av3 = data[2]['pre']
    av4 = data[2]['post']
    
    return av1, av2, av3, av4



# if ran as a script, initializes 4 AUDVIS classes with loaded data and saves them
if __name__ == '__main__':
    args = parse_args()
    if args.create_files.lower() in ('y', 'yes', 'true', 't'):
        print('Please select the root directory with your folders containing data to analyze')
        # will ask for directory from which to load
        # g1_files, g2_files = group_condition_key() # general
        g1_files, g2_files = group_condition_key(root = '/Volumes/my_SSD/NiNdata/data')

        # these will be the 4 instances of the AUDVIS class
        g1pre, g1post = g1_files['pre'], g1_files['post']
        g2pre, g2post = g2_files['pre'], g2_files['post']
        names = ['g1pre', 'g1post', 'g2pre', 'g2post']

        # 1 core (for debugging)
        # for g, g_name in zip((g1pre, g1post, g2pre, g2post), names):
        #     av = CreateAUDVIS(g, g_name)

        # multiprocessing on 4 cores (4 groups)
        params = list(zip([g1pre, g1post, g2pre, g2post], names))
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(run_CreateAUDVIS, params)

    else:
        print('Okay, not changing anything ;)')
        av1, av2, av3, av4 = load_in_data()