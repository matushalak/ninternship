#@matushalak
# main class that holds and organizes data for the Audio-visual task of Dark-rearing experiment at Levelt Lab
# requires python > 3.8, preferably newer for := to function
# aggregates all neurons across recording sessions of same group
from utils import group_condition_key, default_neuron_index, default_session_index, load_audvis_files, trialMAPS
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
                 trialIDs: list,
                 facemap:Dict_to_Class|None = None,
                 baseline_frames:int = 15): #NOTE: was 16 before!!!
        '''
        Z-scored and baseline-corrected behaviors on trial level, to control for trial-evoked running / whisker movements / blinking / pupil
        '''
        # NOTE: baseline frames (15) hardcoded now!
        trialIDs = np.array(trialIDs)
        self.running = self.trial_evoked_non_nan(behavior=speed, trialIDs=trialIDs, baseline_frames=baseline_frames)
        if facemap: # some sessions don't have video & facemap
            self.whisker = self.trial_evoked_non_nan(behavior=facemap.motion, trialIDs=trialIDs, baseline_frames=baseline_frames, 
                                                    #  debug=True
                                                     )
            self.blink = self.trial_evoked_non_nan(behavior=facemap.blink, trialIDs=trialIDs, baseline_frames=baseline_frames)
            self.pupil = self.trial_evoked_non_nan(behavior=facemap.eyeArea, trialIDs=trialIDs, baseline_frames=baseline_frames)
    
    @staticmethod
    def trial_evoked_non_nan(behavior: np.ndarray, 
                             trialIDs: np.ndarray,
                             baseline_frames:int = 15,
                             debug:bool = False) -> np.ndarray:
        '''
        behavior: ndarray is a (trials, timepoints) signal
        trialIDs: ndarray is a (trials) array

        Returns trial-evoked behavior (Z-scored on baseline period for each trial) 
            with NaN trials filled in by the average behavior signal for that behavior in that trial type
        '''
        behavior = behavior.swapaxes(0,1)[:720,:]
        # boolean masks
        nans_behavior = np.any(np.isnan(behavior), axis = 1)
        correct_behavior = ~nans_behavior 
        # indices
        nans_i = np.where(nans_behavior)[0]
        correct_i = np.where(correct_behavior)[0]
        # indices for different trial types
        ttSTRtoINT = trialMAPS(trialIDs)[0]
        trialIDs = AUDVIS.trials_apply_map(trialIDs, ttSTRtoINT)
        tts = np.unique(trialIDs)
        tt_indices = [np.where(trialIDs == tt)[0] for tt in tts]
        # different trial types without or with nans
        correct_tt_indices = [np.intersect1d(correct_i, tti) for tti in tt_indices]
        nan_tt_indices = [np.intersect1d(nans_i, tti) for tti in tt_indices]
        # mean signal by trial type
        mean_behavior_tt_correct = [behavior[corr_tti].mean(axis = 0) for corr_tti in correct_tt_indices]
        # fill-in mean signal for each trial type to nan trials of that trial type
        for itt in range(tts.size):
            behavior[nan_tt_indices[itt]] = mean_behavior_tt_correct[itt]
        assert np.isnan(behavior).all() == False, 'All behavioral NaNs should have been removed in previous step!'
        
        # baseline correction and z-scoring on trial level
        bsl_means = np.mean(behavior[:,:baseline_frames], axis = 1, keepdims=True)
        bsl_stds = np.std(behavior[:,:baseline_frames],axis = 1, keepdims=True) + 0.01 # to prevent divide by 0 error
        behavior_trial_evoked = (behavior - bsl_means) / bsl_stds
        # if debug:
        #     plt.plot(np.nanmean(behavior, axis = 0))
        #     plt.plot(np.nanmean(behavior_trial_evoked, axis = 0))
        #     plt.show()
        #     plt.close()
        return behavior_trial_evoked


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
                 pre_post_trial_time:tuple[float, float] = (1, 2),
                 raw_plots:bool = False,
                 SF:float = 15.4570):
        # Group-condition name of AUDVIS object
        self.NAME = NAME 
        
        # need to be careful when indexing into this
        # indexes in ABAregion are MATLAB (1-7) and 0 means the neuron is outside the regions of interest!
        self.ABA_regions = ['VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISl', 'VISp']
        
        ## 0) Imports
        # neuropil-corrected, trial-locked ∆F/F signal
        self.signal = SIG 
        # z-scored neuropil-corrected, trial-locked ∆F/F signal
        self.zsig = Z 
        self.neurons = neuron_indexing # information about each neuron
        self.sessions = session_indexing # information about each session
        # ranges of neurons corresponding to one recording session
        self.session_neurons = self.update_session_index()
        # information about all ROI locations
        self.rois = ROIs 
        # per session, identify of all the presented trials
        self.trials = TRIALS_ALL 
        # trial window in seconds
        self.trial_sec = pre_post_trial_time

        ## 1) TRIAL INFORMATION
        # Sampling Frequency
        self.SF = SF#self.signal.shape[1] / sum(self.trial_sec) 
        # Trial window (pre_trial, post_trial) in frames
        self.trial_frames = (self.SF * np.array(self.trial_sec)).round().astype(int)
        # TRIAL duration between frames hardcoded TODO: fixed based on indicated trial duration
        self.TRIAL = self.trial_frames#(self.trial_frames[0], 2*self.trial_frames[0]) 

        # Trial MAPS from str->int and int->str
        self.str_to_int_trials_map, self.int_to_str_trials_map = trialMAPS(self.trials)
        # trialIDs as integers 
        self.trials = self.trials_apply_map(self.trials, self.str_to_int_trials_map)
        # create masks to separate different trials
        # this gives the [session (1:9)], [trial (1:720)] indices for each trial type
        self.trial_types =  self.get_trial_types_dict(all_trials=self.trials)

        ## 2) SIGNALS
        if raw_plots:
            self.raw_plots(self.signal)
        # regress out running speed & OR whisker movement
        # (regression needs to be done on all trials, even the ones we will nan later)
        print(NAME)
        self.zsig_CORR = self.regress_out_behavior(self.zsig, signalname='zsig')
        self.signal_CORR = self.regress_out_behavior(self.signal, signalname = 'sig')
        
        # NAN trials with too much confounding trial-locked behavior 
        # (must be AFTER regression, regression cannot deal with NaNs)
        # NOTE: if offset included, consider longer period for excluding trials (+ 250 ms, 4 frames)
        self.extTRIAL = (self.TRIAL[0], self.TRIAL[1]+5) # to capture offset
        self.signal_CORR = self.nantrials(signal=self.signal_CORR, Zthresh=2, 
                                          whiskWindow=self.extTRIAL)
        self.zsig_CORR = self.nantrials(signal=self.zsig_CORR, Zthresh=2, whiskWindow=self.extTRIAL)

        # load spike probability estimated using CASCADE algorithm and regress out trial-evoked whisker and running
        if CASCADE is not None:
            self.CASCADE = CASCADE
            self.CASCADE_CORR = self.regress_out_behavior(self.CASCADE, signalname = 'CASCADE')
            self.CASCADE_CORR = self.nantrials(signal=self.CASCADE_CORR, Zthresh=2, whiskWindow=self.extTRIAL) # NAN trial locked behavior

    # Methods: to use on instance of the class in a script
    # NOTE: subtract mean of baseline period in each trial
    def baseline_correct_signal(self, signal:ndarray, baseline_frames:int = None)->ndarray:
        '''assumes (trial, time, neuron, ...) dimensionality'''
        if baseline_frames is None:
            baseline_frames = self.trial_frames[0]
        
        baseline_subtract = np.nanmean(signal[:,:baseline_frames,:], axis = 1, keepdims=True)  
        assert baseline_subtract.shape == (signal.shape[0], 1, signal.shape[-1]), 'Baseline mean for each trial is subtracted'
        return signal - baseline_subtract

    def separate_signal_by_trial_types(self,  
                                       signal:ndarray,
                                       **kwargs) -> dict[int:ndarray]:
        '''
        Only works on trial-locked signal,
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

        for tt in sorted(list(ttypes)): # 0-n_trial types
            signal_tt = []
            for (n_first, n_last) in self.session_neurons:
                _, trials = ttypes[tt]
                trials = trials[tt*trials_per_TT : (tt+1)*trials_per_TT]
                signal_tt.append(signal[trials,:,n_first:n_last])
            
            signal_by_trials[tt] = np.concatenate(signal_tt,
                                                  axis = 2)
            
        return signal_by_trials
    
    def nantrials(self, signal: np.ndarray, 
                  whiskWindow: tuple[int, int],
                  Zthresh: float = 2,
                  verbose: bool = False, plot: bool = False) -> np.ndarray:
        '''
        Encodes trials with high whisker movement (> Zthresh) or closed eyes with NaN and returns the statistics by session
            signal: ndarray with (trials, timepoint, neurons) shape
            Zthresh: float specifying z-score threshold above which to encode trials as nans
        Returns:
            signalnan: ndarray with (trials, timepoint, neurons) shape, with problematic trials encoded as NaN 
                for all timepoints for all neurons within a session
        '''
        oks_by_sess = []
        nans_by_sess = []
        snames = []
        sigs_w_nans = []
        for i_session, (start, end) in enumerate(self.session_neurons):
            sess_sig = signal[:,:,start:end]
            behavior : Behavior = self.sessions[i_session]['behavior']
            sname = self.sessions[i_session]['session']
            sess_trialIDs = self.trials[i_session,:]
            # indices of trial type in the given session
            sess_itts = [np.where(sess_trialIDs==tt)[0] for tt in np.unique(sess_trialIDs)]
            
            if hasattr(behavior, 'whisker'):
                whisker = behavior.whisker
                # boolean mask
                whisker_problem = whisker[:,whiskWindow[0]:whiskWindow[1]].max(axis = 1) > Zthresh
                whisker_OK = ~whisker_problem
                # indices
                whiskProb_is = np.where(whisker_problem)[0]
                whiskOK_is = np.where(whisker_OK)[0]

                # ok trials left
                OK_itts = [np.intersect1d(whiskOK_is, itts_sess) for itts_sess in sess_itts]
                PROBLEM_itts = [np.intersect1d(whiskProb_is, itts_sess) for itts_sess in sess_itts]
                # number of trial left for each trial type
                nOK = [ok.size for ok in OK_itts]
                nPROB = [prob.size for prob in PROBLEM_itts]
                
                # flatten problem itts
                problem_itts_flat = np.concat(PROBLEM_itts)
                # breakpoint()
                sess_sig_w_nans = sess_sig
                # encode problem trials as nan
                sess_sig_w_nans[problem_itts_flat,:,:] = np.nan
                
                if verbose:
                    print(sname)
                    print(*[f'Trial type {self.int_to_str_trials_map[i]} : {nOK[i]} OK, {nPROB[i]} PROBLEM\n' for i in range(len(nOK))])
            else:
                sess_sig_w_nans = sess_sig # cannot drop nan trials
                # NOTE: dropping entire session here!!!
                if 'g2' in self.NAME:
                    # pass
                    sess_sig_w_nans[:,:,:] = np.nan
                    print(f'Encoding whole session {sname} as np.nan because no facemap')
                elif verbose:
                    print(sname)
                    print('Has no facemap info, so all trials classified as OK\n')
                nOK = [90 for _ in range(len(sess_itts))]
                nPROB = [90 for _ in range(len(sess_itts))]
            
            if not plot:
                sigs_w_nans.append(sess_sig_w_nans)
            else:
                oks_by_sess.append(nOK)
                nans_by_sess.append(nPROB)
                snames.append(sname)

        if not plot:
            sigNaNCorrected = np.dstack(sigs_w_nans)
            assert signal.shape == sigNaNCorrected.shape, f'The output needs to have the shape {signal.shape}, not {sigNaNCorrected.shape}!'
            return sigNaNCorrected
        else:
            oks = np.array(oks_by_sess)
            fig = plt.figure()
            for isess, ok in enumerate(oks):
                plt.plot(oks[isess,:], label = snames[isess], linestyle = '--' if snames[isess]!= 'Epsilon_20211210_002' else '-')
            plt.xticks(ticks = np.arange(oks.shape[1]), labels=[self.int_to_str_trials_map[i] for i in range(oks.shape[1])])
            plt.yticks(np.arange(0,91,5))
            plt.xlabel('Trial type')
            plt.ylabel('Number of trials left')
            plt.title(self.NAME)
            plt.legend(loc = 1, fontsize = 6)
            plt.tight_layout()
            plt.savefig(self.NAME+'nanEXPLORATION.png', dpi = 300)

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

            for rp in reg_params:
                residual_signals = regress_out_neuron(*rp)

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

    def raw_plots(self, signal:np.ndarray):
        sig_by_session, run_by_session, whisk_by_session, pup_by_session = [], [], [], []
        # Get signal and behavioral data organized by session
        for i_session, (start, end) in enumerate(self.session_neurons):
            sig_by_session.append(signal[:,:,start:end])
            behavior : Behavior = self.sessions[i_session]['behavior']
            run_by_session.append(behavior.running)
            if hasattr(behavior, 'whisker'):
                whisk_by_session.append(behavior.whisker)
                pup_by_session.append(behavior.pupil)
            else:
                whisk_by_session.append(None)
                pup_by_session.append(None)
        # hardcoded
        time = np.linspace(-1, 2, 47)
        # Go through sessions
        for isess, (sig, runn, whisk, pup) in enumerate(zip(sig_by_session,
                                                           run_by_session, 
                                                           whisk_by_session, 
                                                           pup_by_session)):
            if whisk is None:
                continue
            n_neur = sig.shape[-1]
            trial_select = np.random.choice(np.arange(sig.shape[0]), 1)
            for it in range(sig.shape[0]):
                if it != trial_select:
                    continue
                fg, axs = plt.subplots(nrows=6, ncols=1, sharex='col')
                neur_select = np.random.choice(np.arange(n_neur), 3)
                # neurons 1 trial
                axs[0].plot(time, sig[it, :, neur_select[0]], color = 'k')
                axs[1].plot(time, sig[it, :, neur_select[1]], color = 'k')
                axs[2].plot(time, sig[it, :, neur_select[2]], color = 'k')
                # running
                axs[3].plot(time, runn[it, :], color = 'k')
                # whisker
                axs[4].plot(time, whisk[it, :], color = 'k')
                # pupil
                axs[5].plot(time, pup[it, :], color = 'k')

                for ax in axs: 
                    ax.axvline(x=0, linestyle = 'dashed', color = 'k')
                    ax.axvline(x=1, linestyle = 'dashed', color = 'k')
                    ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(f'({self.NAME})-Session{isess}_trial{it}_neurons_{neur_select}.svg')
                print(f'Session {isess}, trial {it}, neurons {neur_select}')
                plt.show()
                plt.close()




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
            behavior = Behavior(speed=sp.Res.speed, facemap=sp.Res.facemap, trialIDs=trialIDs) if hasattr(sp.Res, 'facemap') else (Behavior(speed=sp.Res.speed, trialIDs=trialIDs) if hasattr(sp.Res, 'speed') else None)
            
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
                X_movement_flat = X_movement[:, np.newaxis,:]  # shape: (n_trials, 1, n_time_points)
                X_movement_flat = np.concatenate((np.ones(shape=(n_trials, 1, n_timepoints)), X_movement_flat), axis=1)  # add intercept, (n-trials, 2, n_timepoints)

            case neuron_mat, running, whisker:
                X_movement = np.dstack([running, whisker])  # shape: (n_trials, time_points, 2)
                X_movement_flat = X_movement.transpose(0, 2, 1)  # shape: (n_trials, 2, n_timepoints)
                X_movement_flat = np.concatenate((np.ones(shape=(n_trials, 1, n_timepoints)), X_movement_flat), axis=1)  # add intercept, (n_trials, 3,n_timepoints )
        
        # (signal and design_matrix)
        if plot_dir is not None: # return filenames for plots
            sig_X_args.extend([(neuron_mat[:,:,neuron], X_movement_flat, os.path.join(plot_dir, f'neuron{neuron + add_neurons}')) for neuron in range(neuron_mat.shape[-1])])
        else:
            sig_X_args.extend([(neuron_mat[:,:,neuron], X_movement_flat) for neuron in range(neuron_mat.shape[-1])])

        add_neurons += neuron_mat.shape[-1]

    return sig_X_args # arguments for parallelized execution

# z-scored behaviors and z-scored ∆F/F for Fig S1 
def plot_single_neuron_regression(neuron_mat : np.ndarray, X_movement_flat : np.ndarray, 
                                  predicted : np.ndarray, residual : np.ndarray,
                                  plot_dir : str):
    # neuron_n = plot_dir.split('/')[-1].removeprefix('neuron')

    X_mat = X_movement_flat.reshape((neuron_mat.shape[0], neuron_mat.shape[1], -1))[:,:,1:] # first dimension is just intercept
    run_mat = X_mat[:,:,0]
    whisk_mat = X_mat[:,:,1] if X_mat.shape[2] == 2 else None
    pred_mat = predicted.reshape(neuron_mat.shape)
    res_mat = residual.reshape(neuron_mat.shape) # residual signal
    
    # check single trial level
    for i in range(720):
        print(i)
        plt.plot(neuron_mat[i,:], label = 'signal')
        plt.plot(run_mat[i,:], label = 'running')
        plt.plot(whisk_mat[i,:], label = 'whisker')
        plt.plot(res_mat[i,:], label = 'residual signal')
        plt.legend(loc = 2)
        plt.show()
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-files', type = str, default = 'no')
    return parser.parse_args()

def load_in_data(pre_post: Literal['pre', 'post', 'both'] = 'both')->tuple[AUDVIS, AUDVIS, AUDVIS, AUDVIS]:
    # Group - condition
    data = {1:{'pre':None,
            'post':None},
            2:{'pre':None,
            'post':None}}

    match pre_post:
        case 'both':
            names = ['g1pre', 'g1post', 'g2pre', 'g2post']
        case 'pre':
            names = ['g1pre', 'g2pre']
        case 'post':
            names = ['g1post', 'g2post']

    for group_name in names:
        params, Cascade = load_audvis_files(os.path.join('pydata', group_name))
        AVclass = AUDVIS(*params, NAME = group_name, CASCADE=Cascade)

        g = int(group_name[1])
        cond = group_name[2:]

        data[g][cond] = AVclass

    av1 = data[1]['pre']
    av2 = data[1]['post']
    av3 = data[2]['pre']
    av4 = data[2]['post']
    
    match pre_post:
        case 'both':
            return av1, av2, av3, av4 # 'g1pre', 'g1post', 'g2pre', 'g2post'
        case 'pre':
            return av1, av3 # 'g1pre', 'g2pre'
        case 'post':
            return av2, av4 # 'g1post', 'g2post'
    

# if ran as a script, initializes 4 AUDVIS classes with loaded data and saves them
if __name__ == '__main__':
    args = parse_args()
    if args.files.lower() in ('y', 'yes', 'true', 't'):
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