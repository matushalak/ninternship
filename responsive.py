from AUDVIS import AUDVIS
import zetapy as zeta # test for neuronal responsiveness by Montijn et al.
from numpy import ndarray, unique, where, array, save, zeros, concatenate
from numpy.random import default_rng, Generator
from multiprocessing import Pool, cpu_count
import os

def responsive(signal:ndarray, window:tuple[int, int],
               **kwargs)->ndarray:
        '''
        accepts (trials, times, neurons) type data structure for neurons within session or across sessions
                OR (times, neurons) for already aggregated average traces
        
        returns indices of responsive neurons (TO ONE CONDITION)
        '''
        # TODO: need neurons responsive AT LEAST to one condition, & keep track of which is which
        # TODO: figure out how to handle responsive by suppression
        match (criterion, neurons.shape):
            # simply checking average traces of neurons and screening for neurons > criterion (can also look at >= abs(criterion) with .__abs__()
            case (criterion, (n_times, n_nrns)):
                return unique(where(neurons[window[0]:window[1],:] >= criterion)[1])
            
            # trial_shuffle procedure (all_trials, times, neurons), criterion = percentile of shuffled distribution
            case (('trial_shuffle', crit), (n_all_trials, n_times, n_nrns)):
                # 0) setup
                assert 'av' in kwargs and isinstance(kwargs['av'], AUDVIS) and 'n_shuffles' in kwargs, 'For trial_shuffle method, need AUDVIS object with all its attributes'
                AV : AUDVIS = kwargs['av']
                n_shuffles : int = kwargs['n_shuffles']
                rng = default_rng()
                random_shifts = rng.integers(low = 0, high = neurons.shape[1], size = n_shuffles)

                trials_real :ndarray = AV.trials # reference

                # 1) get distributions for each neuron and condition
                # parallelized execution
                sess_neur = AV.session_neurons
                args_list = [(neurons, sess_neur , trials_real, r_shift, window, i_shift) for i_shift, r_shift in enumerate(random_shifts)]
                with Pool(processes = 10) as pool:
                    max_distributions = pool.starmap(shuffle_loop, args_list, chunksize=100)
                
                # for each neuron and each condition will have a distribution of maximum values in trial window
                # (n_shuffles, n_trial_types, n_neurons)
                distributions : ndarray = array(max_distributions)
                save(os.path.join(self.storage_folder, AV.NAME + 'shuffle_dist.npy'), distributions)

                # 2) get responsive neurons       
                tts_z : dict = AV.separate_signal_by_trial_types(neurons)
                breakpoint()
    



###-------------------------------- functions for CPU Parallelization--------------------------------
def shuffle_loop(neurons:ndarray, session_neurons:list[tuple[int, int]], trials_real:ndarray, 
                rand_shift:int, window:tuple[int, int], i:int):
    rng : Generator = default_rng()
    trial_types_list = unique(trials_real[0,:])
    max_distribution : list[ndarray] = [zeros(neurons.shape[-1]) # max for each neuron
                                        for tt in trial_types_list] # max for each condition 
    # shuffles trial IDs in each session
    trials_shuffled : ndarray = rng.permuted(trials_real, axis = 1)
    # dictionary specifying which trial type where
    trial_types_shuffled : dict = {tt : where(trials_shuffled == tt) # this gives the [session (1:9)], [trial (1:720)] indices for each trial type
                                    for tt in unique(trials_shuffled)}
    # shuffle calcium traces for all neurons across sessions by a random shift
    signal_shuffled : ndarray = roll(neurons, shift = rand_shift, axis = 1)
    
    tt_signals_shuffled : dict[int : ndarray] = signal_by_TT(signal= signal_shuffled,
                                                            all_trials = trials_shuffled,
                                                            ttypes = trial_types_shuffled,
                                                            session_neurons = session_neurons)
    # get the maximum in the trial window of the trial-averaged TRIAL-SHUFFLED & CIRCULARLY TIME-SHUFFLED trace
    for tt in trial_types_list:
        average_shuffled_signal :ndarray = tt_signals_shuffled[tt].mean(axis = 0)
        max_distribution[tt] = average_shuffled_signal[window[0]:window[1], :].max(axis = 0)

    print('shuffle', i, ' done')
    return max_distribution

# Repeated here because AUDVIS object cannot be passed to workers to be parallelized
def signal_by_TT(signal:ndarray,
                **kwargs) -> dict[int:ndarray]:
    '''Only works on trial-locked signal,
    signal is assumed to be in (trials_all_conditions, time, neurons_across sessions) format
    
    returns a dictionary that stores separate arrays for each trial type (trials_one_condition, time, neurons_across_sessions)
    '''
    all_trials : ndarray = kwargs['all_trials']
    assert isinstance(all_trials, ndarray), 'all_trials must be (session, trials) array'        
    trials_per_TT = round(signal.shape[0] / len(unique(all_trials)))
    
    signal_by_trials = dict()

    # split signal from different recordings before joining
    ttypes : dict[int:ndarray] = kwargs['ttypes']
    assert isinstance(ttypes, dict), 'ttypes must be a dictionary with indices for all occurences of each trial type across sessions'
    
    # session_neurons
    session_neurons : list[tuple[int, int]] = kwargs['session_neurons']
    assert isinstance(session_neurons, list), 'session_neurons must be a nested list of tuples with indices for neuron ranges for each session'
    
    for tt in list(ttypes): # 0-n_trial types
        signal_tt = []
        for (n_first, n_last) in session_neurons:
            _, trials = ttypes[tt]
            trials = trials[tt*trials_per_TT : (tt+1)*trials_per_TT]
            signal_tt.append(signal[trials,:,n_first:n_last])
        
        signal_by_trials[tt] = concatenate(signal_tt,
                                            axis = 2)
    return signal_by_trials