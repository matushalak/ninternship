#@matushalak
from AUDVIS import AUDVIS, Behavior, load_in_data
from utils import show_me
import matplotlib.pyplot as plt
from matplotlib import artist
import seaborn as sns
from numpy import ndarray, arange, where, std, roll, unique, array, zeros, concatenate, save, percentile
from numpy.random import default_rng, Generator
from scipy.stats import wilcoxon, sem, norm
from collections import defaultdict
import zetapy as zeta # test for neuronal responsiveness by Montijn et al.
from multiprocessing import Pool, cpu_count
import os

class Analyze:
    ''' 
    class to perform analyses on neurons from one group and one condition
    '''
    def __init__(self, av:AUDVIS, storage_folder:str = 'results'):
        self.storage_folder = storage_folder
        # calculate time for plotting
        pre = round(-1/av.SF * av.trial_frames[0])
        post = (1/av.SF * av.trial_frames[1])
        self.time = arange(pre, post, 
                           1/av.SF)
        # Trial names
        self.tt_names = list(av.str_to_int_trials_map)
        # Analyze by trial type
        self.TT_RES = self.tt_average(av)

    
    # Analyze average response to trial type
    def tt_average(self, av:AUDVIS,
                   method:str = 'shuffle') -> tuple[list[ndarray], 
                                                    list[ndarray], 
                                                    list[ndarray]]:
        # 1. Baseline correct z-scored signal
        blc_z = av.baseline_correct_signal(signal=av.zsig)
        
        # shuffling method to identify responsive neurons           
        responsive_indices : ndarray = self.responsive(neurons = blc_z,
                                                       window = av.TRIAL,
                                                       criterion = ('trial_shuffle', 99.0),
                                                       av  = av,
                                                       n_shuffles = 1000)

        # 2. Separate into Trial=types
        tts_z : dict = av.separate_signal_by_trial_types(blc_z)

        # get significantly responding neurons for each trial type
        z_thresh = 2

        # 3. collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []
        # iterate through trial types
        for tt in tts_z.keys():
            trial_averaged_neurons = tts_z[tt].mean(axis = 0) # -> (times, neurons) shape
            
            # simple z-threshold method
            # responsive_indices = self.responsive(neurons = trial_averaged_neurons,
            #                                      window = av.TRIAL,
            #                                      criterion = z_thresh)

            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices)
            responsive_neurons = tts_z[tt][:,:,responsive_indices]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                  PLOT=False) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs) # ndarray at each tt idex gives neurons responsive to that tt

    def responsive(self, neurons:ndarray, window:tuple[int, int],
                   criterion:float|tuple[str,float], **kwargs)->ndarray:
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
                assert 'av' in kwargs and isinstance(kwargs['av'], AUDVIS) and 'n_shuffles' in kwargs, 'For trial_shuffle method, need AUDVIS object with all its attributes'
                AV : AUDVIS = kwargs['av']
                n_shuffles : int = kwargs['n_shuffles']
                rng = default_rng()
                random_shifts = rng.integers(low = 0, high = neurons.shape[1], size = n_shuffles)

                trials_real :ndarray = AV.trials # reference
                sess_neur = AV.session_neurons

                # parallelized execution
                args_list = [(neurons, sess_neur , trials_real, r_shift, window, i_shift) for i_shift, r_shift in enumerate(random_shifts)]
                with Pool(processes = 10) as pool:
                    max_distributions = pool.starmap(shuffle_loop, args_list, chunksize=100)
                
                # for each neuron and each condition will have a distribution of maximum values in trial window
                distributions : ndarray = array(max_distributions)
                save(os.path.join(self.storage_folder, AV.NAME + 'shuffle_dist.npy'), distributions)

                # debugging
                # for params in args_list:
                #     max_distributions = shuffle_loop(*params)
                
                breakpoint()
    



### Parallelization
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





### GENERAL Helper functions
def calc_avrg_trace(trace:ndarray, time:ndarray,
                    PLOT:bool = True, 
                    )->tuple[ndarray, ndarray, ndarray]:
    '''
    here trace to be averaged is for one neuron: (trials, times); 
                 or multiple responsive neurons: (trials, times, responsive_neurons)
    '''
    dims = trace.shape

    match dims:
        # plot average trace of single neuron
        case (n_trials, n_times):
            avrg = trace.mean(axis = 0)
            SEM = sem(trace, axis = 0) # std huge error bars

        # plot average trace of responsive neurons
        case (n_trials, n_times, n_neurons):
            avrg = trace.mean(axis = (0, 2))
            SEM = sem(trace, axis = (0, 2)) # std huge error bars

    if PLOT:
        plot_avrg_trace(time, avrg, SEM, title=f'{n_neurons}')    
    
    return (time, avrg, SEM)

def plot_avrg_trace(time:ndarray, avrg:ndarray, SEM:ndarray,
                    # optional arguments enable customization when this is passed into a function that assembles the plots in a grid
                    Axis:artist = plt, title:str = False, label:str = False, vspan:bool = True,
                    col:str = False, lnstl:str = False): 
    '''
    can be used standalone or as part of larger plotting function
    '''
    if title:
        Axis.set_title(title)  
    
    if vspan:
        Axis.axvspan(0,1,alpha = 0.05, color = 'g') # HARDCODED TODO: change based on trial duration
    
    Axis.fill_between(time, 
                    avrg - SEM,
                    avrg + SEM,
                    alpha = 0.35,
                    color = col if col else 'k',
                    linestyle = lnstl if lnstl else '-')
    if label:
        Axis.plot(time, avrg, label = label, color = col if col else 'k', linestyle = lnstl if lnstl else '-')
    else:
        Axis.plot(time, avrg, color = col if col else 'k', linestyle = lnstl if lnstl else '-')

    if Axis == plt:
        plt.show()
        plt.plot()
        plt.fill_between



### SPECIFIC analyses
def TT_ANALYSIS(tt_grid:dict[int:tuple[int, int]]):
    ''' 
    need to specify how you want to organize the conditions in the grid plot
    '''
    avs = load_in_data() # -> av1, av2, av3, av4
    TT_RESULTS = defaultdict(dict)
    
    fig, axs = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', figsize = (4 * 5, 2*4))
    linestyles = ('-', ':')
    colors = ('royalblue', 'goldenrod')
    for i, av in enumerate(avs):
        ANALYS = Analyze(av)
        trial_names = ANALYS.tt_names
        average_traces, sem_traces, responsive_neurs = ANALYS.TT_RES

        for i_tt, tt in enumerate(trial_names):
            gridloc = tt_grid[i_tt]
            plot_avrg_trace(ANALYS.time, avrg = average_traces[i_tt], SEM = sem_traces[i_tt],
                            Axis = axs[gridloc], title = tt, label = f'{av.NAME} ({len(responsive_neurs[i_tt])} / {len(list(av.neurons))})', 
                            vspan = (i == 0), col = colors[i >= 2], lnstl=linestyles[i%2])

            axs[gridloc].legend(loc = 2)
            if gridloc[1] == 0:
                axs[gridloc].set_ylabel('z(∆F/F)')
            if gridloc[0] == 1:
                axs[gridloc].set_xlabel('Time (s)')


    plt.tight_layout()
    plt.savefig('TT_res.png', dpi = 1000)
    plt.show()


### Main block that runs the file as a script
if __name__ == '__main__':
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    
    TT_ANALYSIS(tt_grid=tt_grid)
    