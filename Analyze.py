#@matushalak
from AUDVIS import AUDVIS, Behavior, load_in_data
from utils import show_me
import matplotlib.pyplot as plt
from matplotlib import artist
import seaborn as sns
from numpy import ndarray, arange, where, std, roll, unique, array, zeros, concatenate, save, quantile, argsort
from scipy.stats import ttest_rel, sem, norm, _result_classes
from collections import defaultdict
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
        # trial is between these frames
        self.TRIAL_FRAMES = av.TRIAL

        # Trial names
        self.tt_names = list(av.str_to_int_trials_map)
        # Analyze by trial type
        self.TT_RES, self.TT_STATS, self.TTS_BLC_Z = self.tt_average(av)

    # Analyze average response to trial type
    def tt_average(self, av:AUDVIS,
                   method:str = 'ttest',
                   criterion:float = 1e-3,
                   **kwargs) -> tuple[list[ndarray], 
                                    list[ndarray], 
                                    list[ndarray]]:
        # 1. Baseline correct z-scored signal
        blc_z = av.baseline_correct_signal(signal=av.zsig)
    
        # 2. Separate into Trial=types
        tts_z : dict = av.separate_signal_by_trial_types(blc_z)

        # get significantly responding neurons for each trial type
        # 3. collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []

        TEST_RESULTS = []

        # iterate through trial types
        for tt in tts_z.keys():
            if 'RESPONSIVE' not in kwargs:
                # get indices responsive to that trial type
                responsive_indices, test_res = self.responsive_trial_locked(neurons = tts_z[tt],
                                                                            window = av.TRIAL,
                                                                            criterion = criterion,
                                                                            method = method)
                if method == 'ttest':
                    TEST_RESULTS.append(test_res)
            else:
                responsive_indices = kwargs['RESPONSIVE'][tt]

            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices)
            responsive_neurons = tts_z[tt][:,:,responsive_indices]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                    PLOT=False) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs # ndarray at each tt idex gives neurons responsive to that tt
                ), TEST_RESULTS, tts_z

    def responsive_trial_locked(self, neurons:ndarray, window:tuple[int, int],
                                criterion:float|tuple[str,float], method:str, **kwargs)->ndarray:
        '''
        accepts (trials, times, neurons) type data structure for neurons within session or across sessions
                OR (times, neurons) for already aggregated average traces
        
        returns indices of responsive neurons (TO ONE CONDITION)
        '''
        # TODO: need neurons responsive AT LEAST to one condition, & keep track of which is which
        # TODO: figure out how to handle responsive by suppression
        match (method, criterion, neurons.shape):
            # simply checking average traces of neurons and screening for neurons > criterion (can also look at >= abs(criterion) with .__abs__()
            case ('zscore', criterion, (n_trials, n_times, n_nrns)):
                trial_averaged_neurons = neurons.mean(axis = 0) # -> (times, neurons) shape
                return unique(where(trial_averaged_neurons[window[0]:window[1],:] >= criterion)[1]), None
            
            # do t-test on the mean values during trial
            case ('ttest', criterion, (n_trials, n_times, n_nrns)): 
                bls, trls = neurons[:,:self.TRIAL_FRAMES[0],:].mean(axis = 1), neurons[:,self.TRIAL_FRAMES[0]:self.TRIAL_FRAMES[1],:].mean(axis = 1)
                TTEST = ttest_rel(bls, trls, alternative = 'two-sided')
                return where(TTEST.pvalue < criterion)[0], TTEST
            
            



###-------------------------------- GENERAL Helper functions --------------------------------
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


def snake_plot(all_neuron_averages:ndarray, 
               stats:_result_classes.TtestResult,
               time:ndarray,
               Axis:artist = plt, title:str = False):
    '''
    Snake plot for all neurons of one group-condition-trial_type
    (neurons, time) shape heatmap
    '''
    average_trace_per_neuron = all_neuron_averages.mean(axis = 0).T # transpose so nice shape for heatmap
    sorted_by_significance = argsort(stats.pvalue)
    neurons_by_significance = average_trace_per_neuron[sorted_by_significance]
    # breakpoint()
    sns.heatmap(neurons_by_significance, xticklabels = False)
    plt.xticks(timestoplot := [0, 16, 32, 46], time.round()[timestoplot])
    plt.vlines([16, 32], 0, neurons_by_significance.shape[0])
    plt.show()




###--------------------------------SPECIFIC analyses----------------------------------------------------
def TT_ANALYSIS(tt_grid:dict[int:tuple[int, int]]):
    ''' 
    need to specify how you want to organize the conditions in the grid plot
    '''
    avs = load_in_data() # -> av1, av2, av3, av4
    TT_RESULTS = defaultdict(dict)
    
    fig1, axs1 = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', figsize = (4 * 5, 2*4))
    linestyles = ('-', ':')
    colors = ('royalblue', 'goldenrod')
    # fig2, axs2 = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', figsize = (4 * 5, 2*4))
    
    for i, av in enumerate(avs):
        ANALYS = Analyze(av)
        trial_names = ANALYS.tt_names
        average_traces, sem_traces, responsive_neurs = ANALYS.TT_RES

        for i_tt, tt in enumerate(trial_names):
            gridloc = tt_grid[i_tt]
            snake_plot(all_neuron_averages = ANALYS.TTS_BLC_Z[i_tt], stats=ANALYS.TT_STATS[i_tt],
                       time = ANALYS.time, title = tt)
 

            plot_avrg_trace(ANALYS.time, avrg = average_traces[i_tt], SEM = sem_traces[i_tt],
                            Axis = axs1[gridloc], title = tt, label = f'{av.NAME} ({len(responsive_neurs[i_tt])} / {len(list(av.neurons))})', 
                            vspan = (i == 0), col = colors[i >= 2], lnstl=linestyles[i%2])

            axs1[gridloc].legend(loc = 2)
            if gridloc[1] == 0:
                axs1[gridloc].set_ylabel('z(∆F/F)')
            if gridloc[0] == 1:
                axs1[gridloc].set_xlabel('Time (s)')


    fig1.tight_layout()
    fig1.savefig('TT_res.png', dpi = 1000)
    fig1.show()


### Main block that runs the file as a script
if __name__ == '__main__':
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    
    TT_ANALYSIS(tt_grid=tt_grid)
    