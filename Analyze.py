#@matushalak
from AUDVIS import AUDVIS, Behavior, load_in_data
from utils import show_me
import matplotlib.pyplot as plt
from matplotlib import artist
import seaborn as sns
from numpy import ndarray, arange, where
from scipy.stats import wilcoxon, sem

class Analyze:
    ''' 
    class to perform analyses on neurons from one group and one condition
    '''
    def __init__(self, av:AUDVIS):
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
    def tt_average(self, av:AUDVIS):
        # 1. Baseline correct z-scored signal
        blc_z = av.baseline_correct_signal(signal=av.zsig)
        
        # 2. Separate into Trial=types
        tts_z : dict = av.separate_signal_by_trial_types(blc_z)

        # get significantly responding neurons for each trial type
        z_thresh = 1.96

        # collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []
        # iterate through trial types
        for tt in tts_z.keys():
            trial_averaged_neurons = tts_z[tt].mean(axis = 0) # -> (times, neurons) shape
            responsive_indices = self.responsive(neurons = trial_averaged_neurons,
                                                 window = av.TRIAL,
                                                 criterion = z_thresh)
            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices)
            responsive_neurons = tts_z[tt][:,:,responsive_indices]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                  PLOT=True) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs) # ndarray at each tt idex gives neurons responsive to that tt

    def responsive(self, neurons:ndarray, window:tuple[int, int],
                   criterion:float|tuple[str,float])->ndarray:
        '''
        accepts (trials, times, neurons) type data structure for neurons within session or across sessions
                OR (times, neurons) for already aggregated average traces
        
        returns indices of responsive neurons (TO ONE CONDITION)
        '''
        # TODO: need neurons responsive AT LEAST to one condition, & keep track of which is which
        match (criterion, neurons.shape):
            # simply checking average traces of neurons and screening for neurons > criterion (can also look at >= abs(criterion) with .__abs__()
            case (criterion, (times, nrns)):
                return where(neurons[window[0]:window[1],:] >= criterion)[1]
            # other statistical tests etc.

# GENERAL Helper functions
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
            SEM = sem(trace, axis = 0)

        # plot average trace of responsive neurons
        case (n_trials, n_times, n_neurons):
            avrg = trace.mean(axis = (0, 2))
            SEM = sem(trace, axis = (0, 2))

    if PLOT:
        plot_avrg_trace(time, avrg, SEM, title=f'{n_neurons}')    
    
    return (time, avrg, SEM)

def plot_avrg_trace(time:ndarray, avrg:ndarray, SEM:ndarray,
                    # optional arguments enable customization when this is passed into a function that assembles the plots in a grid
                    Axis:artist = plt, title:str = False, label:str = False): 
    '''
    can be used standalone or as part of larger plotting function
    '''
    if title and Axis == plt:
        Axis.title(title)  
    
    Axis.axvspan(0,1,alpha = 0.15, color = 'k') # HARDCODED TODO: change based on trial duration
    Axis.fill_between(time, 
                    avrg - SEM,
                    avrg + SEM,
                    alpha = 0.2)
    if label:
        Axis.plot(time, avrg, label = label)
    else:
        Axis.plot(time, avrg)

    if Axis == plt:
        plt.show()
        plt.plot()

# Main block that runs this as a script
if __name__ == '__main__':
    avs = load_in_data() # -> av1, av2, av3, av4
    for av in avs: 
        Analyze(av)
    