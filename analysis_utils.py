# @matushalak
# contains functions to calculate all sorts of things for analysis / plots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel, sem, norm, _result_classes
from matplotlib import artist
from typing import Literal


###-------------------------------- GENERAL Helper functions --------------------------------
def calc_avrg_trace(trace:np.ndarray, time:np.ndarray, PLOT:bool = True
                    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Here trace (∆F/F or spike_prob) to be averaged for: 
        - one neuron: (trials, times); 
        - multiple responsive neurons: (trials, times, responsive_neurons)
    
    For spike_prob (CASCADE input) the output is the estimated n_spikes per time_bin
    '''
    trace = np.squeeze(trace)
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

        case _:
            raise ValueError('Trace should be a np.ndarray with shape (ntrials, ntimes) or (ntrials, ntimes, nneurons).\n Instead provided trace is {} with shape {}'.format(type(trace), dims))
            
    if PLOT:
        plot_avrg_trace(time, avrg, SEM, title=f'{n_neurons}')    
    
    return (time, avrg, SEM)


def plot_avrg_trace(time:np.ndarray, avrg:np.ndarray, SEM:np.ndarray | None = None,
                    # optional arguments enable customization when this is passed into a function that assembles the plots in a grid
                    Axis:artist = plt, title:str = False, label:str = False, vspan:bool = True,
                    col:str = False, lnstl:str = False, alph:float = 1): 
    '''
    can be used standalone or as part of larger plotting function
    '''
    if title:
        Axis.set_title(title)  
    
    if vspan:
        Axis.axvspan(0,1,alpha = 0.05, color = 'navajowhite') # HARDCODED TODO: change based on trial duration
    
    if SEM is not None:
        Axis.fill_between(time, 
                        avrg - SEM,
                        avrg + SEM,
                        alpha = 0.35 * alph,
                        color = col if col else 'k',
                        linestyle = lnstl if lnstl else '-')
    if label:
        Axis.plot(time, avrg, label = label, color = col if col else 'k', linestyle = lnstl if lnstl else '-', alpha = alph)
    else:
        Axis.plot(time, avrg, color = col if col else 'k', linestyle = lnstl if lnstl else '-', alpha = alph)

    if Axis == plt:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_1neuron(all_trials_signal:list[np.ndarray], 
                 single_neuron:int,
                 session_neurons: tuple[list, list],
                 fluorescence:np.ndarray, 
                 trial_names:list[str], time:np.ndarray,
                 CASCADE : list[np.ndarray] = None,
                 STATS : list[object, object, np.ndarray] | None = None):
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    plt.close()
    fig, axs = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', 
                            figsize = (4 * 5, 2*5))
    # for each trial type
    for itt, ttsig in enumerate(all_trials_signal.values()):
        trial_STATS = STATS[itt]
        zetalabel = f'ZETA: ({trial_STATS[0][single_neuron,:].round(3)})\n'
        ttestlabel = f'TTEST: {(float(round(trial_STATS[1].pvalue[single_neuron],3)), float(round(trial_STATS[1].statistic[single_neuron], 3)))}\n'
        wilcoxonlabel = f'WILCOXON: {(float(round(trial_STATS[2].pvalue[single_neuron], 3)), float(round(trial_STATS[2].statistic[single_neuron], 3)))}\n'

        if CASCADE is not None:
            axCASC = axs[tt_grid[itt]].twinx()

        axs[tt_grid[itt]].axvspan(0,1,alpha = 0.1, color = 'khaki')
        neuron_all_trials = ttsig[:,:,single_neuron] 
        # plot all raw traces for each trial
        for i_trial in range(neuron_all_trials.shape[0]):
            axs[tt_grid[itt]].plot(time, neuron_all_trials[i_trial, :], alpha = 0.01, color = 'blue')
            if CASCADE is not None:
                axCASC.plot(time, CASCADE[itt][i_trial, :, single_neuron], alpha = 0.01, color = 'green')
        # as well as the average trace across trials
        axs[tt_grid[itt]].plot(time, trace := neuron_all_trials.mean(axis = 0), color = 'blue', label = zetalabel + ttestlabel + wilcoxonlabel)
        if CASCADE is not None:
            axCASC.plot(time, CASCADE[itt][:, :, single_neuron].mean(axis = 0), color = 'green', label = 'CASCADE')
        
        # Fluorescence response
        axs[tt_grid[itt]].set_title(f'{trial_names[itt]}_FR(µ : {round(fluorescence[single_neuron,0,itt], 3)}, sd : {round(fluorescence[single_neuron,1,itt], 3)}')
        # 0 line
        axs[tt_grid[itt]].hlines(y = 0, xmin = -1, xmax = 2, color = 'r')
        axs[tt_grid[itt]].hlines(y = (mn := np.mean(trace[16:32])), xmin = -1, xmax = 2, color = 'k', alpha = 0.6, 
                                 label = f'av_trace mean {round(mn, 3)}')
        axs[tt_grid[itt]].scatter(x = time[np.argmax(trace[16:32]) + 16], y = (maxi := np.max(trace[16:32])), color = 'gold', 
                                 label = f'av_trace max {round(maxi, 3)}')
        axs[tt_grid[itt]].scatter(x = time[np.argmin(trace[16:32]) + 16], y = (mini := np.min(trace[16:32])), color = 'magenta', 
                                 label = f'av_trace min {round(mini, 3)}')
    
        axs[tt_grid[itt]].legend(loc = 3, fontsize = 8)
        axCASC.legend(loc = 4, fontsize = 8)
    # find back neuron
    ranges, names = session_neurons
    for sess_range, sessname in zip(ranges, names):
        if single_neuron >= sess_range[0] and single_neuron < sess_range[1]:
            sess_index = single_neuron - sess_range[0]
            session = sessname

    fig.suptitle(f'Neuron_{sess_index} from {session}, overall_id ({single_neuron})')
    fig.tight_layout()
    plt.show()
    plt.close()
    

# TODO: add for Neuron-type (& by brain region) analyses !!!
def build_snake_grid(tt_grid):
    """
    Given the original tt_grid (mapping trial_type -> (row, col) in 2x4),
    produce a dictionary snake_grid that maps (trial_type, heatmap_index)
    -> (row, col) in a 4x8 figure.
    """
    snake_grid = {}
    for i_tt, (r0, c0) in tt_grid.items():
        for i in range(4):
            r_off = i // 2
            c_off = i % 2
            snake_grid[(i_tt, i)] = (2*r0 + r_off, 2*c0 + c_off)
    return snake_grid


def snake_plot(all_neuron_averages:np.ndarray, 
               stats:_result_classes.TtestResult | np.ndarray,
               trial_window_frames:tuple[int, int], time:np.ndarray,
               heatmap_range:tuple[int, int] = (None, None),
               Axis:artist = plt, title:str = False,
               colorbar:bool = True, SHOW:bool = False,
               MODE:Literal['onset', 'signif'] = 'onset'):
    '''
    One snake plot for all neurons of one group-condition-trial_type
    (neurons, time) shape heatmap
    '''
    if title:
        Axis.set_title(title)  

    average_trace_per_neuron = all_neuron_averages.mean(axis = 0).T # transpose so nice shape for heatmap
    
    # response onset sorting
    if MODE == 'onset':
        sorted_by_response_onset = np.argsort(np.argmax(average_trace_per_neuron[:,trial_window_frames[0]:trial_window_frames[1]], axis = 1))
        heatmap_neurons = average_trace_per_neuron[sorted_by_response_onset]

    # significance sorting
    elif MODE == 'signif':
        sorted_by_significance = np.argsort(stats.pvalue) if not isinstance(stats, np.ndarray) else np.argsort(stats[:,0])
        heatmap_neurons = average_trace_per_neuron[sorted_by_significance]

    sns.heatmap(heatmap_neurons, vmin = heatmap_range[0], vmax=heatmap_range[1],
                xticklabels = False, ax = Axis, cbar = colorbar)
    
    Axis.vlines(trial_window_frames, ymin = 0, ymax=heatmap_neurons.shape[0])
    Axis.set_xticks(timestoplot := [0, trial_window_frames[0], trial_window_frames[1], heatmap_neurons.shape[1]-1], 
                        time.round()[timestoplot])
    if SHOW:
        plt.show()
