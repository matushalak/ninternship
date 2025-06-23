# @matushalak
# contains functions to calculate all sorts of things for analysis / plots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Categorical
import pandas as pd
from scipy.stats import sem, norm, _result_classes
from matplotlib import artist
from collections import defaultdict
from typing import Literal, Iterable, Callable
import os

from src import PYDATA, PLOTSDIR

###-------------------------------- GENERAL Helper functions --------------------------------
def get_proportionsDF(nbDF:DataFrame,
                      countgroupby:list[str],
                      propgroupby:list[str],)->DataFrame:
    # 0) force all three neighbor‐types to appear
    nbDF = nbDF.copy()
    for col in countgroupby:
        nbDF[col] = nbDF[col].astype('category')

    # 1) count
    counts = (
        nbDF
        .groupby(countgroupby, observed=False)
        .size()
        .reset_index(name='count')
    )
    # 2) within‐block normalization
    counts['prob'] = (
        counts
        .groupby(propgroupby)['count']
        .transform(lambda x: x / x.sum())
    )
    return counts

def get_distancesDF(distDF:DataFrame,
                    groupbyVAR:list[str], valueVAR:str):
    dist = (distDF
            .groupby(groupbyVAR)[valueVAR]
            .mean()
            .reset_index(name='mean_dist'))
    return dist

def local_plotter(data:DataFrame, plot_func:Callable, whichframe:str, 
                  **kwargs):
        data = data.loc[data['frame']==whichframe,:]
        kwargs['hue'] = data[[kwargs['x'], kwargs['hue']]].apply(tuple, axis = 1)
        palette = {('V', 'DRpre'):'dodgerblue',
                   ('A', 'DRpre'):'red',
                   ('M', 'DRpre'):'goldenrod',
                   ('V', 'NRpre'):'lightskyblue',
                   ('A', 'NRpre'):'lightsalmon',
                   ('M', 'NRpre'):'palegoldenrod'}
        kwargs['palette'] = palette
        kwargs['hue_order'] = (('V', 'DRpre'), ('A', 'DRpre'), ('M', 'DRpre'), 
                               ('V', 'NRpre'),('A', 'NRpre'),('M', 'NRpre'),
                               )
        plot_func(data = data, **kwargs)

def catplot_proportions(DF:DataFrame, 
                        countgroupby:list[str],
                        propgroupby:list[str],
                        NULLDF:DataFrame | None = None,
                        plotname:str = 'catplot',
                        svg:bool = True,
                        show:bool = False,
                        x: str | None = None,
                        hue: str | None = None,
                        row: str | None = None,
                        col: str | None = None,
                        kind: Literal['strip', 'swarm', 'box', 'violin', 
                                      'boxen', 'point', 'bar', 'count'] = "point",
                        estimator: str = "mean",
                        order: Iterable[str] | None = None,
                        hue_order: Iterable[str] | None = None,
                        row_order: Iterable[str] | None = None,
                        col_order: Iterable[str] | None = None,
                        savedir : str |None = None):
    if savedir is None:
        savedir = os.path.join(PLOTSDIR, plotname)
    else:
        savedir = os.path.join(savedir, plotname)
    
    counts = get_proportionsDF(DF, countgroupby, propgroupby)
    null = NULLDF.drop(columns=['mean_dist']).copy()

    data = pd.concat([counts.assign(frame = 'counts'),
                      null.assign(frame = 'null')], ignore_index=True)
    
    g = sns.FacetGrid(data, row = row, col = col, 
                      aspect=2 if col is not None else 1,
                      row_order=row_order, col_order=col_order)

    if NULLDF is not None:
        g.map_dataframe(local_plotter,
                        plot_func = sns.pointplot,
                        whichframe = 'null',
                        x = x, y = 'prob',
                        palette='dark:#1f77b4',
                        hue = hue,
                        errorbar = 'pi', # percentile interval (2.5 - 97.5)
                        capsize = .3,
                        dodge = 0.5,
                        marker = "", linestyle = "none",
                        order = order)
        
    # 3. Draw a bar plot of those probs
    g.map_dataframe(local_plotter,
                    plot_func = sns.pointplot,
                    whichframe = 'counts',
                    x =x, y = 'prob',
                    palette='dark:#1f77b4',
                    hue = hue,
                    errorbar = None,
                    marker = "x", linestyle = 'none',
                    dodge = 0.5,
                    order = order)
    
    g.set_axis_labels(x,"Probability")
    g.add_legend()
    g.set(ylim=(-0.05,1))

    extension = '.png' if not svg else '.svg'
    plt.savefig(savedir+extension, dpi = 300 if not svg else 'figure')
    if show:
        plt.show()
    plt.close()

# TODO: combine into one function with above
def catplot_distanc(DF:DataFrame, 
                    NULLDF:DataFrame | None = None,
                    plotname:str = 'catplotDIST',
                    svg:bool = True,
                    show:bool = False,
                    x: str | None = None,
                    y: str | None = None,
                    hue: str | None = None,
                    row: str | None = None,
                    col: str | None = None,
                    kind: Literal['strip', 'swarm', 'box', 'violin', 
                                    'boxen', 'point', 'bar', 'count'] = "point",
                    estimator: str = "mean",
                    order: Iterable[str] | None = None,
                    hue_order: Iterable[str] | None = None,
                    row_order: Iterable[str] | None = None,
                    col_order: Iterable[str] | None = None,
                    savedir : str |None = None):
    if savedir is None:
        savedir = os.path.join(PLOTSDIR, plotname)
    else:
        savedir = os.path.join(savedir, plotname)
        
    null = NULLDF.drop(columns=['prob', 'count']).copy()
    null.rename(columns={'mean_dist':'Distance'}, inplace=True)
    
    dist = DF.drop(columns=['NeuronID']).copy()
    dist.loc[:, 'NeighborTYPE'] = dist['NNtype'].transform(lambda x: x[0]).to_numpy().astype(str)
    dist.drop(columns=['NNtype'], inplace=True)
    
    data = pd.concat([dist.assign(frame = 'dist'),
                      null.assign(frame = 'null')], ignore_index=True)
    
    g = sns.FacetGrid(data, row = row, col = col, 
                      aspect=2 if col is not None else 1,
                      row_order=row_order, col_order=col_order)

    if NULLDF is not None:
        g.map_dataframe(local_plotter,
                        plot_func = sns.pointplot,
                        whichframe = 'null',
                        x = x, y = y,
                        palette='dark:#1f77b4',
                        hue = hue,
                        errorbar = 'pi', # percentile interval (2.5 - 97.5)
                        capsize = .3,
                        dodge = 0.5,
                        marker = "", linestyle = "none",
                        order = order)
        
    # 3. Draw a bar plot of those probs
    g.map_dataframe(local_plotter,
                    plot_func = sns.pointplot,
                    whichframe = 'dist',
                    x =x, y = y,
                    palette='dark:#1f77b4',
                    hue = hue,
                    errorbar = None,
                    marker = "x", linestyle = 'none',
                    dodge = 0.5,
                    order = order)
    
    g.add_legend()

    extension = '.png' if not svg else '.svg'
    plt.savefig(savedir+extension, dpi = 300 if not svg else 'figure')
    if show:
        plt.show()
    plt.close()

# for baseline - distribution not necessary
def tt_fluoro_func(sig:np.ndarray,
                   fluoro_kwargs:dict,
                   sigseparate_kwargs:dict
                   )->np.ndarray:
    ttdict = general_separate_signal(sig, **sigseparate_kwargs)
    return fluorescence_response(ttdict, **fluoro_kwargs)


def fluorescence_response (signal: np.ndarray | dict[int:np.ndarray],
                        window: tuple[int, int],
                        returnF: bool = False,
                        method: Literal['mean', 'peak'] = 'peak',
                        retMEAN_only:bool = False) -> np.ndarray:
    '''
    Takes
    ---------
        1) (trial, times, neurons) array and 
        
        2) {trial_type : (trial, times, neurons) array} dictionary
    
    Returns
    ---------
        1) (neurons, stats) array with mean and std (dim 1) 
        of fluorescence response (F) to a given trial-type for each neuron (dim 0)

        2) returns a (neurons, stats, trial_types) array with mean and std and Cohen's d (dim 1) 
        of fluorescence response (F) to each trial-type (dim 2) for each neuron (dim 0)
    '''
    assert isinstance(signal, dict
                        ) or isinstance(signal, np.ndarray
                                        ), 'Signal must either be a dictionary with signal arrays for each trial type OR just a signal array for one trial type'
    ncols = 1 if retMEAN_only else 3

    if isinstance(signal, dict):
        nts = signal[0].shape[1]
        res = np.zeros((signal[0].shape[-1], ncols, len(signal)))
        tt_sigs = [signal[tt] for tt in sorted(signal.keys())]
    elif isinstance(signal, np.ndarray):
        nts = signal.shape[1]
        res = np.zeros((signal.shape[-1], ncols, 1))
        tt_sigs = [signal]

    assert all(len(sig.shape) == 3 for sig in tt_sigs), 'Signal arrays must be 3D - (ntrials, ntimes, nneurons)!'

    # TODO: add customizability for sub-windows
    if method == 'peak':
        # Setup logic for detecting responses at different latencies
        window_dur = window[1] - window[0]
        half_window = window_dur // 2
        windows_offsets = np.arange(3) * half_window
        assert window[1] + windows_offsets[-1] <= nts, 'Proposed response window falls outside trial'
        response_windows = [slice(window[0]+wo, window[1]+wo) 
                            for wo in windows_offsets]

    if returnF:
        Fs = []
    for itt, sig_array in enumerate(tt_sigs):
        # Fluorescence response adapted from (Meijer et al., 2017)
        # mean fluorescence during stimulus presentation for all trials
        Fmean = np.nanmean(sig_array[:,window[0]:window[1],:], axis = 1)
        F = Fmean.copy()
        if method == 'peak':
            # mean for each of the three response windows
            Fmeans_all_rw = np.dstack([np.nanmean(sig_array[:, rw, :], axis=1) 
                                       for rw in response_windows])
            # tries to estimate if after stimulus onset excitatory or inhibitory response
            Fmedian = np.nanmedian(sig_array[:,window[0]:,:], axis = 1)
            exc, inh = Fmedian >= 0, Fmedian < 0

            Fmax = np.nanmax(Fmeans_all_rw, axis=2)
            Fmin = np.nanmin(Fmeans_all_rw, axis=2)
            
            F[exc] = Fmax[exc]
            F[inh] = Fmin[inh]
            
        if returnF:
            Fs.append(F) # add trial-level
        # signal being fed in is already baseline corrected, so all these
        # are about ∆FR
        # this should get rid of the nans
        meanF = np.nanmean(F, axis = 0) # mean fluorescence response over trials
        res[:, 0, itt] = meanF
        
        if not retMEAN_only:
            stdF = np.nanstd(F, axis = 0) # std of fluorescence response over trials
            cohdF = meanF / stdF # Cohen's d of fluorescence response over trials

            res[:, 1, itt] = stdF
            res[:, 2, itt] = cohdF

    # assert not np.isnan(res).any(), '[BUG]: NaNs were NOT removed during FR aggregation per neuron!!!'
    if not returnF:
        return np.squeeze(res) # removes trailing dimension in case want output only for 1 trial type
    else:
        return np.array(Fs)


def general_separate_signal(sig:np.ndarray,
                            trial_types:np.ndarray|list,
                            trial_type_combinations:list[tuple]|None = None,
                            separation_labels:list[str]|None = None
                            )-> dict[str | int : np.ndarray]:
    '''
    Input:
    ------------
    sig : 2D / 3D np.ndarray 
        w dimensions (ntrials, nts, (optional: nneurons))
    trial_types : 1D np.ndarray 
        shape (ntrials)
    trial_type_combinations : list[tuple[int, ...], ...] 
        individual entries of this list contain trial types that will be grouped.
            if None, separate signal between all unique values within trial_types
    separation_labels : list[str, ...] 
        labels for entries of trial_type_combinations.
            if None, give integer labels from 0 : len(trial_type_combinations) - 1
    
    Returns:
    ------------
    separated_signal: dict[str | int : np.ndarray(nTT_trials, nts, (optional: nneurons))]
    '''
    if isinstance(trial_types, list):
        trial_types = np.array(trial_types)
    assert sig.shape[0] == trial_types.shape[0], (
        f'First dimension of signal {sig.shape[0]} must match first dimension of trial_types {trial_types.shape[0]}'
        )
    if trial_type_combinations is None:
        trial_type_combinations = [(tt,) for tt in np.unique(trial_types)]
    
    if separation_labels is None:
        separation_labels = [i for i in range(len(trial_type_combinations))]
    
    separated_signal = dict()

    for label, tt_combo in zip (separation_labels, trial_type_combinations):
        tt_indices = np.where(np.isin(trial_types, tt_combo))[0]
        # Ellipses (...) slices everything from following dimensions
        separated_signal[label] = sig[tt_indices, ...]

    return separated_signal
    

def group_separated_signal(trial_type_combinations:list[tuple],
                           separation_labels:list[str],
                           tt_separated_signal:dict[int:np.ndarray]):
    '''
    Interface to group trial-types of dictionary created by src.AUDVIS.separate_signal_by_trial_types
    '''
    oldTTs = list(tt_separated_signal)
    out = defaultdict(list)

    assert len(trial_type_combinations) == len(separation_labels)
    for newTT, newTTCOMBO in zip(separation_labels, trial_type_combinations):
        newARRs = [tt_separated_signal[tt] for tt in oldTTs if tt in newTTCOMBO]
        out[newTT] = np.vstack(newARRs)
    
    return out


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
            avrg = np.nanmean(trace, axis=0)
            SEM = sem(trace, axis = 0, nan_policy='omit') # std huge error bars
            
        # plot average trace of responsive neurons
        case (n_trials, n_times, n_neurons):
            if n_neurons != 0:
                avrg = np.nanmean(trace, axis = (0, 2))
                SEM = sem(trace, axis = (0, 2), nan_policy='omit') # std huge error bars
            else:
                # this happens if a brain area has no intersection with the significant neurons,
                # then we actually want to return NaNs
                avrg = np.full(n_times, np.nan)
                SEM = np.full(n_times, np.nan)

        case _:
            raise ValueError('Trace should be a np.ndarray with shape (ntrials, ntimes) or (ntrials, ntimes, nneurons).\n Instead provided trace is {} with shape {}'.format(type(trace), dims))
            
    if PLOT:
        plot_avrg_trace(time, avrg, SEM, title=f'{n_neurons}')    
    
    # this happens if a brain area has no intersection with the significant neurons,
    # then we actually want to return NaNs
    if dims[-1] != 0: 
        # otherwise, nans are not acceptable
        assert not np.isnan(avrg).any(), '[BUG]: NaNs present in average trace'
        assert not np.isnan(SEM).any(), '[BUG]: NaNs present in SEM trace'
    return (time, avrg, SEM)


def plot_avrg_trace(time:np.ndarray, avrg:np.ndarray, SEM:np.ndarray | None = None,
                    # optional arguments enable customization when this is passed into a function that assembles the plots in a grid
                    Axis:artist = plt, title:str = False, label:str = False, vspan:bool = True,
                    col:str = False, lnstl:str = False, alph:float = 1, tt: int = 0): 
    '''
    can be used standalone or as part of larger plotting function
    '''
    if title:
        Axis.set_title(title)  
    
    if vspan:
        tt_to_col = {0:'dodgerblue',
                     1:'red',
                     2:'goldenrod',
                     3:'goldenrod'}
        # Axis.axvspan(0,1,alpha = 0.15, color = tt_to_col[tt]) # HARDCODED TODO: change based on trial duration
        Axis.axvline(x=0, linestyle = 'dashed', color = tt_to_col[tt])
        Axis.axvline(x=1, linestyle = 'dashed', color = tt_to_col[tt])

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
            # nans will be ignored
            axs[tt_grid[itt]].plot(time, neuron_all_trials[i_trial, :], alpha = 0.1, color = 'blue')
            if CASCADE is not None:
                axCASC.plot(time, CASCADE[itt][i_trial, :, single_neuron], alpha = 0.1, color = 'green')
        # as well as the average trace across trials
        trace = np.nanmean(neuron_all_trials, axis = 0)
        axs[tt_grid[itt]].plot(time, trace, color = 'blue', label = zetalabel + ttestlabel + wilcoxonlabel)
        if CASCADE is not None:
            CASCADEtrace = np.nanmean(CASCADE[itt][:, :, single_neuron], axis = 0)
            axCASC.plot(time, CASCADEtrace, color = 'green', label = 'CASCADE')
        
        # Fluorescence response
        axs[tt_grid[itt]].set_title(
            f'{trial_names[itt]}_FR(µ : {round(fluorescence[single_neuron,0,itt], 3)}, sd : {round(fluorescence[single_neuron,1,itt], 3)}, d: {round(fluorescence[single_neuron,2,itt], 3)}')
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
               MODE:Literal['onset', 'signif'] = 'onset',
               cmap:str | None = None):
    '''
    One snake plot for all neurons of one group-condition-trial_type
    (neurons, time) shape heatmap
    '''
    if title:
        Axis.set_title(title)  

    average_trace_per_neuron = np.nanmean(all_neuron_averages, axis = 0).T # transpose so nice shape for heatmap
    
    # response onset sorting
    if MODE == 'onset':
        sorted_by_response_onset = np.argsort(np.argmax(average_trace_per_neuron[:,trial_window_frames[0]:trial_window_frames[1]], axis = 1))
        heatmap_neurons = average_trace_per_neuron[sorted_by_response_onset]

    # significance sorting
    elif MODE == 'signif':
        sorted_by_significance = np.argsort(stats.pvalue) if not isinstance(stats, np.ndarray) else np.argsort(stats[:,0])
        heatmap_neurons = average_trace_per_neuron[sorted_by_significance]

    nans = np.any(np.isnan(heatmap_neurons), axis = 1)
    heatmap_neurons = heatmap_neurons[~nans]
    if cmap is not None:
        cmap = sns.light_palette(cmap, as_cmap=True, reverse=False)
    sns.heatmap(heatmap_neurons, vmin = heatmap_range[0], vmax=heatmap_range[1],
                xticklabels = False, ax = Axis, cbar = colorbar, cmap=cmap)
    
    Axis.vlines(trial_window_frames, ymin = 0, ymax=heatmap_neurons.shape[0])
    Axis.set_xticks(timestoplot := [0, trial_window_frames[0], trial_window_frames[1], heatmap_neurons.shape[1]-1], 
                        time.round()[timestoplot])
    if SHOW:
        plt.show()
