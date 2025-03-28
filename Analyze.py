#@matushalak
from AUDVIS import AUDVIS, Behavior, load_in_data
from Responsive import responsive_zeta
from utils import show_me
import matplotlib.pyplot as plt
from matplotlib import artist
from matplotlib_venn import venn3
import seaborn as sns
from numpy import ndarray, arange, where, unique, argsort, argmax
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, sem, norm, _result_classes
from collections import defaultdict
from typing import Literal
import os

# TODO: by brain area (average, snake, Venn Diagram / PieChart)
# TODO: multisensory integration (for modulated VIS and AUD neurons), response selectivity index
# TODO: congruent vs incongruent -> FOR MODULATED neurons & for MST_only neurons
# TODO: anova for detecting responsiveness higher yield?

class Analyze:
    ''' 
    class to perform analyses on neurons from one group and one condition
    '''
    def __init__(self, av:AUDVIS, storage_folder:str = 'results', include_zeta : bool = True):
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

        if include_zeta:
            # Zeta test results
            self.TT_zeta = responsive_zeta(SPECIFIEDcond = av.NAME)
        
        # Analyze by trial type
        self.TT_RES, self.TT_STATS, self.TTS_BLC_Z, self.TTS_Z, self.NEURON_groups = self.tt_average(av)

    # Analyze average response to trial type
    # NOTE: currently for z-score data, better to generalize to any signal
    def tt_average(self, av:AUDVIS,
                   method:str = 'zeta',
                   criterion:float = 0.01 # p-value threshold (bonferroni corrected)
                   ) -> tuple[list[ndarray], 
                              list[ndarray], 
                              list[ndarray]]:
        SIG = av.zsig_CORR
        # 1. Baseline correct z-scored signal; residual signal without running speed OR whisker movement
        BLC_SIG = av.baseline_correct_signal(signal=SIG)

        # 2. Separate into Trial=types
        tts_z : dict = av.separate_signal_by_trial_types(SIG) # z-scored signal
        blc_tts_z : dict = av.separate_signal_by_trial_types(BLC_SIG) # baseline-corrected z-scored signal

        # get significantly responding neurons for each trial type
        # 3. collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []

        TEST_RESULTS = []
        # iterate through trial types
        for tt in tts_z.keys():
            # get indices responsive to that trial type
            responsive_indices, test_res = self.responsive_trial_locked(neurons = tts_z[tt] if method != 'zeta' else self.TT_zeta[tt],
                                                                        window = av.TRIAL,
                                                                        criterion = criterion / len(list(tts_z)) if method != 'zscore' else criterion, # bonferroni correction
                                                                        method = method)
            if method != 'zscore':
                TEST_RESULTS.append(test_res)
            print(tts_z[tt].shape, len(self.TT_zeta[tt]))
            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices)
            # save baseline corrected signal
            responsive_neurons = blc_tts_z[tt][:,:,responsive_indices]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                    PLOT=False) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs # ndarray at each tt idex gives neurons responsive to that tt
                ), TEST_RESULTS, blc_tts_z, tts_z, self.neuron_groups(responsive = responsive_neurs)

    def responsive_trial_locked(self, neurons:ndarray | list, window:tuple[int, int],
                                criterion:float|tuple[str,float], method:str)-> tuple[ndarray, ndarray | None]:
        '''
        accepts (trials, times, neurons) type data structure for neurons within session or across sessions
                OR (times, neurons) for already aggregated average traces
        
        returns indices of responsive neurons (TO ONE CONDITION)
        '''
        if isinstance(neurons, list):
            neurons = np.array(neurons)
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
            
            # wilcoxon signed-rank test
            case ('wilcoxon', criterion, (n_trials, n_times, n_nrns)):
                # NOTE: average over time bins (axis = 0) vs average over baseline vs stimulus window in each trial (axis = 1)(before)
                bls, trls = np.mean(neurons[:,:self.TRIAL_FRAMES[0],:],axis = 1), np.mean(neurons[:,self.TRIAL_FRAMES[0]:self.TRIAL_FRAMES[1],:], axis = 1)
                WCOX = wilcoxon(bls, trls, alternative='two-sided')
                return where(WCOX.pvalue < criterion)[0], WCOX
            
            # already have zeta significances and Zeta values (n_nrns, 2[zeta_p, zeta_score])
            case ('zeta', criterion, (n_nrns, stats)):
                # p-value below threshold
                return where(neurons[:,0] < criterion)[0], neurons
        
            # TODO: z-score proportion trials (zscore > x in X proportion of trials)
            
    @staticmethod
    def neuron_groups(responsive:list[ndarray]) -> dict[str : set]:
        '''
        Input:
            for each trial type, gives neurons significantly responsive to it
        Output:
            returns VIS neurons, AUD neurons and MST neurons for further Venn diagram analysis
        '''            
        resp_sets = [set(arr) for arr in responsive]
        # First define the big sets as all neurons that showed response to ONE of the AUD / VIS / MST trials
        # AUD: union between TT 0 (Al) and 3 (Ar)
        AUD_set = resp_sets[0] | resp_sets[3] 
        # VIS: union between TT 6 (Vl) and 7 (Vr)
        VIS_set = resp_sets[6] | resp_sets[7] 
        # MST: union between TT {1 (AlVl) and 5 (ArVr) = CONGRUENT} && {2 (AlVr) and 4 (ArVl) = INCONGRUENT}
        MST_set = resp_sets[1] | resp_sets[5] | resp_sets[2] | resp_sets[4] 
        
        # Modulated
        AUD_modulated = (AUD_set & MST_set) - VIS_set
        VIS_modulated = (VIS_set & MST_set) - AUD_set 

        # Always responding (most)
        ALWAYS_responding = VIS_set & MST_set & AUD_set

        # venn3(subsets=(VIS_set, AUD_set, MST_set), set_labels = ('VIS', 'AUD', 'MST'), set_colors = ('g', 'r', 'purple'))

        # TODO: potentially add congruent / incongruent distinction
        return {'VIS':VIS_set,
                'VIS_modulated':VIS_modulated,
                'VIS_only':VIS_set - (MST_set | AUD_set),
                'AUD':AUD_set,
                'AUD_modulated':AUD_modulated,
                'AUD_only':AUD_set - (MST_set | VIS_set),
                'MST':MST_set,
                'MST_only': MST_set - (AUD_set | VIS_set),
                'ALWAYS_responding' : ALWAYS_responding,
                'TOTAL' : MST_set | AUD_set | VIS_set,
                'diagram_setup' : [(VIS_set, AUD_set, MST_set), ('VIS', 'AUD', 'MST'), ('g', 'r', 'purple')]}
    
    def tt_BY_neuron_group(self, 
                           GROUP:Literal['VIS', 'AUD', 'MST'] = 'VIS',
                           GROUP_type:Literal['modulated', 'modality_specific', 'all'] = 'modulated',
                           BrainRegionIndices : np.ndarray | None = None
                           ) -> list[tuple[ndarray, ndarray]]:
        # combine stimuli VIS trials (6, 7), AUD trials (0, 3), MST congruent (1, 5) MST incongruent trials (2, 4)
        VIS_trials = np.concatenate([self.TTS_BLC_Z[6], self.TTS_BLC_Z[7]]) # 0
        AUD_trials = np.concatenate([self.TTS_BLC_Z[0], self.TTS_BLC_Z[3]]) # 1
        MST_congruent_trials = np.concatenate([self.TTS_BLC_Z[1], self.TTS_BLC_Z[5]]) # 2
        MST_incongruent_trials = np.concatenate([self.TTS_BLC_Z[2], self.TTS_BLC_Z[4]]) # 3
        
        signals = (VIS_trials, AUD_trials, MST_congruent_trials, MST_incongruent_trials)

        # can't use set as index for ndarray
        match GROUP_type:
            case 'modulated':
                # for all trial select the neurons from that group
                # only neurons on intersections
                assert GROUP in {'AUD','VIS'}, 'only visual (VIS) and auditory (AUD) neurons can be modulated by multisensory input, does not make sense for MST neurons'
                indices = np.fromiter(ind := self.NEURON_groups[f'{GROUP}_{GROUP_type}'], int, len(ind))
                
            case 'modality_specific':
                indices = np.fromiter(ind := self.NEURON_groups[f'{GROUP}_only'], int, len(ind))
            
            case 'all':
                indices = np.fromiter(ind := self.NEURON_groups[f'{GROUP}'], int, len(ind))
        
        # incorporate brain region indices if provided
        if BrainRegionIndices is not None:
            indices = np.intersect1d(BrainRegionIndices, indices)

        # to plot single neurons for debugging
        STATS = {i:[] for i in range(8)}
        for trial in range(8):
            for test in ('zeta', 'ttest', 'wilcoxon'):
                sig = self.TTS_Z[trial] if test != 'zeta' else self.TT_zeta[trial]
                _, stat = self.responsive_trial_locked(neurons = sig,
                                                       window = (16, 32),
                                                       criterion = 0.01 / 8,
                                                       method=test)
                STATS[trial].append(stat)

        for index in indices:
            plot_1neuron(all_trials_signal=self.TTS_BLC_Z,
                         single_neuron=index,
                         trial_names=self.tt_names,
                         time = self.time,
                         STATS=STATS)
            
        neurons_to_study = [sig[:,:, indices] for sig in signals]
        return [(avr, sem) for _, avr, sem in (calc_avrg_trace(trace, self.time, PLOT = False)
                for trace in neurons_to_study)], len(indices)


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
        Axis.axvspan(0,1,alpha = 0.05, color = 'navajowhite') # HARDCODED TODO: change based on trial duration
    
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
        plt.tight_layout()
        plt.show()
        plt.close()

def plot_1neuron(all_trials_signal:list[np.ndarray], 
                 single_neuron:int,
                 trial_names:list[str], time:ndarray,
                 STATS : list[object, object, np.ndarray] | None = None):
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    plt.close()
    fig, axs = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', 
                            figsize = (4 * 5, 2*4))
    # for each trial type
    for itt, ttsig in enumerate(all_trials_signal.values()):
        trial_STATS = STATS[itt]
        zetalabel = f'ZETA: ({trial_STATS[0][single_neuron,:].round(2)})\n'
        ttestlabel = f'TTEST: {(float(round(trial_STATS[1].pvalue[single_neuron],2)), float(round(trial_STATS[1].statistic[single_neuron], 3)))}\n'
        wilcoxonlabel = f'WILCOXON: {(float(round(trial_STATS[2].pvalue[single_neuron], 2)), float(round(trial_STATS[2].statistic[single_neuron], 3)))}\n'

        axs[tt_grid[itt]].axvspan(0,1,alpha = 0.1, color = 'navajowhite')
        axs[tt_grid[itt]].hlines(y = 0, xmin = 0, xmax = 2, color = 'r')
        neuron_all_trials = ttsig[:,:,single_neuron] 
        # plot all raw traces for each trial
        for i_trial in range(neuron_all_trials.shape[0]):
            axs[tt_grid[itt]].plot(time, neuron_all_trials[i_trial, :], alpha = 0.1)
        # as well as the average trace across trials
        axs[tt_grid[itt]].plot(time, neuron_all_trials.mean(axis = 0), color = 'blue', label = zetalabel + ttestlabel + wilcoxonlabel)
        axs[tt_grid[itt]].set_title(trial_names[itt])
        axs[tt_grid[itt]].legend(loc = 3, fontsize = 8)
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

def snake_plot(all_neuron_averages:ndarray, 
               stats:_result_classes.TtestResult | np.ndarray,
               trial_window_frames:tuple[int, int], time:ndarray,
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
        sorted_by_response_onset = argsort(argmax(average_trace_per_neuron[:,trial_window_frames[0]:trial_window_frames[1]], axis = 1))
        heatmap_neurons = average_trace_per_neuron[sorted_by_response_onset]

    # significance sorting
    elif MODE == 'signif':
        sorted_by_significance = argsort(stats.pvalue) if not isinstance(stats, np.ndarray) else argsort(stats[:,0])
        heatmap_neurons = average_trace_per_neuron[sorted_by_significance]

    sns.heatmap(heatmap_neurons, vmin = heatmap_range[0], vmax=heatmap_range[1],
                xticklabels = False, ax = Axis, cbar = colorbar)
    
    Axis.vlines(trial_window_frames, ymin = 0, ymax=heatmap_neurons.shape[0])
    Axis.set_xticks(timestoplot := [0, trial_window_frames[0], trial_window_frames[1], heatmap_neurons.shape[1]-1], 
                        time.round()[timestoplot])
    if SHOW:
        plt.show()


###--------------------------------SPECIFIC analyses----------------------------------------------------
# overall, not split into neuron groups based on Venn diagram
def TT_ANALYSIS(tt_grid:dict[int:tuple[int, int]],
                SNAKE_MODE:Literal['onset', 'signif'] = 'onset'):
    ''' 
    need to specify how you want to organize the conditions in the grid plot
    '''
    avs : list[AUDVIS] = load_in_data() # -> av1, av2, av3, av4
    TT_RESULTS = defaultdict(dict)
    
    fig1, axs1 = plt.subplots(nrows = 2, ncols = 4, sharey = 'row', sharex='col', figsize = (4 * 5, 2*4))
    linestyles = ('-', ':')
    colors = ('royalblue', 'goldenrod')
    fig2, axs2 = plt.subplots(nrows = 2*2, ncols = 4*2, sharex='col', figsize = (4 * 5, 2*4))
    snake_grid = build_snake_grid(tt_grid)
    
    # get cbar range & calculate results
    analyses = [Analyze(audvis) for audvis in avs]

    # PLOTTING LOOP
    for i, (av, ANALYS) in enumerate(zip(avs, analyses)):
        print(f'Starting analysis of {av.NAME}')
        trial_names = ANALYS.tt_names
        average_traces, sem_traces, responsive_neurs = ANALYS.TT_RES

        for i_tt, tt in enumerate(trial_names):
            gridloc = tt_grid[i_tt]
            snakeloc = snake_grid[i_tt, i]
            colorbar = True #if snakeloc[1] == 7 else False
            # Snake PLOT
            snake_plot(all_neuron_averages = ANALYS.TTS_BLC_Z[i_tt], stats=ANALYS.TT_STATS[i_tt],
                    #    heatmap_range = cbar_range,
                       trial_window_frames = ANALYS.TRIAL_FRAMES, Axis = axs2[snakeloc],
                       colorbar=colorbar, title=f'{av.NAME}_{tt}', time = ANALYS.time,
                       MODE=SNAKE_MODE)
 
            # Average traces plot
            plot_avrg_trace(ANALYS.time, avrg = average_traces[i_tt], SEM = sem_traces[i_tt],
                            Axis = axs1[gridloc], title = tt, label = f'{av.NAME} ({len(responsive_neurs[i_tt])} / {len(list(av.neurons))})', 
                            vspan = (i == 0), col = colors[i >= 2], lnstl=linestyles[i%2])

            axs1[gridloc].legend(loc = 2)
            if gridloc[1] == 0:
                axs1[gridloc].set_ylabel('z(∆F/F)')
            if gridloc[0] == 1:
                axs1[gridloc].set_xlabel('Time (s)')
            
            if snakeloc[0] == 3:
                axs2[snakeloc].set_xlabel('Time (s)')
            
    fig2.tight_layout()
    fig2.savefig(f'TT_snake_{SNAKE_MODE}.png', dpi = 1000)

    fig1.tight_layout()
    fig1.savefig('TT_average_res.png', dpi = 1000)

# overall responsive neuron groups, not by brain region
def neuron_typesVENN_analysis():
    avs : AUDVIS = load_in_data() # -> av1, av2, av3, av4
    # venn diagram figure
    fig3, axs3 = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8))

    for i, (av, ax) in enumerate(zip(avs, axs3.flatten())):
        ANALYSIS : Analyze = Analyze(av)
        neuronGROUPS = ANALYSIS.NEURON_groups
        plot_args = neuronGROUPS['diagram_setup']
        total_responsive = len(neuronGROUPS['TOTAL'])
        ax.set_title(f'{av.NAME} ({total_responsive}/{av.rois.shape[0]})')
        venn3(*plot_args, ax = ax,
              # percentage of responsive neurons
              subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total_responsive):1.0%}" + ")")
    
    fig3.tight_layout()
    fig3.savefig('VennDiagram.png', dpi = 1000)
    plt.show()
    plt.close()

# TODO: add snake plot
# snake plot and average plot for different neuron types (based on venn diagram)
def NEURON_TYPES_TT_ANALYSIS(GROUP_type:Literal['modulated', 
                                                'modality_specific', 
                                                'all'] = 'modulated'):
    # Load in all data and perform necessary calculations
    AVS : list[AUDVIS] = load_in_data() # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]

    # prepare everything for plotting
    if GROUP_type == 'modulated':
        tsnrows = 2
        GROUPS = ('VIS', 'AUD')
    else:
        tsnrows = 3
        GROUPS = ('VIS', 'AUD', 'MST')

    linestyles = ('-', '--', '-', '--') # pre -, post :
    colors = {'VIS':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
              'AUD':('darkred', 'darkred', 'coral', 'coral'),
              'MST':('darkmagenta','darkmagenta', 'orchid', 'orchid')}

    tsncols = 4 # 4 combined trial types
    snake_rows, snake_cols = 2, 2

    # prepare timeseries figure and axes
    ts_fig, ts_ax = plt.subplots(nrows=tsnrows, ncols=tsncols, 
                                 sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * 3))
    trials = ['VT', 'AT', 'MS+', 'MS-']
    for ig, group in enumerate(GROUPS):
        for icond, (AV, AN) in enumerate(zip(AVS, ANS)):
            TT_info, group_size = AN.tt_BY_neuron_group(group, GROUP_type)
            for itt, (avr, sem) in enumerate(TT_info):
                plot_avrg_trace(time = AN.time, avrg=avr, SEM = sem, Axis=ts_ax[ig, itt],
                                title = trials[itt] if ig == 0 else False,
                                label = f'{AV.NAME} ({group_size})' if itt == len(list(TT_info)) - 1 else None, 
                                col = colors[group][icond], lnstl=linestyles[icond])

                if icond == len(AVS) - 1:
                    if ig == len(GROUPS) - 1:
                        ts_ax[ig, itt].set_xlabel('Time (s)')

                    if itt == 0:
                        ts_ax[ig, itt].set_ylabel('z(∆F/F)')

                    if itt == len(list(TT_info)) - 1:
                        twin = ts_ax[ig, itt].twinx()
                        twin.set_ylabel(GROUPS[ig], rotation = 270, 
                                                    va = 'bottom', 
                                                    color = colors[group][0],
                                                    fontsize = 20)
                        twin.set_yticks([])

    ts_fig.tight_layout(rect = [0,0,0.85,1])
    ts_fig.legend(loc = 'outside center right')
    ts_fig.savefig(f'Neuron_type({GROUP_type})_average_res.png', dpi = 1000)



### Main block that runs the file as a script
if __name__ == '__main__':
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    
    # Neuron types analysis (venn diagrams)
    neuron_typesVENN_analysis()
    # NEURON_TYPES_TT_ANALYSIS('modulated')
    # NEURON_TYPES_TT_ANALYSIS('modality_specific')
    # NEURON_TYPES_TT_ANALYSIS('all')

    # Trial type analysis (average & snake plot) - all neurons, not taking into account neuron types groups
    # TT_ANALYSIS(tt_grid=tt_grid, SNAKE_MODE='onset')
    