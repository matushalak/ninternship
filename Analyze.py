#@matushalak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from AUDVIS import AUDVIS, Behavior, load_in_data
from analysis_utils import calc_avrg_trace, build_snake_grid, snake_plot, plot_avrg_trace
import multisens_calcs as MScalc
from Responsive import responsive_zeta
from utils import show_me

from matplotlib import artist
from matplotlib_venn import venn3
from numpy import ndarray, arange, where, unique
from scipy.stats import wilcoxon, ttest_rel
from collections import defaultdict
from typing import Literal

# TODO: multisensory integration (for modulated VIS and AUD neurons), response selectivity index
# TODO: congruent vs incongruent -> FOR MODULATED neurons & for MST_only neurons

# ----------- Main ANALYSIS class -----------
class Analyze:
    ''' 
    class to perform analyses on neurons from one group and one condition
    '''
    def __init__(self, av:AUDVIS, signal : Literal['dF/F0', 'spike_prob'] = 'dF/F0', 
                 storage_folder:str = 'results', include_zeta : bool = True):
        self.storage_folder = storage_folder
        
        # self.SIGNAL dF/F OR spike_prob
        self.signal_type = signal
        assert self.signal_type in ('dF/F0', 'spike_prob'
                                    ), 'signal must be either "{}" or "{}"'.format('dF/F0', 'spike_prob')
        
        # calculate time for plotting
        pre = round(-1/av.SF * av.trial_frames[0])
        post = (1/av.SF * av.trial_frames[1])
        self.time = arange(pre, post, 
                           1/av.SF)
        # trial is between these frames
        self.TRIAL_FRAMES = av.TRIAL

        # SAMPLING FREQUENCY
        self.SF : float = av.SF

        # Trial names
        self.tt_names = list(av.str_to_int_trials_map)

        if include_zeta:
            # Zeta test results dF/F0 or spike_prob
            self.TT_zeta = responsive_zeta(SPECIFIEDcond = av.NAME, signal_to_use = 'dF/F0') # NOTE: zeta on regressed-out dF/F0 seems to be the best

        # Get Fluorescence response statistics
        self.FLUORO_RESP: np.ndarray = self.fluorescence_response(
            signal = av.separate_signal_by_trial_types(
                av.baseline_correct_signal(av.signal_CORR, baseline_frames=self.TRIAL_FRAMES[0])),
            window = self.TRIAL_FRAMES)
        
        # Analyze by trial type
        self.TT_RES, self.TT_STATS, self.byTTS_blc, self.byTTS, self.NEURON_groups = self.tt_average(av, signal_to_use = self.signal_type)
        _, _, self.CASCADE, _, _ = self.tt_average(av, signal_to_use = 'spike_prob') # for debugging


    # Analyze average response to trial type
    def tt_average(self, av:AUDVIS,
                   method:str = 'zeta',
                   criterion:float = 0.05, # p-value threshold .05 or .01 (bonferroni corrected - consider different correction)
                   signal_to_use: Literal['dF/F0', 'spike_prob'] = 'dF/F0',
                   zsignal: bool = True
                   ) -> tuple[list[ndarray], 
                              list[ndarray], 
                              list[ndarray]]:
        match signal_to_use, zsignal:
            case 'dF/F0', False:
                signal = av.signal_CORR
            case 'dF/F0', True:
                signal = av.zsig_CORR
            case 'spike_prob', _:
                signal = av.CASCADE_CORR * self.SF # to convert to est. IFR

        SIG = signal 
        # 1. Baseline correct z-scored signal; residual signal without running speed OR whisker movement
        BLC_SIG = av.baseline_correct_signal(signal=SIG)

        # 2. Separate into Trial=types
        by_tts : dict = av.separate_signal_by_trial_types(SIG) # z-scored signal
        by_tts_blc : dict = av.separate_signal_by_trial_types(BLC_SIG) # baseline-corrected z-scored signal

        # get significantly responding neurons for each trial type
        # 3. collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []

        TEST_RESULTS = []
        # TODO: consider different corrections
        corrected_criterion = criterion / len(list(by_tts)) 
        # A single spike in the predictions will have an amplitude of 0.266 and a width (FWHM) of 0.24 seconds.
        zeta_crit = (corrected_criterion, 0.05) if signal_to_use == 'spike_prob' else (corrected_criterion, 0.3)
        # iterate through trial types
        for tt in by_tts.keys():
            # get indices responsive to that trial type
            responsive_indices, test_res = self.responsive_trial_locked(neurons = by_tts[tt] if method != 'zeta' else self.TT_zeta[tt],
                                                                        # potentially separate analysis into ON-responsive and OFF-responsive
                                                                        window = (av.TRIAL[0], av.TRIAL[1] #+ (av.TRIAL[0]//2) # add .5 seconds after trial for offset responses
                                                                                  ), 
                                                                        criterion = (corrected_criterion if method != 'zeta' else zeta_crit
                                                                                     ) if method != 'zscore' else criterion, # multiple comparison correction
                                                                        method = method,
                                                                        blc_data = by_tts_blc[tt] if method == 'zeta' else None)
            if method != 'zscore':
                TEST_RESULTS.append(test_res)
            # print(by_tts[tt].shape, len(self.TT_zeta[tt])) # good debugging check
            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices)
            # save baseline corrected signal
            responsive_neurons = by_tts_blc[tt][:,:,responsive_indices]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                  PLOT=False) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs # ndarray at each tt idex gives neurons responsive to that tt
                ), TEST_RESULTS, by_tts_blc, by_tts, self.neuron_groups(responsive = responsive_neurs)


    def responsive_trial_locked(self, neurons:ndarray | list, window:tuple[int, int],
                                criterion:float|tuple[float,float], method:str, blc_data : ndarray | None = None)-> tuple[ndarray, ndarray | None]:
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
                bls, trls = neurons[:,:window[0],:].mean(axis = 1), neurons[:,window[0]:window[1],:].mean(axis = 1)
                TTEST = ttest_rel(bls, trls, alternative = 'two-sided')
                return where(TTEST.pvalue < criterion)[0], TTEST
            
            # wilcoxon signed-rank test
            case ('wilcoxon', criterion, (n_trials, n_times, n_nrns)):
                # NOTE: average over time bins (axis = 0)[16 vs 16 values] vs average over baseline vs stimulus window in each trial (axis = 1)(before)[90 vs 90 values]
                bls, trls = np.mean(neurons[:,:window[0],:],axis = 1), np.mean(neurons[:,window[0]:window[1],:], axis = 1)
                WCOX = wilcoxon(bls, trls, alternative='two-sided')
                return where(WCOX.pvalue < criterion)[0], WCOX
            
            # already have zeta significances and Zeta values (n_nrns, 2[zeta_p, zeta_score])
            # consider this for threshold: 
            # '''The fluorescence response of a neuron in a given trial was defined as 
            # the average F/F0 over all imaging frames during the 3 s stimulus period.''' Meijer (2017)
            case ('zeta', (p_criterion, amp_criterion), (n_nrns, stats)):
                # p-value below threshold
                assert blc_data is not None, 'Need baseline-corrected data to select based on amplitude threshold'
                responsive =  where((neurons[:,0] < p_criterion) & (
                    (np.mean(blc_data, axis = 0)[window[0]:window[1],:].max(axis = 0) >= amp_criterion) | 
                    # include inhibited
                    (np.mean(blc_data, axis = 0)[window[0]:window[1],:].min(axis = 0) <= -amp_criterion)))[0]
                return responsive, neurons


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
        # VIS: union between TT 6 (Vl) and 7 (Vr)
        VIS_set = resp_sets[6] | resp_sets[7] 
        # AUD: union between TT 0 (Al) and 3 (Ar)
        AUD_set = resp_sets[0] | resp_sets[3] 
        # MST: union between TT {1 (AlVl) and 5 (ArVr) = CONGRUENT} && {2 (AlVr) and 4 (ArVl) = INCONGRUENT}
        MST_set = resp_sets[1] | resp_sets[5] | resp_sets[2] | resp_sets[4] 
        
        # Always responding (most)
        ALWAYS_responding = VIS_set & MST_set & AUD_set
        
        # Modulated
        VIS_modulated = (VIS_set & MST_set) - ALWAYS_responding 
        AUD_modulated = (AUD_set & MST_set) - ALWAYS_responding

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


    # TODO> potentially show only response to preferred direction, now averages preferred and nonpreferred
    def tt_BY_neuron_group(self, 
                           GROUP:Literal['VIS', 'AUD', 'MST'] = 'VIS',
                           GROUP_type:Literal['modulated', 'modality_specific', 'all'] = 'modulated',
                           BrainRegionIndices : np.ndarray | None = None,
                           SIGNALS_TO_USE: dict[int : ndarray] | None = None
                           ) -> list[tuple[ndarray, ndarray]]:
        TT_blc_signal = self.byTTS_blc if SIGNALS_TO_USE is None else SIGNALS_TO_USE
        assert isinstance(TT_blc_signal, dict) & all(isinstance(TT_blc_signal[tt], np.ndarray) for tt in TT_blc_signal.keys()
                                                     ), 'Provided SIGNALS to use is not a dictionary or doesnt contain np.ndarrays with signal'
        # combine stimuli VIS trials (6, 7), AUD trials (0, 3), MST congruent (1, 5) MST incongruent trials (2, 4)
        VIS_trials = np.concatenate([TT_blc_signal[6], TT_blc_signal[7]]) # 0
        AUD_trials = np.concatenate([TT_blc_signal[0], TT_blc_signal[3]]) # 1
        MST_congruent_trials = np.concatenate([TT_blc_signal[1], TT_blc_signal[5]]) # 2
        MST_incongruent_trials = np.concatenate([TT_blc_signal[2], TT_blc_signal[4]]) # 3
        
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

        # DEBUGGING only (or example neurons potentially) to plot single neurons for debugging
        # print(GROUP, GROUP_type, indices.size)
        # STATS = {i:[] for i in range(8)}
        # for trial in range(8):
        #     for test in ('zeta', 'ttest', 'wilcoxon'):
        #         sig = self.byTTS[trial] if test != 'zeta' else self.TT_zeta[trial]
        #         _, stat = self.responsive_trial_locked(neurons = sig,
        #                                                window = self.TRIAL_FRAMES,
        #                                                criterion = 0.01 / 8 if test != 'zeta' else (0.01 / 8, .3),
        #                                                method=test,
        #                                                blc_data= None if test != 'zeta' else TT_blc_signal[trial])
        #         STATS[trial].append(stat)

        # IFR = [self.CASCADE[tt] for tt in self.CASCADE]
        # for index in indices:
        #     plot_1neuron(all_trials_signal=TT_blc_signal,
        #                  single_neuron=index,
        #                  trial_names=self.tt_names,
        #                  time = self.time,
        #                  CASCADE=IFR,
        #                  STATS=STATS)
        # breakpoint()
        neurons_to_study = [sig[:,:, indices] for sig in signals]
        return [(avr, sem) for _, avr, sem in (calc_avrg_trace(trace, self.time, PLOT = False)
                for trace in neurons_to_study)], len(indices)

    @staticmethod
    def fluorescence_response (signal: np.ndarray | dict[int:np.ndarray],
                               window: tuple[int, int] = (16,32)) -> np.ndarray:
        '''
        Takes
        ---------
            1) (trial, times, neurons) array and 
            
            2) {trial_type : (trial, times, neurons) array} dictionary
        
        Returns
        ---------
            1) (neurons, stats) array with mean and std (dim 1) 
            of fluorescence response (F) to a given trial-type for each neuron (dim 0)

            2) returns a (neurons, stats, trial_types) array with mean and std (dim 1) 
            of fluorescence response (F) to each trial-type (dim 2) for each neuron (dim 0)
        '''
        assert isinstance(signal, dict
                          ) or isinstance(signal, np.ndarray
                                          ), 'Signal must either be a dictionary with signal arrays for each trial type OR just a signal array for one trial type'
        if isinstance(signal, dict):
            res = np.empty((signal[0].shape[-1], 2, len(signal)))
            tt_sigs = [signal[tt] for tt in signal.keys()]
        elif isinstance(signal, dict):
            res = np.empty((signal.shape[-1], 2, 1))
            tt_sigs = [signal]

        assert all(len(sig.shape) == 3 for sig in tt_sigs), 'Signal arrays must be 3D - (ntrials, ntimes, nneurons)!'
        for itt, sig_array in enumerate(tt_sigs):
            # Fluorescence response adapted from (Meijer et al., 2017)
            Fmean  = sig_array[:,window[0]:window[1],:].mean(axis = 1)
            F = np.empty_like(Fmean)
            Fmax  = sig_array[:,window[0]:window[1],:].max(axis = 1)
            Fmin  = sig_array[:,window[0]:window[1],:].min(axis = 1)
            max_mask = where(Fmean > 0)
            min_mask = where(Fmean < 0)
            
            F[max_mask] = Fmax[max_mask]
            F[min_mask] = Fmin[min_mask]
            # F = Fmean

            meanF = F.mean(axis = 0) # mean fluorescence response over trials
            stdF = F.std(axis = 0) # std of fluorescence response over trials
            res[:, 0, itt] = meanF
            res[:, 1, itt] = stdF
        
        return np.squeeze(res) # removes trailing dimension in case want output only for 1 trial type


###--------------------------------SPECIFIC analyses with plots-----------------------------------------------
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
            snake_plot(all_neuron_averages = ANALYS.byTTS_blc[i_tt], stats=ANALYS.TT_STATS[i_tt],
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
                                                'all'] = 'modulated',
                             add_CASCADE: bool = False, 
                             pre_post: Literal['pre', 'post', 'both'] = 'pre'):
    # Load in all data and perform necessary initial calculations
    AVS : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]

    # prepare everything for plotting
    if GROUP_type == 'modulated':
        tsnrows = 2
        GROUPS = ('VIS', 'AUD')
    else:
        tsnrows = 3
        GROUPS = ('VIS', 'AUD', 'MST')

    match pre_post:
        case 'both':
            linestyles = ('-', '--', '-', '--') # pre -, post --
            colors = {'VIS':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
                    'AUD':('darkred', 'darkred', 'coral', 'coral'),
                    'MST':('darkmagenta','darkmagenta', 'orchid', 'orchid')}
        case 'pre':
            linestyles = ('-', '-') # pre -, 
            colors = {'VIS':('darkgreen','mediumseagreen'),
                    'AUD':('darkred', 'coral'),
                    'MST':('darkmagenta', 'orchid')}
        case 'post':
            linestyles = ('--', '--') # pre -, 
            colors = {'VIS':('darkgreen','mediumseagreen'),
                    'AUD':('darkred', 'coral'),
                    'MST':('darkmagenta', 'orchid')}

    tsncols = 4 # 4 combined trial types
    # snake_rows, snake_cols = 2, 2

    # prepare timeseries figure and axes
    ts_fig, ts_ax = plt.subplots(nrows=tsnrows, ncols=tsncols, 
                                 sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * 3))
    trials = ['VT', 'AT', 'MS+', 'MS-']
    for ig, group in enumerate(GROUPS):
        for icond, (AV, AN) in enumerate(zip(AVS, ANS)):
            TT_info, group_size = AN.tt_BY_neuron_group(group, GROUP_type)
            # also plot cascade traces
            if add_CASCADE:
                CASCADE_TT_info, _ = AN.tt_BY_neuron_group(group, GROUP_type, SIGNALS_TO_USE= AN.CASCADE)
            for itt, (avr, sem) in enumerate(TT_info):
                plot_avrg_trace(time = AN.time, avrg=avr, SEM = sem, Axis=ts_ax[ig, itt],
                                title = trials[itt] if ig == 0 else False,
                                label = f'{AV.NAME} ({group_size})' if itt == len(list(TT_info)) - 1 else None, 
                                col = colors[group][icond], lnstl=linestyles[icond])
                if add_CASCADE:
                    plot_avrg_trace(time = AN.time, avrg=CASCADE_TT_info[itt][0], SEM = None, Axis=ts_ax[ig, itt],
                                    label = 'Est. FR' if itt == len(list(TT_info)) - 1 else None, 
                                    col = colors[group][icond], lnstl=linestyles[icond], alph=.5)
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
    ts_fig.savefig(f'Neuron_type({GROUP_type})_average_res({pre_post}).png', dpi = 1000)

# ---------------------- MULTISENSORY ENHANCEMENT ANALYSIS -----------------------------
def MI(pre_post: Literal['pre', 'post', 'both'] = 'pre'):
    # Load in all data and perform necessary initial calculations
    AVS : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]

    # collect DSIs
    dsis_vis = []
    dsis_aud = []
    # collect group_names
    g_names = []
    # collect RCIs
    rcis_cong_vis = []
    rcis_incong_vis = []

    rcis_cong_aud = []
    rcis_incong_aud = []

    for i, (Av, Analys) in enumerate(zip(AVS, ANS)):
        #1) get direction selectivity info
        pref_stats, orth_stats, congruent_stats, incongruent_stats = np.split(MScalc.direction_selectivity(Analys.FLUORO_RESP), 
                                                                              indices_or_sections=4, 
                                                                              axis = 2)
        #2) get direction selectivity index for visual and auditory separately
        DSI_vis, DSI_aud = np.split(MScalc.DSI(pref_stats, orth_stats),
                                    indices_or_sections=2,
                                    axis = 2)
        
        g_names += [Av.NAME] * DSI_vis.size
        dsis_vis += [*DSI_vis.squeeze()]
        dsis_aud += [*DSI_aud.squeeze()]
        #3) Get response change index for visual and auditory separately
        # for congruent MST
        RCI_vis_congruent, RCI_aud_congruent = np.split(MScalc.RCI(congruent_stats, pref_stats),
                                                        indices_or_sections=2,
                                                        axis = 2)
        rcis_cong_vis += [*RCI_vis_congruent.squeeze()]
        rcis_cong_aud += [*RCI_aud_congruent.squeeze()]
        # for incongruent MST
        RCI_vis_incongruent, RCI_aud_incongruent = np.split(MScalc.RCI(incongruent_stats, pref_stats),
                                                            indices_or_sections=2,
                                                            axis = 2)
        rcis_incong_vis += [*RCI_vis_incongruent.squeeze()]
        rcis_incong_aud += [*RCI_aud_incongruent.squeeze()]

    MIdata : pd.DataFrame = pd.DataFrame({'DSI (VIS)' : dsis_vis, 'DSI (AUD)': dsis_aud,
                                          'Group':g_names,
                                          'RCI (VIS congruent)': rcis_cong_vis, 'RCI (VIS incongruent)' : rcis_incong_vis,
                                          'RCI (AUD congruent)': rcis_cong_aud, 'RCI (AUD incongruent)' : rcis_incong_aud})
    # make plots!
    # Direction Selectivity Index Plot
    dsi_plot = MScalc.scatter_hist_reg_join(MIdata, NAME='DSI_plot', X_VAR='DSI (VIS)', Y_VAR='DSI (AUD)', HUE_VAR='Group',
                                            kde = True, reg = True)
    print('DSI plot done!')

    # Response change index plots
    # VIS
    rci_plotV = MScalc.scatter_hist_reg_join(MIdata, NAME='RCI_vis', 
                                             X_VAR='RCI (VIS congruent)', Y_VAR='RCI (VIS incongruent)', HUE_VAR='Group',
                                             square=True, reg= True, kde=True)
    print('VIS RCI plot done!')
    # AUD
    rci_plotA = MScalc.scatter_hist_reg_join(MIdata, NAME='RCI_aud', 
                                             X_VAR='RCI (AUD congruent)', Y_VAR='RCI (AUD incongruent)', HUE_VAR='Group',
                                             square=True, reg= True, kde=True)
    print('AUD RCI plot done!')
    # breakpoint()

### ---------- Main block that runs the file as a script
if __name__ == '__main__':
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    
    # Neuron types analysis (venn diagrams)
    # neuron_typesVENN_analysis()
    # NEURON_TYPES_TT_ANALYSIS('modulated', add_CASCADE=True, pre_post='pre')
    # NEURON_TYPES_TT_ANALYSIS('modality_specific', add_CASCADE=True, pre_post='pre')
    # NEURON_TYPES_TT_ANALYSIS('all', add_CASCADE=True, pre_post='pre')

    # Trial type analysis (average & snake plot) - all neurons, not taking into account neuron types groups
    # NOTE: doesnt work well on cascade
    # TT_ANALYSIS(tt_grid=tt_grid, SNAKE_MODE='onset')

    #
    MI('pre')
    