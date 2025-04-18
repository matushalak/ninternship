#@matushalak
import numpy as np
import matplotlib.pyplot as plt

from AUDVIS import AUDVIS, Behavior, load_in_data
from analysis_utils import calc_avrg_trace, build_snake_grid, snake_plot, plot_avrg_trace, plot_1neuron
from Responsive import responsive_zeta

from matplotlib_venn import venn3
from numpy import ndarray, arange, where, unique
from scipy.stats import wilcoxon, ttest_rel
from functools import reduce
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
                 storage_folder:str = 'results', include_zeta : bool = False):
        self.storage_folder = storage_folder
        
        # self.SIGNAL dF/F OR spike_prob
        self.signal_type = signal
        assert self.signal_type in ('dF/F0', 'spike_prob'
                                    ), 'signal must be either "dF/F0" or "spike_prob"'
        
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

        # Session neurons to find back a neuron from index in big matrix
        self.sess_neur = av.session_neurons
        self.session_names = [av.sessions[sess]['session'] for sess in av.sessions.keys()] # gives name for each range in session_neurons
        
        if include_zeta:
            # Zeta test results dF/F0 or spike_prob
            self.TT_zeta = responsive_zeta(SPECIFIEDcond = av.NAME, signal_to_use = 'dF/F0',
                                           RegressOUT_behavior=False)

        # Get Fluorescence response statistics
        self.FLUORO_RESP: np.ndarray = self.fluorescence_response(
            signal = av.separate_signal_by_trial_types(
                av.baseline_correct_signal(av.signal_CORR, 
                                           baseline_frames=self.TRIAL_FRAMES[0])
                                           ),
            window = self.TRIAL_FRAMES, 
            method='peak')
        
        # Analyze by trial type
        self.TT_RES, self.TT_STATS, self.byTTS_blc, self.byTTS, self.NEURON_groups = self.tt_average(av, signal_to_use = self.signal_type)
        _, _, self.CASCADE, _, _ = self.tt_average(av, signal_to_use = 'spike_prob') # for debugging


    # Analyze average response to trial type
    def tt_average(self, av:AUDVIS,
                   method:str = 'ttest',
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
        trial_types = sorted(list(by_tts))

        # get significantly responding neurons for each trial type
        # 3. collect average and sem of traces of responsive neurons for each condition
        average_traces, sem_traces, responsive_neurs = [], [], []

        TEST_RESULTS = []
        # TODO: consider different corrections
        corrected_criterion = criterion / len(trial_types) 
        # A single spike in the predictions will have an amplitude of 0.266 and a width (FWHM) of 0.24 seconds.
        # NOTE: using Cohen's d effect size threshold
        stat_crit = (corrected_criterion, 0.45)
        # iterate through trial types
        for tt in trial_types:
            # get indices responsive to that trial type
            # TODO: separate analysis for inhibited
            responsive_indices_EXC, responsive_indices_INH, test_res = self.responsive_trial_locked(neurons = by_tts[tt] if method != 'zeta' else self.TT_zeta[tt],
                                                                                                    # potentially separate analysis into ON-responsive and OFF-responsive
                                                                                                    window = (av.TRIAL[0], av.TRIAL[1] #+ (av.TRIAL[0]//2) # add .5 seconds after trial for offset responses
                                                                                                            ), 
                                                                                                    trial_ID=tt,
                                                                                                    criterion = stat_crit if method != 'zscore' else criterion, 
                                                                                                    method = method)
            if method != 'zscore':
                TEST_RESULTS.append(test_res)
            # print(by_tts[tt].shape, len(self.TT_zeta[tt])) # good debugging check
            # indices, TO LATER IDENTIFY WHICH NEURON IS WHAT NATURE
            responsive_neurs.append(responsive_indices_EXC)
            # save baseline corrected signal
            responsive_neurons = by_tts_blc[tt][:,:,responsive_indices_EXC]

            _, tt_avrg, tt_sem =  calc_avrg_trace(trace=responsive_neurons, time = self.time, 
                                                  PLOT=False) # for testing set to True to see
            average_traces.append(tt_avrg)
            sem_traces.append(tt_sem)
        
        return (average_traces, sem_traces, 
                responsive_neurs # ndarray at each tt idex gives neurons responsive to that tt
                ), TEST_RESULTS, by_tts_blc, by_tts, self.neuron_groups(responsive = responsive_neurs)

    def responsive_trial_locked(self, neurons:ndarray | list, window:tuple[int, int], 
                                trial_ID: int,
                                criterion:float|tuple[float,float], method:str
                                )-> tuple[ndarray, ndarray | None]:
        '''
        accepts (trials, times, neurons) type data structure for neurons within session or across sessions
                OR (times, neurons) for already aggregated average traces
        
        returns indices of responsive neurons (TO ONE CONDITION)
        '''
        if isinstance(neurons, list):
            neurons = np.array(neurons)
        match (method, criterion, neurons.shape):
            # simply checking average traces of neurons and screening for neurons > criterion (can also look at >= abs(criterion) with .__abs__()
            case ('zscore', criterion, (n_trials, n_times, n_nrns)):
                trial_averaged_neurons = np.nanmean(neurons, axis = 0) # -> (times, neurons) shape
                return unique(where(trial_averaged_neurons[window[0]:window[1],:] >= criterion)[1]), None, None
            
            # do t-test on the mean values during trial
            # TODO: try on fluorescence response + effect size (FR is already baseline corrected)
            # NOTE: Cohen's d = DiffF := (F_trial - F_baseline) / SD(DiffF)
            case ('ttest', (p_criterion, amp_criterion), (n_trials, n_times, n_nrns)): 
                bls, trls = np.nanmean(neurons[:,:window[0],:], axis = 1), np.nanmean(neurons[:,window[0]:window[1],:], axis = 1)
                TTEST = ttest_rel(trls, bls, alternative = 'two-sided', nan_policy='omit')
                # Amplitude theshold using Cohen's d effect size of fluorescence responses
                EXC_thresh = self.FLUORO_RESP[:,2,trial_ID] > amp_criterion
                # include inhibited Cohen's d effect size of fluorescence responses
                INH_thresh = self.FLUORO_RESP[:,2,trial_ID] < -amp_criterion
                return where((TTEST.pvalue < p_criterion) & (EXC_thresh))[0], where((TTEST.pvalue < p_criterion) & (INH_thresh))[0], TTEST
            
            # wilcoxon signed-rank test
            case ('wilcoxon', (p_criterion, amp_criterion), (n_trials, n_times, n_nrns),):
                # NOTE: average over time bins (axis = 0)[16 vs 16 values] vs average over baseline vs stimulus window in each trial (axis = 1)(before)[90 vs 90 values]
                bls, trls = np.nanmean(neurons[:,:window[0],:],axis = 1), np.nanmean(neurons[:,window[0]:window[1],:], axis = 1)
                WCOX = wilcoxon(trls, bls, alternative='two-sided', nan_policy = 'omit')
                # Amplitude theshold
                EXC_thresh = self.FLUORO_RESP[:,2,trial_ID] > amp_criterion
                # include inhibited
                INH_thresh = self.FLUORO_RESP[:,2,trial_ID] < -amp_criterion
                
                return where((WCOX.pvalue < p_criterion) & (EXC_thresh))[0], where((WCOX.pvalue < p_criterion) & (INH_thresh))[0], WCOX
            
            # already have zeta significances and Zeta values (n_nrns, 2[zeta_p, zeta_score])
            case ('zeta', (p_criterion, amp_criterion), (n_nrns, stats)):
                # Amplitude theshold
                EXC_thresh = self.FLUORO_RESP[:,2,trial_ID] > amp_criterion
                # include inhibited
                INH_thresh = self.FLUORO_RESP[:,2,trial_ID] < -amp_criterion
                
                responsiveEXC =  where((neurons[:,0] < p_criterion) & (EXC_thresh))[0] 
                responsiveINH =  where((neurons[:,0] < p_criterion) & (INH_thresh))[0]

                return responsiveEXC, responsiveINH, neurons


    @staticmethod
    def neuron_groups(responsive:list[ndarray]) -> dict[str : np.ndarray]:
        '''
        Input:
            for each trial type, gives neurons significantly responsive to it
        Output:
            returns VIS neurons, AUD neurons and MST neurons for further Venn diagram analysis
        '''            
        # First define the big sets as all neurons that showed response to ONE of the AUD / VIS / MST trials
        # VIS: union between TT 6 (Vl) and 7 (Vr)
        VIS_set = np.union1d(responsive[6], responsive[7])
        # AUD: union between TT 0 (Al) and 3 (Ar)
        AUD_set = np.union1d(responsive[0], responsive[3])
        # MST: union between TT {1 (AlVl) and 5 (ArVr) = CONGRUENT} && {2 (AlVr) and 4 (ArVl) = INCONGRUENT}
        MST_set = reduce(np.union1d, (responsive[1], responsive[5], responsive[2], responsive[4])) 
        
        # Always responding (most) - intersection
        ALWAYS_responding = reduce(np.intersect1d, (VIS_set, MST_set, AUD_set))
        
        # Modulated (intersection and difference)
        VIS_modulated = np.setdiff1d(np.intersect1d(VIS_set, MST_set), AUD_set) 
        AUD_modulated = np.setdiff1d(np.intersect1d(AUD_set, MST_set), VIS_set)

        # TODO: potentially add congruent / incongruent distinction
        return {'VIS':VIS_set,
                'VIS_modulated':VIS_modulated,
                'VIS_only': np.setdiff1d(VIS_set, np.union1d(MST_set, AUD_set)),
                'AUD':AUD_set,
                'AUD_modulated':AUD_modulated,
                'AUD_only':np.setdiff1d(AUD_set, np.union1d(MST_set, VIS_set)),
                'MST':MST_set,
                'MST_only': np.setdiff1d(MST_set, np.union1d(AUD_set, VIS_set)),
                'ALWAYS_responding' : ALWAYS_responding,
                'TOTAL': reduce(np.union1d, (MST_set, AUD_set, VIS_set)),
                'diagram_setup' : [(set(VIS_set), set(AUD_set), set(MST_set)), ('VIS', 'AUD', 'MST'), ('g', 'r', 'purple')]}


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
                indices = self.NEURON_groups[f'{GROUP}_{GROUP_type}']
                
            case 'modality_specific':
                indices = self.NEURON_groups[f'{GROUP}_only']
            
            case 'all':
                indices = self.NEURON_groups[f'{GROUP}']
        
        # incorporate brain region indices if provided
        if BrainRegionIndices is not None:
            indices = np.intersect1d(BrainRegionIndices, indices)

        if len(indices) == 0:
            print(f'{GROUP}_{GROUP_type} in the given brain region is an empty set, no SUCH neurons!!!')
        ###
        # DEBUGGING only (or example neurons potentially) to plot single neurons for debugging
        # print(GROUP, GROUP_type, indices.size)
        # STATS = {i:[] for i in range(8)}
        # for trial in range(8):
        #     for test in ('zeta', 'ttest', 'wilcoxon'):
        #         sig = self.byTTS[trial] if test != 'zeta' else self.TT_zeta[trial]
        #         _, _, stat = self.responsive_trial_locked(neurons = sig,
        #                                                window = self.TRIAL_FRAMES,
        #                                                criterion = (0.05, 0.45),
        #                                                trial_ID=trial,
        #                                                method=test)
        #         STATS[trial].append(stat)

        # IFR = [self.CASCADE[tt] for tt in self.CASCADE]
        # toplot = np.random.choice(indices, 50)
        # for index in toplot:
        #     plot_1neuron(all_trials_signal=TT_blc_signal,
        #                  single_neuron=index,
        #                  session_neurons=(self.sess_neur, self.session_names),
        #                  fluorescence=self.FLUORO_RESP,
        #                  trial_names=self.tt_names,
        #                  time = self.time,
        #                  CASCADE=IFR,
        #                  STATS=STATS)
        # breakpoint()
        ###

        # all the trials (w all timepoints) of 4 combined trial types 
        # for significant neurons (from brain area) in indices
        neurons_to_study = [sig[:,:, indices] for sig in signals]
        return [(avr, sem) for _, avr, sem in (calc_avrg_trace(trace, self.time, PLOT = False)
                for trace in neurons_to_study)], len(indices)

    @staticmethod
    def fluorescence_response (signal: np.ndarray | dict[int:np.ndarray],
                               window: tuple[int, int] = (16,32),
                               method: Literal['mean', 'peak'] = 'peak') -> np.ndarray:
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
        if isinstance(signal, dict):
            res = np.empty((signal[0].shape[-1], 3, len(signal)))
            tt_sigs = [signal[tt] for tt in sorted(signal.keys())]
        elif isinstance(signal, np.ndarray):
            res = np.empty((signal.shape[-1], 3, 1))
            tt_sigs = [signal]

        assert all(len(sig.shape) == 3 for sig in tt_sigs), 'Signal arrays must be 3D - (ntrials, ntimes, nneurons)!'
        for itt, sig_array in enumerate(tt_sigs):
            # Fluorescence response adapted from (Meijer et al., 2017)
            # mean fluorescence during stimulus presentation for all trials
            Fmean  = np.nanmean(sig_array[:,window[0]:window[1],:], axis = 1)
            F = Fmean.copy()
            if method == 'peak':
                Fmax  = np.nanmax(sig_array[:,window[0]:window[1],:], axis = 1)
                Fmin  = np.nanmin(sig_array[:,window[0]:window[1],:], axis = 1)
                max_mask = where(Fmean > 0)
                min_mask = where(Fmean < 0)
                # nan trials are left as NaNs
                F[max_mask] = Fmax[max_mask]
                F[min_mask] = Fmin[min_mask]

            # signal being fed in is already baseline corrected, so all these
            # are about ∆FR
            # this should get rid of the nans
            meanF = np.nanmean(F, axis = 0) # mean fluorescence response over trials
            stdF = np.nanstd(F, axis = 0) # std of fluorescence response over trials
            cohdF = meanF / stdF # Cohen's d of fluorescence response over trials

            res[:, 0, itt] = meanF
            res[:, 1, itt] = stdF
            res[:, 2, itt] = cohdF

        # assert not np.isnan(res).any(), '[BUG]: NaNs were NOT removed during FR aggregation per neuron!!!'
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
            # if AV.NAME == 'g1pre':
            #     continue
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



### ---------- Main block that runs the file as a script
if __name__ == '__main__':
    tt_grid = {0:(0,2),1:(0,0),2:(0,1),3:(1,2),
               4:(1,1),5:(1,0),6:(0,3),7:(1,3)}
    
    # Neuron types analysis (venn diagrams)
    neuron_typesVENN_analysis()
    NEURON_TYPES_TT_ANALYSIS('modulated', add_CASCADE=True, pre_post='pre')
    NEURON_TYPES_TT_ANALYSIS('modality_specific', add_CASCADE=True, pre_post='pre')
    NEURON_TYPES_TT_ANALYSIS('all', add_CASCADE=True, pre_post='pre')

    # Trial type analysis (average & snake plot) - all neurons, not taking into account neuron types groups
    # NOTE: doesnt work well on cascade
    # TT_ANALYSIS(tt_grid=tt_grid, SNAKE_MODE='signif')