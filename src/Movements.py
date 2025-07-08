import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from sklearn import decomposition
import os
from collections import defaultdict
from typing import Literal
from tqdm import tqdm
import pickle as pkl
from joblib import Parallel, delayed, cpu_count

from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.analysis_utils import general_separate_signal, group_separated_signal, fluorescence_response
from src.VisualAreas import Areas

from src import PYDATA, PLOTSDIR

class Movements:
    '''
    Figure 1: Stimulus-evoked neural activity correlates with stimulus-evoked movement across visuo-cortical regions
    Explores running, whisker and pupil signals
    '''
    def __init__(self,
                 pre_post: Literal['pre', 'post', 'both'] = 'pre'):
        # Setup attributes used across methods
        self.AVs : list[AUDVIS] = load_in_data(pre_post=pre_post)
        self.n_ts = self.AVs[0].signal.shape[1]
        self.time = np.linspace(-self.AVs[0].trial_sec[0], self.AVs[0].trial_sec[1], self.n_ts)
        self.tnames = list(self.AVs[0].str_to_int_trials_map)
        self.gname_map = {'g1':'DR', 'g2':'NR'}

        self.tt_col_lstl = {
            6:('dodgerblue', '-'),
            7:('dodgerblue', '--'),
            'V':('dodgerblue', '-'),
            0:('red','-'),
            3:('red', '--'),
            'A':('red','-'),
            1:('goldenrod', '-'),
            5:('goldenrod', '--'),
            2:('goldenrod', '-.'),
            4:('goldenrod', ':'),
            'AV':('goldenrod', '-')}

        self.tt_col_lstl2 = {
            6:('lightskyblue', '-'),
            7:('lightskyblue', '--'),
            'V':('lightskyblue', '--'),
            0:('lightsalmon','-'),
            3:('lightsalmon', '--'),
            'A':('lightsalmon', '--'),
            1:('palegoldenrod', '-'),
            5:('palegoldenrod', '--'),
            2:('palegoldenrod', '-.'),
            4:('palegoldenrod', ':'),
            'AV':('palegoldenrod', '--')}
        
        # Perform intividual analyses
        # 0) behavior by group sessions (self.BBGSs)
        self.BBGSs:list[dict[str:dict]] = self.get_behaviors_by_session()
        print('0) Behavior by Group, Session, and Trial-type DONE')
        
        # 1) first plot average behavior (over trials) per session per trial type (overlaid)
        self.aggregated_behavior, self.pltdir = self.aggregate_trial_type_behavior_by_session(
            plot=False
            )
        print('1) Behavior by Group, Trial-type and Session DONE')
        
        # 2) second plot comparing average behavior (over trials and sessions) per trial type between DR and NR (overlaid)
        self.DRNR_QUANT = self.behaviors_DR_vs_NR(show=False)
        self.quantify()
        print('2) Averaged comparison between DR and NR DONE')
        
        # 3) third series of plots is per session raw signal during each trial type and corresponding behaviors
        # separate plot per trial type
        # self.raw_signals()

        # 4) separate plot per session (random chunk from session of consecutive trials of all trial types)
        self.raw_signals(all_TT_together=True, 
                         N_example_trials=15,
                         averaged=False,
                         )

        # Correlation Behavior (Whisker, Running and Pulil) and signal throughout trial (47 bins)
        # In each time bin, take correlation between 90 values from signal and 90 values from behavior [or 180 if combining LR]
        # **(calculate seperately for each neuron** and **each trial type**, then show average over all neurons of a given brain area (4 columns)
        # across 3 types of stimuli) (V / A / AV) - 3 lines
        # for DR and NR separately (2 rows)
        self.CORR, self.singleCORR = self.signal_correlations()
        
        # Heatmap, Lineplot, Brain map
        self.analyze_correlations()

        # BONUS
        # TODO: Direction selectivity in whisker movements / running / pupil vs Neuronal activity!

    
    def get_behaviors_by_session(self, addTT:bool = True
                                 )->list[dict[str:dict]]:
        BBGSs = []
        for AV in self.AVs:
            behavior_by_session = {
                'running':dict(),
                'whisker':dict(),
                'pupil':dict()}

            TTs = ['V', 'A', 'AV'] #np.unique(AV.trials)
            ttsizes = [180, 180, 360]
            for isess in list(AV.sessions):
                beh = AV.sessions[isess]['behavior']
                for attr in list(behavior_by_session):
                    if hasattr(beh, attr):
                        beh_sig = getattr(beh, attr)
                        if addTT:
                            behavior_by_session[attr][f'sess{isess}'] = general_separate_signal(
                                sig=beh_sig, trial_types=AV.sessions[isess]['trialIDs'],
                                # comment out if want RAW trial_types
                                trial_type_combinations=[('Vr', 'Vl'), ('Al', "Ar"), ('AlVl', 'ArVr', 'AlVr', 'ArVl')],
                                separation_labels= ['V', 'A', 'AV']
                                )
                        else:
                            behavior_by_session[attr][f'sess{isess}'] = beh_sig
                    else:
                        print(f'Session{isess}, {attr} missing')
                        if addTT:
                            behavior_by_session[attr][f'sess{isess}'] = {tt:np.full((ttsize, self.n_ts), np.nan) 
                                                                         for tt, ttsize in zip(TTs, ttsizes)}
                        else:
                            behavior_by_session[attr][f'sess{isess}'] = np.full((720, self.n_ts), np.nan)

            BBGSs.append(behavior_by_session)
        
        return BBGSs


    def aggregate_trial_type_behavior_by_session(
            self,plot:bool = True)->dict:
        # output dictionary
        aggregated_behavior = {
            'g1':defaultdict(lambda:defaultdict(lambda:[])), 
            'g2':defaultdict(lambda:defaultdict(lambda:[]))}
        
        # make plots for the different behaviors across trial types
        for ig, BBSs in enumerate(self.BBGSs):
            if plot:
                # plot for each session and each behavior across trial types
                f, axs = plt.subplots(nrows=3, ncols=len(BBSs['running']), 
                                    figsize= (2.5*len(BBSs['running']), 9), 
                                    sharex='all', sharey='row')
            
            # go through behaviors (rows), in each have data on every session
            for ib, (beh_name, beh_by_sess) in enumerate(BBSs.items()):
                for iss, (sess_name, TT_beh_dict) in enumerate(beh_by_sess.items()):
                    if ib == 0 and plot:
                        axs[ib, iss].set_title(f'Sess_{iss} \n({self.AVs[ig].sessions[iss]['session']})')
                    if iss == 0 and plot:
                        axs[ib, iss].set_ylabel(f'Z-{beh_name}')
                    
                    for itt, ttBeh in TT_beh_dict.items():
                        aggregated_behavior[f'g{ig+1}'][beh_name][itt].append(ttBeh)
                        if plot:
                            if itt == list(TT_beh_dict)[0]:
                                axs[ib, iss].axvline(0)
                                axs[ib, iss].axvline(1)
                                axs[ib, iss].set_xlabel('self.time (s)')
                            
                            # plot for each session and each behavior across trial types
                            col, lnstl = self.tt_col_lstl[itt]
                            avrg = np.nanmean(ttBeh, axis=0)
                            sem = stats.sem(ttBeh, axis=0, nan_policy='omit')
                            axs[ib, iss].fill_between(self.time, avrg - sem, avrg + sem,
                                                    alpha = 0.1, color = col, linestyle = lnstl )
                            axs[ib, iss].plot(self.time, avrg,color=col, linestyle = lnstl)

            if not os.path.exists(pltdir:=os.path.join(PLOTSDIR, 'MOVEMENT')):
                os.makedirs(pltdir)
            
            if plot:
                # plot for each session and each behavior across trial types
                f.tight_layout()
                f.savefig(os.path.join(pltdir, f'group{ig+1}_behavior_exploration.png'), dpi = 500)
                plt.show()
                plt.close()
        
        return aggregated_behavior, pltdir

    def behaviors_DR_vs_NR(self,show:bool = True
                           )->pd.DataFrame:
        # to plot averages over sessions and compare DR vs NR
        f2, ax2 = plt.subplots(nrows=3, 
                               #TODO: change automatically based on group (3) vs non-grouped (8) trials 
                               ncols=3,#8, 
                               figsize = (2.5*3, 
                                          9), 
                            sharex='col', sharey='all')
        
        quantres = {'Group':[], 'Behavior':[], 'Trial':[], 'Signal':[]}

        for gnum, (gname, Gdict) in enumerate(self.aggregated_behavior.items()):

            for ibeh, (bname, Bdict) in enumerate(Gdict.items()):
                for it, (TT, all_sess_behaviors) in enumerate(Bdict.items()):
                    if it == 0:
                        ax2[ibeh, it].set_ylabel(f'Z-{bname}')
                    if ibeh == len(Gdict) - 1:
                        ax2[ibeh, it].set_xlabel('Time (s)')
                    
                    all_sessions = np.concat(all_sess_behaviors)
                    col, lnstl = self.tt_col_lstl2[TT] if '2' in gname else self.tt_col_lstl[TT]
                    alpha = 0.8 if '2' in gname else 1
                    
                    if '1' in gname:
                        ax2[ibeh, it].axvline(0, color = 'k', alpha = 0.5, linestyle = '--')
                        ax2[ibeh, it].axvline(1, color = 'k', alpha = 0.5, linestyle = '--')
                        
                        if ibeh == 0:
                            if TT not in ('V', 'A', 'AV'):
                                ax2[ibeh, it].set_title(self.tnames[it], color = col)
                            else:
                                ax2[ibeh, it].set_title(TT, color = col)
                    # quantification
                    avrbeh = fluorescence_response(np.dstack(all_sess_behaviors), window=self.AVs[gnum].trial_frames, 
                                                   returnF=True, 
                                                #    retMEAN_only=True # taking session-average: underpowered
                                                   ).squeeze().flatten().tolist()
                    quantres['Signal'] += avrbeh
                    quantres['Group'] += [gname] * len(avrbeh)
                    quantres['Behavior'] += [bname] * len(avrbeh)
                    quantres['Trial'] += [TT] * len(avrbeh)
                    
                    # plot
                    avrg = np.nanmean(all_sessions, axis=0)
                    sem = stats.sem(all_sessions, axis = 0, nan_policy='omit')
                    ax2[ibeh, it].fill_between(self.time, avrg - sem, avrg + sem,
                                            alpha = 0.2 * alpha, color = col, linestyle = lnstl)
                    ax2[ibeh, it].plot(self.time, avrg,color=col, linestyle = lnstl, 
                                    label = self.gname_map[gname])
                    
                    if '2' in gname:
                        ax2[ibeh, it].legend(loc = 2, fontsize = 8)
                    
                    ax2[ibeh, it].axis('off')

        f2.tight_layout()
        f2.savefig(os.path.join(self.pltdir, f'behavior_exploration(group_comparison).svg'), 
                #    dpi = 500
                   )
        if show:
            plt.show()
        plt.close()

        out = pd.DataFrame(quantres).dropna()
        return out

    
    def raw_signals(self,
                    all_TT_together:bool = False,
                    N_example_trials:int = 135,
                    averaged:bool = True,
                    ):
        if all_TT_together:
            Group_Beh_by_sess = self.get_behaviors_by_session(addTT=False)
            print('Full behavior by session done')
        for ig, AV in enumerate(self.AVs):
            if not all_TT_together:
                Group_all_TT_behaviors: dict[
                    str:dict[
                        int:list[np.ndarray]
                        ]] = self.aggregated_behavior[f'g{ig+1}']
            n_sessions = len(AV.session_neurons)
            # concatenated plot
            for isess in range(n_sessions):
                if not os.path.exists(sess_dir:= os.path.join(
                    self.pltdir, f'g{ig+1}', f'session{isess}')):
                    os.makedirs(sess_dir)

                averaged_neurons = AV.session_average_zsig[isess]
                if averaged:
                    nrows = 4
                    figsize = (20,9)
                else:
                    # pick 3 random neurons
                    random_chosen = np.random.choice(np.arange(*AV.session_neurons[isess]), size=3)
                    nrows = 6
                    figsize = (12, 9)

                # one plot per session with all trial types
                if all_TT_together:
                    sess_TT_indices = find_session_tt_indices(isess, AV)
                    Beh_by_sess = Group_Beh_by_sess[ig]
                    f4, a4 = plt.subplots(nrows=nrows, figsize = figsize, sharex='col')    
                    row = 0
                    ntrials = averaged_neurons.shape[0]
                    avsig_flat_full = averaged_neurons.flatten()
                    full_flat_time = np.linspace(0, avsig_flat_full.size * (1/AV.SF),
                                                 avsig_flat_full.size)
                    
                    random_segment = np.random.choice(np.arange(0, avsig_flat_full.size - (N_example_trials*self.n_ts)))
                    # can finetune how big example want
                    example_trials = np.array([full_flat_time[random_segment], full_flat_time[random_segment] + (N_example_trials*self.n_ts*1/AV.SF)])
                    
                    if averaged:
                        # plot population average signal for whole session
                        a4[row].plot(full_flat_time, avsig_flat_full, color = 'k')
                        # stimulus onsets
                        yrange = (np.nanmin(avsig_flat_full), np.nanmax(avsig_flat_full))
                        ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                        add_TT_onsets(ax = a4[row], session_ttindices=sess_TT_indices, SF = AV.SF,
                                    ymin=ymin, ymax = yrange[1], tt_col_lstl=self.tt_col_lstl,
                                    full_sig_size=avsig_flat_full.size,
                                    stim_start_idx_within_trial=AV.TRIAL[0])
                        a4[row].set_ylabel('Z-dF/F')
                        a4[row].set_xlim(example_trials)
                        row +=1
                    
                    # plot 3 randomly chosen neurons
                    else:
                        for chosen in random_chosen:
                            dff_full = AV.zsig[:, :, chosen].flatten()
                            a4[row].plot(full_flat_time, dff_full, color = 'k')
                            a4[row].axis('off')
                            yrange = (np.nanmin(dff_full), np.nanmax(dff_full))
                            ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                            add_TT_onsets(ax = a4[row], session_ttindices=sess_TT_indices, SF = AV.SF,
                                        ymin=ymin, ymax = yrange[1], tt_col_lstl=self.tt_col_lstl,
                                        full_sig_size=dff_full.size,
                                        stim_start_idx_within_trial=AV.TRIAL[0])
                            a4[row].set_ylabel('z-∆F/F')
                            a4[row].set_xlim(example_trials)
                            row += 1


                    # add behaviors
                    for beh_name, allsessbeh in Beh_by_sess.items():
                        behsess = allsessbeh[f'sess{isess}']
                        beh_flat = behsess.flatten()
                        a4[row].plot(full_flat_time, beh_flat, color = 'k')
                        a4[row].axis('off')
                        # stimulus onsets
                        yrange = (np.nanmin(beh_flat), np.nanmax(beh_flat))
                        ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                        add_TT_onsets(ax = a4[row], session_ttindices=sess_TT_indices, SF = AV.SF,
                                      ymin=ymin, ymax = yrange[1], tt_col_lstl=self.tt_col_lstl,
                                      full_sig_size=avsig_flat_full.size,
                                      stim_start_idx_within_trial=AV.TRIAL[0])
                        a4[row].set_ylabel(f'Z-{beh_name}')
                        a4[row].set_xlim(example_trials)
                        if row == 3:
                            a4[row].set_xlabel(f'Time (s)')
                        
                        row +=1
                    
                    
                    f4.tight_layout()
                    f4.savefig(os.path.join(sess_dir, f'raw_SIGS_allTT(ntrials:{N_example_trials}).svg'), dpi = 300)
                    plt.close()
                    print(f'Group {ig} Session {isess} raw_SIGS_allTT(ntrials:{N_example_trials}) saved!')
                    

                # separate plot per TT and session
                else:
                    averaged_neurons_TT:dict[
                        int:np.ndarray] = general_separate_signal(
                            sig=averaged_neurons, trial_types=AV.sessions[isess]['trialIDs'])
                    
                    for itt in range(len(self.tnames)):
                        f3, a3 = plt.subplots(nrows=4, figsize = (20, 9), sharex='col')    
                        row = 0
                        clr, lnstl = self.tt_col_lstl[itt] #if ig == 0 else self.tt_col_lstl2[itt]
                        ntrials = averaged_neurons_TT[itt].shape[0]
                        average_sig_flat = averaged_neurons_TT[itt].flatten()
                        flat_time = np.linspace(0, average_sig_flat.size * (1/AV.SF), 
                                                average_sig_flat.size)
                        # plot raw signal (average over neurons)
                        a3[row].plot(flat_time, average_sig_flat, 
                                    color = 'k') 
                        # stimulus onsets
                        yrange = (np.nanmin(average_sig_flat), np.nanmax(average_sig_flat))
                        ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                        a3[row].vlines(x = np.arange(AV.TRIAL[0], (47+average_sig_flat.size) - 32, 47
                                                    ) * (1/AV.SF),
                                    ymin = ymin, ymax = yrange[1],
                                    color = clr, linestyle = lnstl, linewidth = 2.5,
                                    label = self.tnames[itt])
                        a3[row].legend(loc = 1)
                        a3[row].set_ylabel('Z-dF/F')
                        
                        for ib, (behavior_name, behavior_all_tt) in enumerate(Group_all_TT_behaviors.items()):
                            row += 1
                            # plot raw behaviors
                            beh_tt_flat = behavior_all_tt[itt][isess].flatten()
                            a3[row].plot(flat_time, beh_tt_flat, 
                                        color = 'k') 
                            # stimulus onsets
                            yrange = (np.nanmin(beh_tt_flat), np.nanmax(beh_tt_flat))
                            ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                            a3[row].vlines(x = np.arange(AV.TRIAL[0], (47+average_sig_flat.size) - 32, 47
                                                        ) * (1/AV.SF),
                                        ymin = ymin, ymax = yrange[1],
                                        color = clr, linestyle = lnstl, linewidth = 2.5)
                            a3[row].set_ylabel(f'Z-{behavior_name}')
                            if row == len(Group_all_TT_behaviors):
                                a3[row].set_xlabel(f'Time (s)')
                    
                        f3.tight_layout()
                        f3.savefig(os.path.join(sess_dir, f'raw_SIGS_{self.tnames[itt]}.png'), dpi = 300)
                        plt.close()
                        print(f'Group {ig} Session {isess} raw_SIGS_{self.tnames[itt]} saved!')


    def signal_correlations(self)->tuple[dict, dict]:
        '''
        extracts signal correlations for each neuron for each trial-type for each behavior
        in a dict: group -> behavior -> TT -> corr_matrix (ts x ts x neurons)
        '''
        # if os.path.exists(
        #     CORRsavepath := os.path.join(PYDATA, 'MovementCORRts.pkl')
        #                   ) and os.path.exists(
        #     SINGLECORRsavepath := os.path.join(PYDATA, 'MovementCORRsingle.pkl')):
        #     # Load files
        #     with open(CORRsavepath, 'rb') as CORR_f:
        #         correlations = pkl.load(CORR_f)
        #     with open(SINGLECORRsavepath, 'rb') as SINGLECORR_f:
        #         singlevalcorrelations = pkl.load(SINGLECORR_f)
            
        #     return correlations, singlevalcorrelations
        
        # different brain areas
        self.ARs:list[Areas] = [Areas(av) for av in self.AVs]

        # Visual trials
        groupedTTnames = ['V', 'A', 'AV']
        groupedTTgroups = [(6, 7), (0, 3), (1, 2, 4, 5)]

        correlations = defaultdict(lambda:defaultdict(dict))
        singlevalcorrelations = defaultdict(lambda:defaultdict(dict))
        for ig, (g, gdict) in enumerate(self.aggregated_behavior.items()):
            AV = self.AVs[ig]
            # get raw group signals separated by trial type
            groupsig = AV.separate_signal_by_trial_types(
                signal=AV.zsig)
            
            # only divided into V, A and AV trials
            groupsig = group_separated_signal(trial_type_combinations=groupedTTgroups,
                                              separation_labels=groupedTTnames,
                                              tt_separated_signal=groupsig)
            
            # get neuron boundaries between each session
            sess_neur = AV.session_neurons

            for ib, (b, bTTdict) in enumerate(gdict.items()):
                for tt, ttlist in tqdm(bTTdict.items()):
                    ttsig = groupsig[tt]

                    # prepare parallel args
                    corr_worker_args = []
                    tt_all_neur = []
                    for isess, sessARR in enumerate(ttlist):
                        neurons_slice = slice(*sess_neur[isess])
                        # ntrials (of trial type tt), nts, nneurons (of session isess)
                        sessDFF = ttsig[:,:,neurons_slice]

                        corr_worker_args.append((sessDFF, sessARR, AV.TRIAL))
                        tt_all_neur.append(corr_worker(sessDFF, sessARR, AV.TRIAL))
                    
                    timewise, singlevals = [t for t, _ in tt_all_neur], [s for _, s in tt_all_neur]
                    temporal_corr_all_neur = np.dstack(timewise)
                    singlevals_all_neur = np.concatenate(singlevals)
                    
                    # add to results array
                    correlations[g][b][tt] = temporal_corr_all_neur
                    singlevalcorrelations[g][b][tt] = singlevals_all_neur
        
        # save files
        # with open(CORRsavepath, 'wb') as CORR_f:
        #     pkl.dump(dict(correlations), CORR_f)
    
        # with open(SINGLECORRsavepath, 'wb') as SINGLECORR_f:
        #     pkl.dump(dict(singlevalcorrelations), SINGLECORR_f)
        
        # return self.signal_correlations()
        return correlations, singlevalcorrelations
    
    def analyze_correlations(self):
        # 0) Heatmap (3 x 2 [DR vs NR]) - neurons from all areas all sessions combined
        # one for each behavior
        heatmap_artists =  [plt.subplots(nrows=2, ncols=3, figsize = (7, 4), sharex='all', sharey='all') for _ in range(3)]
        
        # 1) Basic correlation lineplot 
        # (3 colors - TT, dashed control, solid DR), 1 per area
        # one row per behavior (3)
        lfig, laxs = plt.subplots(nrows=3, ncols=4, sharex='all', sharey='all')

        # collect dataframes for each group-behavior combination
        overallQuantDF = []

        for ig, (g, gdict) in enumerate(self.CORR.items()):
            av = self.AVs[ig]
            AR = Areas(av)
            collstl = self.tt_col_lstl if '1' in g else self.tt_col_lstl2

            for ib, (b, bdict) in enumerate(gdict.items()):
                
                single_neuron_beh_corrs = []
                
                for itt, (tt, ttARR) in enumerate(bdict.items()):
                    col, lstl = collstl[tt]
                    nansessions = np.isnan(ttARR[0,0,:]) # boolean array
                    goodindices = np.nonzero(~nansessions) # integer indexing array
                    # ttGOOD = ttARR[:,:, ~nansessions]
                    
                    # for brainmap
                    single_neuron_beh_corrs.append(self.singleCORR[g][b][tt])

                    # mean over sessions first before overall mean
                    sessmeans = []
                    for st, en in av.session_neurons:
                        if np.isnan(ttARR[0,0,st]):
                            continue
                        sessmeans.append(np.mean(ttARR[:,:,st:en], axis = 2))
                    sessmeans = np.dstack(sessmeans)

                    # get heatmap artists for this behavior
                    hf, haxs = heatmap_artists[ib]
                    hm = sns.heatmap(data = np.mean(sessmeans, axis = 2), ax = haxs[ig, itt],
                                     vmin = -0.02, vmax = 0.02, 
                                     square=True,
                                     cmap='rocket' if ig == 1 else 'mako',
                                     cbar=(itt == 2), 
                                     )
                    haxs[ig, itt].set_xticks((0, 15, 32, 47), (-1,0,1,2), rotation = 'horizontal')
                    haxs[ig, itt].set_yticks((0, 15, 32, 47), (-1,0,1,2))
                    if ig ==1:
                        haxs[ig, itt].set_xlabel('∆F/F time (s)')
                    else:
                        haxs[ig, itt].set_title(f'{tt} trials')
                    if itt ==0:
                        haxs[ig, itt].set_ylabel(f'{b} time (s)')

                    for iarea, (area_name, area_indices) in enumerate(AR.region_indices.items()):
                        lax = laxs[ib, iarea]

                        areagood = np.mean(ttARR[:,:,
                                                   np.intersect1d(area_indices, goodindices)], 
                                            axis = 2).diagonal()
                        lax.axvspan(0,1, color = 'k', alpha = 0.15)
                        lax.plot(self.time, areagood, color = col, linestyle = lstl, label = f'{tt} ({self.gname_map[g]})')
                        lax.set_xlim(self.time.min(), self.time.max())
                        
                        if ib == 0:
                            lax.set_title(area_name)
                        if iarea == 0:
                            lax.set_ylabel(f'r({b}-∆F/F)')
                        if ib == 2:
                            lax.set_xticks((-1,0,1,2))
                            lax.set_xlabel('Time (s)')
            
                single_neuron_beh_corrs = np.column_stack(single_neuron_beh_corrs)
                colnames = ['V', 'A', 'AV']
                
                regions = AR.dfROI['Region'].tolist()
                
                # add correlations for each trial-type
                TT, Correlations = [], []
                for icolnm, colnm in enumerate(colnames):
                    Correlations += single_neuron_beh_corrs[:,icolnm].tolist()
                    TT += [colnm] * len(regions)
                
                regions = regions * len(colnames)
                assert len(regions) == len(TT) == len(Correlations)
                
                # add group number
                group = [g] * len(regions)
                # add behavior
                behavior = [b] * len(regions)

                GroupBehaviorDf = pd.DataFrame({'Group':group,
                                         'Region':regions,
                                         'Behavior':behavior,
                                         'Stimulus':TT,
                                         'Correlation':Correlations})
                
                overallQuantDF.append(GroupBehaviorDf)
                self.corrBrainmap(AR=AR, single_neuron_corrs=single_neuron_beh_corrs, 
                                behavior_name=b, group = g, groupname=self.gname_map[g],
                                colnames=colnames, indices=goodindices)

        # Lineplot
        lfig.tight_layout()
        lfig.savefig(os.path.join(self.pltdir, 'Correlations_AREA_tsplot.png'), dpi = 300)
        
        # Heatmap plot
        for (hf, _), b in zip(heatmap_artists, gdict.keys()):
            hf.tight_layout()
            hf.savefig(os.path.join(self.pltdir, f'{b}_correlations_heatmap.png'), dpi = 300)
        plt.close()

        # Quantification plot
        # assemble dataframe
        QuantDF = pd.concat(overallQuantDF, ignore_index=True)
        QuantDF['R2'] = QuantDF['Correlation']**2
        QuantDF['|r|'] = QuantDF['Correlation'].abs()
        # Filter out neurons not in Visual Areas
        QuantDF = QuantDF.loc[QuantDF['Region'] != '']
        quantplot = sns.catplot(data = QuantDF, x = 'Stimulus', 
                                y = '|r|', # |r| or R2 makes sense here
                                hue = 'Group', 
                                row = 'Behavior', col='Region',
                                col_order=['V1', 'AM/PM', 'A/RL/AL', 'LM'],
                                kind = 'point', dodge = True, estimator='mean')
        plt.tight_layout()
        plt.savefig(os.path.join(self.pltdir, f'BehaviorCorrelations_Quant.svg'))


    def corrBrainmap(self, AR:Areas, single_neuron_corrs:np.ndarray, 
                     behavior_name:str, group:str,  groupname:str,
                     colnames:list[str], indices:np.ndarray):
        abscorrs = single_neuron_corrs.__abs__()
        argmaxes = np.argmax(abscorrs, axis = 1)
        # just take the max correlation (across trial types) for each neuron for given behavior
        maxcorrs = np.take_along_axis(single_neuron_corrs, indices=argmaxes[:,np.newaxis], axis = 1).squeeze()
        
        # to separate V, A, AV
        # indices = [np.where(argmaxes == i)[0] for i in range(abscorrs.shape[1])]
        # indcolors = ['dodgerblue', 'red', 'goldenrod']
        # assert maxcorrs.size == indices[0].size
        assert group in AR.NAME, f'Print {group} is not the same as {AR.NAME}'
        AR.show_neurons(indices=indices, ONLY_indices=True, ZOOM=True, CONTOURS=True, 
                        title=f'{groupname}_{behavior_name}', svg=True, 
                        values=maxcorrs, colorbar = f'r({behavior_name}-∆F/F)',
                        suppressAreaColors=False, interpolate_vals=False)

    
    def quantify(self):
        '''
        Quantifies differences in movements (& movement characteristics)
        between DR and NR animals across trial types
        '''
        between_pairs = [
            (("V", "g1"), ("V", "g2")),
            (("A", "g1"), ("A", "g2")),
            (("AV", "g1"), ("AV", "g2")),
        ]
        within_pairs = [
            (("V", "g1"), ("A", "g1")),
            (("V", "g2"), ("A", "g2")),
        ]
        # palette = {('V', 'g1'):'dodgerblue',
        #         ('A', 'g1'):'red',
        #         ('AV', 'g1'):'goldenrod',
        #         ('V', 'g2'):'lightskyblue',
        #         ('A', 'g2'):'lightsalmon',
        #         ('AV', 'g2'):'palegoldenrod'}
        palette = {'g1':'darkorange', 'g2':'grey'}
        
        for b in self.DRNR_QUANT.Behavior.unique():
            sigcol = f'Z-{b}'
            data = self.DRNR_QUANT.loc[self.DRNR_QUANT['Behavior'] == b].copy()
            data.rename(columns = {'Signal':sigcol}, inplace = True)
            cp = sns.catplot(data = data, x = 'Trial', y = sigcol, 
                             hue = 'Group', #data[['Trial','Group']].apply(tuple, axis = 1),
                             hue_order=['g2', 'g1'],
                            #  (('V', 'g2'),('A', 'g2'),('AV', 'g2'),('V', 'g1'), ('A', 'g1'), ('AV', 'g1')),
                            palette=palette,
                            kind = 'point', dodge = .3, capsize = .2, 
                            legend=True)
            # first: between‑group comparisons
            annot_bw = Annotator(
                cp.ax,
                between_pairs,
                data=data,
                x='Trial',
                y=sigcol,
                hue='Group',
            )
            annot_bw.configure(
                test='Mann-Whitney',#'t-test_ind', 'Mann-Whitney',
                comparisons_correction='Bonferroni',
                text_format='star',
                loc='outside',
                # hide_non_significant = True,
                correction_format="replace"
            )
            annot_bw.apply_and_annotate()

            # second: within-group comparison
            annot_wi = Annotator(
            cp.ax, within_pairs,
            data=data, x='Trial', y=sigcol, hue='Group'
            )
            annot_wi.configure(
                test='Wilcoxon',#'t-test_paired', 'Wilcoxon',
                comparisons_correction='Bonferroni',
                text_format='star',
                loc='outside',
                # hide_non_significant = True,
                correction_format="replace"
            )
            annot_wi.apply_and_annotate()
            
            plt.tight_layout()
            plt.ylim(-0.4, 0.6)
            plt.savefig(os.path.join(self.pltdir, f'MovementsDRNRQuant({b}).svg'))
    
# -------------- Helpers ----------------
def corr_worker(sessDFF:np.ndarray, 
                sessARR:np.ndarray,
                trialSLICE:tuple,
                )->np.ndarray:
    '''
    sessDFF: (ntrials x nts x nneurons) - neural signals
    sessARR: (ntrials x nts) - behavioral signal
    trialSLICE: (trialStart, trialEND) contains frames between which stimulus presentation happens
    
    NOTE: would be better to get principal components of the neural signal or something like that
    and correlate that to the behavior
    '''
    ntrials, nts, nneur = sessDFF.shape
    
    # 3D array nts x nts x nneurons
    # average correlations of each timebin of signal - behavior over trials for each neuron
    neuron_corrs = np.empty(shape=(nts, nts, nneur))
    # 1D array (nneurons) holding the average overall correlation of signal w behavior (over trials)
    single_val_corrs = np.empty(shape = nneur)
    for ineur in range(nneur):
        # part of a bigger matrix that includes corr of behavior w itself [:nts, :nts] and
        # corr of signal w itself [nts:, :nts]
        neuron_corrs[:,:, ineur] = np.corrcoef(sessARR, sessDFF[:,:,ineur], rowvar = False)[:nts, nts:]
        # now includes nans for sessions without facemap
        
        # diagonal of trial-wise correlation matrix contains correlations of behavior and signal 
        # in each trial, the mean of that diagonal contains the average correlation of neuron's
        
        # correlation of trial signal with trial behavior of the mouse (very low, a lot of single neuron noise hard to capture)
        trialsCorr = np.corrcoef(sessARR[:,trialSLICE[0]:trialSLICE[1]], sessDFF[:,trialSLICE[0]:trialSLICE[1],ineur], 
                                rowvar = True
                                )[:ntrials, ntrials:].diagonal()

        # Fisher-z correction to take correlation mean
        fisher = np.arctanh(trialsCorr)
        # inverse fisher of mean
        single_val_corrs[ineur] = np.tanh(np.nanmean(fisher))
        
        # correlation of session averages for each neuron (high and not meaningful)
        # single_val_corrs[ineur] = np.corrcoef(sessARR[:,trialSLICE[0]:trialSLICE[1]].mean(axis=0),
        #                                       sessDFF[:,trialSLICE[0]:trialSLICE[1],ineur].mean(axis = 0),
        #                                       rowvar=True)[0,1]
    
    # Average correlation on trial-level across trials with average population signal (fisher-z correction)
    # trialsCorr = np.corrcoef(sessARR[:,trialSLICE[0]:trialSLICE[1]], sessDFF[:,trialSLICE[0]:trialSLICE[1],:].mean(axis=2), 
    #                         rowvar = True
    #                         )[:ntrials, ntrials:].diagonal()

    # # Fisher-z correction to take correlation mean
    # fisher = np.arctanh(trialsCorr)
    # # inverse fisher of mean
    # single_val_corrs = np.tanh(np.nanmean(fisher))

    # average relationship of signal (xaxis) with given behavior (yaxis)
    # over all neurons in the session for given TT

    # diagonal here is the "line-plot" correlation / time-bin relationship
    # sns.heatmap(neuron_corrs.mean(axis = 2))
    return neuron_corrs, single_val_corrs

def find_session_tt_indices(sess_idx:int, AV:AUDVIS
                    )->np.ndarray:
    tt_indices = []
    for tt, (sess_indices, trial_indices) in AV.trial_types.items():
        sess_mask = sess_indices == sess_idx
        assert sum(sess_mask) == 90
        tt_indices.append(trial_indices[sess_mask])
    
    return np.array(tt_indices)

def add_TT_onsets(ax, 
                  stim_start_idx_within_trial:int,
                  full_sig_size:int,
                  session_ttindices:np.ndarray,
                  SF:float,
                  ymin:float, ymax:float, 
                  tt_col_lstl:dict[int:tuple[str, str]]):
    for itt in range(session_ttindices.shape[0]):
        clr, lnstl = tt_col_lstl[itt]
        x = np.arange(stim_start_idx_within_trial, (47+full_sig_size) - 32, 47) * (1/SF)

        ax.vlines(x = x[session_ttindices[itt,:]],
                  ymin = ymin, ymax = ymax,
                  color = clr, linestyle = lnstl, linewidth = 2.5)



if __name__ == '__main__':
    Movements()
