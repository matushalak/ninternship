import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

from collections import defaultdict
from AUDVIS import AUDVIS, Behavior, load_in_data
from typing import Literal

# TODO: correlation with signal across time bins all neurons (between groups)
# TODO: quantify differences
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
            0:('red','-'),
            3:('red', '--'),
            1:('goldenrod', '-'),
            5:('goldenrod', '--'),
            2:('goldenrod', '-.'),
            4:('goldenrod', ':')}

        self.tt_col_lstl2 = {
            6:('lightskyblue', '-'),
            7:('lightskyblue', '--'),
            0:('lightsalmon','-'),
            3:('lightsalmon', '--'),
            1:('palegoldenrod', '-'),
            5:('palegoldenrod', '--'),
            2:('palegoldenrod', '-.'),
            4:('palegoldenrod', ':')}
        
        # Perform intividual analyses
        # 0) behavior by group sessions (self.BBGSs)
        self.BBGSs = self.get_behaviors_by_session()
        print('0) Behavior by Group, Session, and Trial-type DONE')
        
        # 1) first plot average behavior (over trials) per session per trial type (overlaid)
        self.aggregated_behavior, self.pltdir = self.aggregate_trial_type_behavior_by_session(
            plot=False)
        print('1) Behavior by Group, Trial-type and Session DONE')
        
        # 2) second plot comparing average behavior (over trials and sessions) per trial type between DR and NR (overlaid)
        # self.behaviors_DR_vs_NR(show=True)
        # print('2) Averaged comparison between DR and NR DONE')

        # 3) third series of plots is per session raw signal during each trial type and corresponding behaviors
        # separate plot per trial type
        self.raw_signals()

        # 4) separate plot per session (random chunk from session of consecutive trials of all trial types)
        # self.raw_signals(all_TT_together=True, N_example_trials=100)

        # TODO: Correlation Behavior (choose Whisker) and signal throughout trial (47 bins)
        # In each time bin, take correlation between 90 values from signal and 90 values from behavior [or 180 if combining LR]
        # **(calculate seperately for each neuron** and **each trial type**, then show average over all neurons of a given brain area (4 columns)
        # across 3 types of stimuli) (V / A / AV) - 3 lines
        # for DR and NR separately (2 rows)

        # TODO: Supplementary - same figure for Running and Pupil

        # TODO: Heatmap / snakeplot

        # TODO: Direction selectivity in whisker movements / running / pupil vs Neuronal activity!

        # TODO: contourplot across brain regions (on top of ABA) with color indicating correlation with whisker, run, pupil
        # (overall of one per trial type)
        # can place neurons with np.meshgrid(neuron_xs, neuron_ys)
    
    
    def get_behaviors_by_session(self, addTT:bool = True)->dict:
        BBGSs = []
        for AV in self.AVs:
            behavior_by_session = {
                'running':dict(),
                'whisker':dict(),
                'pupil':dict()}

            for isess in list(AV.sessions):
                beh = AV.sessions[isess]['behavior']
                for attr in list(behavior_by_session):
                    if hasattr(beh, attr):
                        beh_sig = getattr(beh, attr)
                        if addTT:
                            behavior_by_session[attr][f'sess{isess}'] = AV.separate_signal_by_trial_types(beh_sig)
                        else:
                            behavior_by_session[attr][f'sess{isess}'] = beh_sig
                    else:
                        print(f'Session{isess}, {attr} missing')
                        if addTT:
                            behavior_by_session[attr][f'sess{isess}'] = {tt:np.full((90, self.n_ts), np.nan) 
                                                                         for tt in np.unique(AV.trials)}
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
                            if itt == 0:
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

            if not os.path.exists(pltdir:='plots'):
                os.makedirs(pltdir)
            
            if plot:
                # plot for each session and each behavior across trial types
                f.tight_layout()
                f.savefig(os.path.join(pltdir, f'group{ig+1}_behavior_exploration.png'), dpi = 500)
                plt.show()
                plt.close()
        
        return aggregated_behavior, pltdir


    def behaviors_DR_vs_NR(self,show:bool = True):
        # to plot averages over sessions and compare DR vs NR
        f2, ax2 = plt.subplots(nrows=3, ncols=8, figsize = (2.5*8, 9), 
                            sharex='col', sharey='row')
        
        for gnum, (gname, Gdict) in enumerate(self.aggregated_behavior.items()):

            for ibeh, (bname, Bdict) in enumerate(Gdict.items()):
                for it, all_sess_behaviors in Bdict.items():
                    if it == 0:
                        ax2[ibeh, it].set_ylabel(f'Z-{bname}')
                    if ibeh == len(Gdict) - 1:
                        ax2[ibeh, it].set_xlabel('self.time (s)')
                    
                    all_sessions = np.concat(all_sess_behaviors)
                    col, lnstl = self.tt_col_lstl2[it] if '2' in gname else self.tt_col_lstl[it]
                    alpha = 0.8 if '2' in gname else 1
                    
                    if '1' in gname:
                        ax2[ibeh, it].axvline(0, color = 'k', alpha = 0.5, linestyle = '--')
                        ax2[ibeh, it].axvline(1, color = 'k', alpha = 0.5, linestyle = '--')
                        
                        if ibeh == 0:
                            ax2[ibeh, it].set_title(self.tnames[it], color = col)
                    
                    avrg = np.nanmean(all_sessions, axis=0)
                    sem = stats.sem(all_sessions, axis = 0, nan_policy='omit')
                    ax2[ibeh, it].fill_between(self.time, avrg - sem, avrg + sem,
                                            alpha = 0.2 * alpha, color = col, linestyle = lnstl)
                    ax2[ibeh, it].plot(self.time, avrg,color=col, linestyle = lnstl, 
                                    label = self.gname_map[gname])
                    
                    if '2' in gname:
                        ax2[ibeh, it].legend(loc = 2, fontsize = 8)

        f2.tight_layout()
        f2.savefig(os.path.join(self.pltdir, f'behavior_exploration(group_comparison).png'), dpi = 500)
        if show:
            plt.show()
        plt.close()

    
    def raw_signals(self,
                    all_TT_together:bool = False,
                    N_example_trials:int = 135):
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
                # one plot per session with all trial types
                if all_TT_together:
                    sess_TT_indices = find_session_tt_indices(isess, AV)
                    Beh_by_sess = Group_Beh_by_sess[ig]
                    f4, a4 = plt.subplots(nrows=4, figsize = (20, 9), sharex='col')    
                    row = 0
                    ntrials = averaged_neurons.shape[0]
                    avsig_flat_full = averaged_neurons.flatten()
                    full_flat_time = np.linspace(0, avsig_flat_full.size * (1/AV.SF),
                                                 avsig_flat_full.size)
                    
                    random_segment = np.random.choice(np.arange(0, avsig_flat_full.size - (N_example_trials*self.n_ts)))
                    # can finetune how big example want
                    example_trials = np.array([full_flat_time[random_segment], full_flat_time[random_segment] + (N_example_trials*self.n_ts*1/AV.SF)])
                    
                    # plot population average signal for whole session
                    a4[row].plot(full_flat_time, avsig_flat_full, color = 'k')
                    # stimulus onsets
                    yrange = (np.nanmin(avsig_flat_full), np.nanmax(avsig_flat_full))
                    ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                    add_TT_onsets(ax = a4[row], session_ttindices=sess_TT_indices, SF = AV.SF,
                                  ymin=ymin, ymax = yrange[1], tt_col_lstl=self.tt_col_lstl,
                                  full_sig_size=avsig_flat_full.size,
                                  stim_start_idx_within_trial=AV.TRIAL[0])
                    a4[row].legend(loc = 1)
                    a4[row].set_ylabel('Z-dF/F')
                    a4[row].set_xlim(example_trials)

                    # add behaviors
                    for beh_name, allsessbeh in Beh_by_sess.items():
                        row +=1
                        behsess = allsessbeh[f'sess{isess}']
                        beh_flat = behsess.flatten()
                        a4[row].plot(full_flat_time, beh_flat, color = 'k')
                        # stimulus onsets
                        yrange = (np.nanmin(beh_flat), np.nanmax(beh_flat))
                        ymin = yrange[1] - 0.2 * (yrange[1] - yrange[0])
                        add_TT_onsets(ax = a4[row], session_ttindices=sess_TT_indices, SF = AV.SF,
                                      ymin=ymin, ymax = yrange[1], tt_col_lstl=self.tt_col_lstl,
                                      full_sig_size=avsig_flat_full.size,
                                      stim_start_idx_within_trial=AV.TRIAL[0])
                        a4[row].legend(loc = 1)
                        a4[row].set_ylabel(f'Z-{beh_name}')
                        a4[row].set_xlim(example_trials)
                        if row == 3:
                            a4[row].set_xlabel(f'Time (s)')
                    
                    
                    f4.tight_layout()
                    f4.savefig(os.path.join(sess_dir, f'raw_SIGS_allTT(ntrials:{N_example_trials}).png'), dpi = 300)
                    plt.close()
                    print(f'Group {ig} Session {isess} raw_SIGS_allTT(ntrials:{N_example_trials}) saved!')
                    

                # separate plot per TT and session
                else:
                    averaged_neurons_TT:dict[
                        int:np.ndarray] = AV.separate_signal_by_trial_types(averaged_neurons)
                    
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


    # TODO:
    def signal_correlations(self):
        pass
    
    def Quantify(self):
        '''
        Quantifies differences in movements (& movement characteristics)
        between DR and NR animals across trial types
        '''
        raise NotImplementedError
    
    
# -------------- Helpers ----------------
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
