import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

from collections import defaultdict
from AUDVIS import AUDVIS, Behavior, load_in_data
from typing import Literal

# TODO: clean up into separate functions for each plot
class Movements:
    def __init__(self,
                 pre_post: Literal['pre', 'post', 'both'] = 'pre'):
        '''
        Builds design matrix for GLM
        '''
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
        # 1) first plot average behavior (over trials) per session per trial type (overlaid)
        self.aggregated_behavior, self.pltdir = self.aggregate_trial_type_behavior_by_session(
            plot=False)
        # 2) second plot comparing average behavior (over trials and sessions) per trial type between DR and NR (overlaid)
        # self.behaviors_DR_vs_NR(show=True)

        # 3) third series of plots is per session raw signal during each trial type and corresponding behaviors
        self.raw_signals()
    
    
    def get_behaviors_by_session(self)->dict:
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
                        behavior_by_session[attr][f'sess{isess}'] = AV.separate_signal_by_trial_types(beh_sig)
                    else:
                        print(f'Session{isess}, {attr} missing')
                        behavior_by_session[attr][f'sess{isess}'] = {tt:np.full((90, self.n_ts), np.nan) 
                                                                    for tt in np.unique(AV.trials)}

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

    
    # to plot raw concatenated signals
    # TODO: 1 plot / session with comlete signal
    def raw_signals(self):
        # tt_grid = {
        #     # AUD trials
        #     0:(4,0), 3:(4,0),
        #     # MST+ congruent
        #     1:(0,1), 5:(0,1),
        #     # MST- incongruent
        #     2:(4,1), 4:(4,1),
        #     # VIS trials
        #     6:(0,0), 7:(0,0)}
        
        for ig, AV in enumerate(self.AVs):
            if ig == 0:
                continue
            Group_all_TT_behaviors: dict[
                str:dict[
                    int:list[np.ndarray]
                    ]] = self.aggregated_behavior[f'g{ig+1}']
            n_sessions = len(AV.session_neurons)
            # concatenated plot
            for isess in range(n_sessions):
                if isess < 8:
                    continue
                if not os.path.exists(sess_dir:= os.path.join(
                    self.pltdir, f'g{ig+1}', f'session{isess}')):
                    os.makedirs(sess_dir)

                averaged_neurons = AV.session_average_zsig[isess]
                averaged_neurons_TT:dict[
                    int:np.ndarray] = AV.separate_signal_by_trial_types(averaged_neurons)
                
                for itt in range(len(self.tnames)):
                    f3, a3 = plt.subplots(nrows=4, figsize = (20, 9), sharex='col')    
                    # row, col = tt_grid[itt]
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
                    a3[row].vlines(x = np.arange(AV.TRIAL[0], average_sig_flat.size - 32, 47
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
                        a3[row].vlines(x = np.arange(AV.TRIAL[0], average_sig_flat.size - 32, 47
                                                    ) * (1/AV.SF),
                                    ymin = ymin, ymax = yrange[1],
                                    color = clr, linestyle = lnstl, linewidth = 2.5)
                        a3[row].set_ylabel(f'Z-{behavior_name}')
                        if row == len(Group_all_TT_behaviors):
                            a3[row].set_xlabel(f'Time (s)')
                
                    f3.tight_layout()
                    f3.savefig(os.path.join(sess_dir, f'raw_SIGS_{self.tnames[itt]}.png'), dpi = 300)
                    plt.close()

if __name__ == '__main__':
    Movements()
