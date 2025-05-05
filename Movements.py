import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

from collections import defaultdict
from AUDVIS import AUDVIS, Behavior, load_in_data
from typing import Literal

def explore_movements(pre_post: Literal['pre', 'post', 'both'] = 'pre'):
    '''
    Builds design matrix for GLM
    '''
    AVs = load_in_data(pre_post=pre_post)
    n_ts = AVs[0].signal.shape[1]
    time = np.linspace(-AVs[0].trial_sec[0], AVs[0].trial_sec[1], n_ts)

    tt_col_lstl = {6:('dodgerblue', '-'),
                    7:('dodgerblue', '--'),
                    0:('red','-'),
                    3:('red', '--'),
                    1:('goldenrod', '-'),
                    5:('goldenrod', '--'),
                    2:('goldenrod', '-.'),
                    4:('goldenrod', ':')}
    
    tt_col_lstl2 = {6:('lightskyblue', '-'),
                    7:('lightskyblue', '--'),
                    0:('lightsalmon','-'),
                    3:('lightsalmon', '--'),
                    1:('palegoldenrod', '-'),
                    5:('palegoldenrod', '--'),
                    2:('palegoldenrod', '-.'),
                    4:('palegoldenrod', ':')}
    
    BBGSs = []
    for AV in AVs:
        behavior_by_session = {'running':dict(),
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
                    behavior_by_session[attr][f'sess{isess}'] = {tt:np.full((1, n_ts), np.nan) 
                                                                 for tt in np.unique(AV.trials)}

        BBGSs.append(behavior_by_session)
    
    aggregated_behavior = {'g1':defaultdict(lambda:defaultdict(lambda:[])), 
                           'g2':defaultdict(lambda:defaultdict(lambda:[]))}
    
    # make plots for the different behaviors across trial types
    for ig, BBSs in enumerate(BBGSs):
        # plot for each session and each behavior across trial types
        f, axs = plt.subplots(nrows=3, ncols=len(BBSs['running']), 
                              figsize= (2.5*len(BBSs['running']), 9), 
                              sharex='all', sharey='row')

        # go through behaviors (rows), in each have data on every session
        for ib, (beh_name, beh_by_sess) in enumerate(BBSs.items()):
            for iss, (sess_name, TT_beh_dict) in enumerate(beh_by_sess.items()):
                if ib == 0:
                    axs[ib, iss].set_title(f'Sess_{iss} \n({AVs[ig].sessions[iss]['session']})')
                if iss == 0:
                    axs[ib, iss].set_ylabel(f'Z-{beh_name}')
                
                for itt, ttBeh in TT_beh_dict.items():
                    aggregated_behavior[f'g{ig+1}'][beh_name][itt].append(ttBeh)
                    if itt == 0:
                        axs[ib, iss].axvline(0)
                        axs[ib, iss].axvline(1)
                        axs[ib, iss].set_xlabel('Time (s)')
                    
                    # plot for each session and each behavior across trial types
                    col, lnstl = tt_col_lstl[itt]
                    avrg = np.nanmean(ttBeh, axis=0)
                    sem = stats.sem(ttBeh, axis=0, nan_policy='omit')
                    axs[ib, iss].fill_between(time, avrg - sem, avrg + sem,
                                              alpha = 0.1, color = col, linestyle = lnstl )
                    axs[ib, iss].plot(time, avrg,color=col, linestyle = lnstl)

        # plot for each session and each behavior across trial types
        f.tight_layout()
        if not os.path.exists(pltdir:='plots'):
            os.makedirs(pltdir)
        f.savefig(os.path.join(pltdir, f'group{ig+1}_behavior_exploration.png'), dpi = 500)
        plt.close()

    # to plot averages over sessions and compare DR vs NR
    f2, ax2 = plt.subplots(nrows=3, ncols=8, figsize = (2.5*8, 9), 
                           sharex='col', sharey='row')
    
    tnames = list(AV.str_to_int_trials_map)
    gname_map = {'g1':'DR', 'g2':'NR'}
    for gname, Gdict in aggregated_behavior.items():
        for ibeh, (bname, Bdict) in enumerate(Gdict.items()):
            for it, all_sess_behaviors in Bdict.items():
                if it == 0:
                    ax2[ibeh, it].set_ylabel(f'Z-{bname}')
                if ibeh == len(Gdict) - 1:
                    ax2[ibeh, it].set_xlabel('Time (s)')
                
                all_sessions = np.concat(all_sess_behaviors)
                col, lnstl = tt_col_lstl2[it] if '2' in gname else tt_col_lstl[it]
                alpha = 0.8 if '2' in gname else 1
                
                if '1' in gname:
                    ax2[ibeh, it].axvline(0, color = 'k', alpha = 0.5, linestyle = '--')
                    ax2[ibeh, it].axvline(1, color = 'k', alpha = 0.5, linestyle = '--')
                    
                    if ibeh == 0:
                        ax2[ibeh, it].set_title(tnames[it], color = col)
                
                avrg = np.nanmean(all_sessions, axis=0)
                sem = stats.sem(all_sessions, axis = 0, nan_policy='omit')
                ax2[ibeh, it].fill_between(time, avrg - sem, avrg + sem,
                                           alpha = 0.2 * alpha, color = col, linestyle = lnstl)
                ax2[ibeh, it].plot(time, avrg,color=col, linestyle = lnstl, 
                                   label = gname_map[gname])
                
                if '2' in gname:
                    ax2[ibeh, it].legend(loc = 2, fontsize = 8)
    
    f2.tight_layout()
    f2.savefig(os.path.join(pltdir, f'behavior_exploration(group_comparison).png'), dpi = 500)
    plt.show()

if __name__ == '__main__':
    explore_movements()
