import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    
    # make plots for the different behaviors across trial types
    for ig, BBSs in enumerate(BBGSs):
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
                    if itt == 0:
                        axs[ib, iss].axvline(0)
                        axs[ib, iss].axvline(1)
                        axs[ib, iss].set_xlabel('Time (s)')
                        
                    col, lnstl = tt_col_lstl[itt]
                    avrg = np.nanmean(ttBeh, axis=0)
                    sem = stats.sem(ttBeh, axis=0, nan_policy='omit')
                    axs[ib, iss].fill_between(time, avrg - sem, avrg + sem,
                                              alpha = 0.1, color = col, linestyle = lnstl )
                    axs[ib, iss].plot(time, avrg,color=col, linestyle = lnstl)

        f.tight_layout()
        f.savefig(f'group{ig}_behavior_exploration.png', dpi = 500)


if __name__ == '__main__':
    explore_movements()
