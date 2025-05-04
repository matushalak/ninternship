import pandas as pd
import pyglmnet as glm
import numpy as np
import scipy as sp
import scipy.signal as sig
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt

from AUDVIS import Behavior, AUDVIS, load_in_data
from typing import Literal


def design_matrix(pre_post: Literal['pre', 'post', 'both'] = 'pre'):
    '''
    Builds design matrix for GLM
    '''
    AVs = load_in_data(pre_post=pre_post)
    
    for AV in AVs:
        str_tbs = AV.trials_apply_map(AV.trials, AV.int_to_str_trials_map)
        # Stimulus kernels - same for each session [SESSION-level]
        stim_kernels = stimulus_kernels(tbs=str_tbs, 
                                        nts=AV.signal.shape[1],
                                        SF = AV.SF,
                                        trial_frames=AV.TRIAL,
                                        n_basis=9,
                                        basis_window=(0,0.4),
                                        basis_width=0.1)

        breakpoint()        
        # Behavior kernels - only for sessions with behavioral information [SESSION-level]
        behav_kernels = behavior_kernels(sessions=AV.sessions,
                                        nts=AV.signal.shape[1],
                                        SF = AV.SF,
                                        trial_frames=AV.TRIAL,
                                        n_basis=9,
                                        basis_window=(0,0.4),
                                        basis_width=0.1)

        # TODO? [NEURON-level]
        # Spike history

        # Other neurons' spike history


def stimulus_kernels(tbs:np.ndarray, nts:int, SF:float,
                     trial_frames:tuple[int, int],
                     n_basis:int,
                     basis_window:tuple[float, float],
                     basis_width:float)->np.ndarray:
    '''
    Generates stimulus kernels for GLM separately for bars and sounds (2 mod)
        (for M in (bar, sound)
        M_present: (0,1)), 
        M_direction: rightward(+1, -1)leftward
        M_absolute_position: left(0,...,1)right, 
        M_onset_kernel (could be 2 based on direction),
        M_offset_kernel (could be 2 based on direction)

    
    INPUT:
        tbs: trials by session
        nts: number of samples in a trial
        SF: sampling frequency
    '''
    # trials as rows
    trs = tbs.transpose(1,0) if tbs.shape[0] < tbs.shape[1] else tbs
    ntrials, nsessions = trs.shape
    trial_frames = np.array(trial_frames)
    direction_dict = {'l':1, 'r':-1}
    nstim_ts = trial_frames[1] - trial_frames[0]
    # linearly move from side to side
    stimuli = np.linspace(0,1, nstim_ts)
    # Raised cosine bases (for now, same used for all variables, 
    # can VARY for each column)
    CosineBases = rcb(n_basis=n_basis, window_s=basis_window, width_s=basis_width,
                      dt = 1/SF)
    all_session_Xs = []
    for isess in range(nsessions):
        trials = trs[:,isess]
        # diff features as columns of bigger matrix
        Xstim = np.zeros(shape=(ntrials*nts, 
                                5 * 2 # 2 modalities of stimuli
                                ))
        trial_idx = 0
        for it in range(ntrials):
            tname = trials[it]

            stimStart, stimEnd = trial_frames + trial_idx
            # Visual stimulus characteristics
            if 'V' in tname:
                # column 0: Vpresent
                Xstim[stimStart:stimEnd, 0] = 1
                # column 1: Vdirection
                Xstim[stimStart:stimEnd, 1] = (Vdirection := direction_dict[tname[tname.index('V')+1]])
                # column 2: Vposition 
                # (0,...1) if bar is moving from left to right 
                # (1,...,0) if bar is moving right to left
                Xstim[stimStart:stimEnd, 2] = stimuli[::Vdirection]
                # column 3: Onset kernel to capture sudden bar onset
                Xstim[stimStart, 3] = 1
                # column 4: Offset kernel to capture sudden bar offset
                Xstim[stimEnd, 4] = 1
            
            if 'A' in tname:
                # column 0: Apresent
                Xstim[stimStart:stimEnd, 5] = 1
                # column 1: Adirection
                Xstim[stimStart:stimEnd, 6] = (Adirection := direction_dict[tname[tname.index('A')+1]])
                # column 2: Aposition 
                # (0,...1) if sound is moving from left to right 
                # (1,...,0) if sound is moving right to left
                Xstim[stimStart:stimEnd, 7] = stimuli[::Adirection]
                # column 3: Onset kernel to capture sudden sound onset
                Xstim[stimStart, 8] = 1
                # column 4: Offset kernel to capture sudden sound offset
                Xstim[stimEnd, 9] = 1
            
            trial_idx += nts
        
        # All trials done, get bases to account for lags
        xbases = [getBases(Xcol=Xstim[:,ic], Bases=CosineBases)
                  for ic in range(Xstim.shape[1])]
        Xbases = np.column_stack(xbases)
        X_sess = np.column_stack([Xstim, Xbases])
        all_session_Xs.append(X_sess)
    X = np.dstack(all_session_Xs)
    return X


def behavior_kernels(sessions:dict,
                     nts:int, SF:float,
                     trial_frames:tuple[int, int],
                     n_basis:int,
                     basis_window:tuple[float, float],
                     basis_width:float)->np.ndarray:
    behaviors = [sessions[i]['behavior'] 
                 for i in sorted(sessions.keys())]
    all_session_Xb = []
    for isess in range(len(behaviors)):
        pass





# Raised cosine bases (lag features)
def rcb(n_basis:int, 
        window_s:tuple[float, float], 
        width_s:float, dt:float,
        returnLags:bool = False,
        plot:bool = False):
    """
    raised_cosine_basis
    --------------------
    n_basis   : number of cosines
    window_s  : (t_min, t_max) in seconds
    width_s   : half-width of each cosine (seconds)
    dt        : frame duration (seconds)
    returns   : (n_basis, n_lags) matrix
    """
    t_min, t_max = window_s
    lags   = np.arange(int(np.round(t_min/dt)),
                       int(np.round(t_max/dt))+1)
    centres= np.linspace(lags[0], lags[-1], n_basis)
    width  = width_s/dt

    if plot:
        fg, ax = plt.subplots()

    B = [] # collects cosine bases
    for c in centres:
        x        = (lags - c) / width
        phi      = 0.5*(1 + np.cos(np.pi*x))
        phi[np.abs(x) > 1] = 0
        B.append(phi)
        if plot:
            ax.plot(phi)

    if plot:
        plt.tight_layout()
        plt.show() 

    if not returnLags:
        return np.asarray(B) # (n_basis x n_lags)
    else:
        return (np.asarray(B), 
                lags) # array of frame lags


def getBases(Xcol:np.ndarray, 
             Bases:np.ndarray,
             dropFirst:bool = True) -> np.ndarray:
    '''
    Xcol is a 1D array shape (n_trials * n_ts_per_trial)
    Bases is 2D array shape (n_bases, n_lags)

    Returns convolutions of Xcol with provided bases
        The purpose of this is to be able to account for different delays in response
    '''
    basesCols = []
    nbases = Bases.shape[0]
    for ib in range(nbases):
        if dropFirst and ib == 0:
            continue

        # Convolution with the given basis
        Xcol_conv = np.convolve(Xcol, Bases[ib, :])[:Xcol.size]
        basesCols.append(Xcol_conv)
    
    Xcol_bases = np.column_stack(basesCols)
    assert Xcol_bases.shape == (Xcol.size, Bases.shape[0] if not dropFirst 
                                else Bases.shape[0]-1)
    return Xcol_bases



if __name__ == '__main__':
    design_matrix()
