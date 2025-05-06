import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import seaborn as sns
import matplotlib.pyplot as plt

# Encoding model
import pyglmnet as glm
import sklearn.linear_model as skLin

from AUDVIS import Behavior, AUDVIS, load_in_data
from typing import Literal
from collections import defaultdict

# TODO:
class EncodingModel:
    '''
    Encoding model for neural activity,
    passing params dictionary specifies type of encoding model, 
    otherwise default settings are used

    Input:
    ------------------
    X: design matrix (samples, predictors)
    y: target (samples, 1) - neural activity (1 neuron | population)
    params: dictionary with the following model parameters (if unspecified, default value used)
        'model_type': Literal['GLM', 'XGboost', 'Ensemble', 'RandomForrest', ...]
        'regularization_type': Literal['Ridge', 'L2', 'Lasso', 'L1', 'Elastic-Net', 'L1L2']
        'cross_validation_params': dict()
    '''
    def __init__(self,
                 X:np.ndarray | pd.DataFrame,
                 y:np.ndarray | pd.DataFrame,
                 params:dict | None = None):
        
        if params is None:
            model_type = 'GLM'
            regularization_type = 'Ridge'
            train_test_split:float = 0.8

        self.Xtrain = X[:round(X.shape[0]*train_test_split)]
        self.ytrain = y[:round(y.shape[0]*train_test_split)]

        self.Xtest = X[round(X.shape[0]*train_test_split):]
        self.ytest = y[round(y.shape[0]*train_test_split):]

        self.model = skLin.RidgeCV(alphas = np.logspace(-3,3,20), fit_intercept=True)

    def fit_model(self):
        # TODO: better train test split 
        # (on trials (& random trials to prevent data leakage))
        self.model.fit(X = self.Xtrain, y = self.ytrain)


    def test_model(self) -> np.ndarray:
        return self.model.predict(X = self.Xtest)


def design_matrix(pre_post: Literal['pre', 'post', 'both'] = 'pre'
                  )-> dict[str:dict[str:np.ndarray]]:
    '''
    Builds design matrix for GLM
    '''
    AVs = load_in_data(pre_post=pre_post)
    
    output = defaultdict(lambda : defaultdict(dict))

    for AV in AVs:
        n_sessions = len(AV.session_neurons)
        str_tbs = AV.trials_apply_map(AV.trials, AV.int_to_str_trials_map)
        # Stimulus kernels - same for each session [SESSION-level]
        stim_kernels = stimulus_kernels(tbs=str_tbs, 
                                        nts=AV.signal.shape[1],
                                        SF = AV.SF,
                                        trial_frames=AV.TRIAL,
                                        n_basis=9,
                                        basis_window=(0,1),
                                        basis_width=0.25,
                                        plot_bases=False)
        
        # plt.imshow(stim_kernels[:180, :, 0])
        # plt.show()
        # Behavior kernels - only for sessions with behavioral information [SESSION-level]
        behav_kernels = behavior_kernels(sessions=AV.sessions,
                                        nts=AV.signal.shape[1],
                                        SF = AV.SF,
                                        trial_frames=AV.TRIAL,
                                        n_basis=9,
                                        basis_window=(-0.2,0.3),
                                        basis_width=0.1,
                                        plot_bases=False)
        # plt.imshow(behav_kernels[:180, :, 7])
        # plt.show()

        # TODO? [NEURON-level]
        # Spike history

        # TODO? [Population-level]
        # (COUPLED) Other neurons' spike history
        
        # Save output
        # 3D - ts, predictor, session
        X = np.column_stack((stim_kernels,
                             behav_kernels,
                             ))
        
        # 3D - trial, ts, neuron
        y = AV.baseline_correct_signal(AV.zsig)
        # 4D - (trial, ts, neuron), session
        y = [y[:,:,start:stop]
             for start, stop in AV.session_neurons]


        output[AV.NAME]['X'] = X
        output[AV.NAME]['y'] = y
    
    return output


def stimulus_kernels(tbs:np.ndarray, nts:int, SF:float,
                     trial_frames:tuple[int, int],
                     n_basis:int,
                     basis_window:tuple[float, float],
                     basis_width:float,
                     plot_bases:bool = False)->np.ndarray:
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
    
    # Raised cosine bases 
    # NOTE: (for now, same used for all variables, can VARY for each column)
    CosineBases, frame_lags = rcb(n_basis=n_basis, window_s=basis_window, width_s=basis_width,
                                  dt = 1/SF, plot=plot_bases)
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
        xbases = [getBases(Xcol=Xstim[:,ic], Bases=CosineBases, lags=frame_lags)
                  for ic in range(Xstim.shape[1])]
        Xbases = np.column_stack(xbases)
        X_sess = np.column_stack([Xstim, Xbases])
        all_session_Xs.append(X_sess)
    X = np.dstack(all_session_Xs)
    return X


# TODO: consider extracting more PCs for whisker movement (not just average movement energy)
def behavior_kernels(sessions:dict,
                     nts:int, SF:float,
                     trial_frames:tuple[int, int],
                     n_basis:int,
                     basis_window:tuple[float, float],
                     basis_width:float,
                     plot_bases:bool = False,
                     onset_kernels:bool = False)->np.ndarray:
    '''
    Generates behavior kernels for GLM separately for 
    trial-evoked running, whisking and pupil measurements
        (for B in (running, whisker, pupil)
        B_sig: continuous signal for that behavior
        B_onset_kernel maybe?,
    
    INPUT:
        tbs: trials by session
        nts: number of samples in a trial
        SF: sampling frequency
    '''
    behaviors = [sessions[i]['behavior'] 
                 for i in sorted(sessions.keys())]
    ntrials = behaviors[0].running.shape[0]

    # Raised cosine bases 
    # NOTE: (for now, same used for all variables, can VARY for each column)
    CosineBases, frame_lags = rcb(n_basis=n_basis, window_s=basis_window, width_s=basis_width,
                                  dt = 1/SF, plot=plot_bases)

    all_session_Xb = []
    for isess, BS in enumerate(behaviors):
        Xbeh = np.zeros(shape=(ntrials*nts, 3))
        for ib, bname in enumerate(['running', 'whisker', 'pupil']):
            if hasattr(BS, bname):
                behavior = getattr(BS, bname)
                Xbeh[:,ib] = behavior.flatten()
        
        # have all behaviors, get bases
        xbases = [getBases(Xcol=Xbeh[:,ic], Bases=CosineBases, lags=frame_lags)
                  for ic in range(Xbeh.shape[1])]
        Xbases = np.column_stack(xbases)
        X_sess = np.column_stack([Xbeh, Xbases])
        all_session_Xb.append(X_sess)
    X = np.dstack(all_session_Xb)
    return X


# Raised cosine bases (lag features)
def rcb(n_basis:int, 
        window_s:tuple[float, float], 
        width_s:float, dt:float,
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
            ax.plot(np.linspace(t_min, t_max, len(lags)), phi)

    if plot:
        plt.tight_layout()
        print(lags)
        plt.show() 

    return (np.asarray(B), 
            lags) # array of frame lags


def getBases(Xcol:np.ndarray, 
             Bases:np.ndarray,
             lags:np.ndarray,
             dropFirst:bool = True) -> np.ndarray:
    '''
    Xcol is a 1D array shape (n_trials * n_ts_per_trial)
    Bases is 2D array shape (n_bases, n_lags)

    Returns convolutions of Xcol with provided bases
        The purpose of this is to be able to account for different delays in response
    '''
    basesCols = []
    lag_offset = -lags[0]
    n_ts = Xcol.size
    for ib, basis in enumerate(Bases):
        # if dropFirst and ib == 0:
        #     continue

        # Convolution with the given basis
        Xcol_conv = np.convolve(Xcol, basis)
        # align around trial start
        Xcol_conv_trimmed = Xcol_conv[lag_offset:lag_offset+n_ts]
        basesCols.append(Xcol_conv_trimmed)
    
    Xcol_bases = np.column_stack(basesCols)
    assert Xcol_bases.shape == (Xcol.size, Bases.shape[0])
    return Xcol_bases


def evaluate_encoding_models(gXY, 
                             yTYPE:Literal['neuron', 'population'] = 'population'):
    for ig, (groupName, groupDict) in enumerate(gXY.items()):
        Xall, yall = groupDict['X'], groupDict['y']
        assert isinstance(yall, list) and len(yall) == Xall.shape[-1], 'y should be a list of (trials x ts x neurons) arrays for each session'
        assert all(Xall.shape[0] == yall[i].shape[0] * yall[i].shape[1] for i in range(len(yall))
                   ), 'Entries of y should still be in shape (all_trials, n_ts, neurons)'
        
        for isess in range(len(yall)):
            X = Xall[:,:,isess]
            ysession = yall[isess]

            if yTYPE == 'neuron':
                for ineuron in ysession.shape[-1]:
                    y = ysession[:,:,ineuron].flatten()
                    SessionModel = EncodingModel(X = X, y = y)
                    SessionModel.fit_model()
                    predicted = SessionModel.test_model()

            else: # population-level encoding model
                y = np.nanmean(ysession, axis = 2).flatten()
                SessionModel = EncodingModel(X = X, y = y)
                SessionModel.fit_model()
                predicted = SessionModel.test_model()
                f, ax = plt.subplots()
                ax.plot(SessionModel.ytest, color = 'k', alpha = 0.8, label = 'dF/F data')
                ax.plot(predicted, color = 'magenta', label = 'full model')
                ax.legend(loc = 2)
                f.tight_layout()
                plt.show()



if __name__ == '__main__':
    gXY = design_matrix()
    evaluate_encoding_models(gXY=gXY)
    
