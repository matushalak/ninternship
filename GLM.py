import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltGrid

import pyglmnet as glm
import sklearn.linear_model as skLin

from AUDVIS import Behavior, AUDVIS, load_in_data
from analysis_utils import general_separate_signal
from typing import Literal
from collections import defaultdict

# ----------------------- Encoding model ---------------------------
# TODO: umberto figure 6F
# TODO: include ElasticNet
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
        ...
    '''
    def __init__(self,
                 X:np.ndarray,
                 y:np.ndarray,
                
                 trial_size:int, 
                 stimulus_window:tuple[int, int], 
                 trial_types:np.ndarray,

                 Xcolumn_names:list[str,],

                 params:dict | None = None):
        
        # default parameters
        if params is None:
            model_type = 'GLM'
            regularization_type = 'Ridge'
            train_test_prop:float = 0.8
        else:
            raise NotImplementedError

        # basic params from input arguments
        self.trial_size = trial_size
        self.stimulus_window = stimulus_window
        self.trial_types = trial_types

        # Full design matrix and neural activity vector
        self.X, self.y, self.yTRIAL = X, y, y.reshape((len(y)//trial_size, trial_size))
        self.colnames = Xcolumn_names

        # Train / Test split for quantifications
        (self.Xtrain, self.ytrain), (self.Xtest, self.ytest) = self.train_test_split(
            X=self.X, y=self.y, split_proportion=train_test_prop, trial_size=trial_size, trial_types=trial_types)

        self.model = self.define_model(params=params)


    def define_model(self, params:dict | None = None):
        if params is None:
            return skLin.RidgeCV(alphas = np.logspace(-3,3,20), fit_intercept=True)
        else:
            raise NotImplementedError


    def train_test_split(self, 
                         X:np.ndarray, y:np.ndarray,
                         split_proportion:float,
                         trial_size:int, trial_types:np.ndarray):
        '''
        NOTE on overfitting & train / test split:
        -----------------------------------------
        When trying to simply fit the model to data as good as possible for the purpose of 
        isolating non-whisker-related activity, we "want" to "overfit" and therefore, 
        it does not make sense to do a train / test split, since we do not want to generalize
        (Cross-validation is still carried out within RidgeCV to determine the regularization parameter)

        To make claims about contibutions of Stimulus (V / A / AV) vs Behavior (Run, Whisk, Pupil) to explained variance (EV),
        we need to base the estimated EV on held-out test set.
        
        Similarly, when we want to compare EV between reduced models fit only on (certain) Stimulus / Behavior features, 
        we need to report numbers from held-out test set.
        '''
        all_tt_indices = np.arange(len(trial_types))
        n_train_trials = round(len(trial_types) * split_proportion)
        n_test_trials = len(trial_types) - n_train_trials

        train_i = []
        test_i = []
        # fair split between trial types
        for tt in np.unique(trial_types):
            ttindices = np.where(trial_types == tt)[0]
            n_train_tttrials = round(len(ttindices) * split_proportion)

            train_i.append(train_tt_indices := np.random.choice(ttindices, size = n_train_tttrials, replace=False))
            test_i.append(ttindices[(np.isin(ttindices, train_tt_indices, invert=True))])

        train_i = np.concatenate(train_i)
        test_i = np.concatenate(test_i)
        
        assert len(train_i) == n_train_trials and len(test_i) == n_test_trials, f'Train-test split mismatch Tr:{len(train_i)} instead of {n_train_trials}, Tst:{len(test_i)} instead of {n_test_trials}'
        assert sum(np.isin(test_i, train_i)) == 0, 'Error! Data leakage!'

        # Slice X and y
        Xtrain, ytrain, Xtest, ytest = [], [], [], []
        # Train set
        for start in train_i:
            trial_start = start * trial_size
            trial_stop = trial_start + trial_size
            Xtrain.append(X[trial_start:trial_stop])
            ytrain.append(y[trial_start:trial_stop])
        Xtrain = np.concatenate(Xtrain)
        ytrain = np.concatenate(ytrain)

        # Test set
        for start in test_i:
            trial_start = start * trial_size
            trial_stop = trial_start + trial_size
            Xtest.append(X[trial_start:trial_stop])
            ytest.append(y[trial_start:trial_stop])
        Xtest = np.concatenate(Xtest)
        ytest = np.concatenate(ytest)
        

        assert Xtrain.size == 4*Xtest.size and ytrain.size == 4*ytest.size

        return ((Xtrain, ytrain), 
                (Xtest, ytest))


    # TODO: QR decomposition (within each CV fold separately)
    # maybe not necessary?
    def orthogonalize(self):
        pass


    def fit(self, full:bool = True, 
            X:np.ndarray | None = None, 
            y:np.ndarray | None = None):
        '''
        if X, y are not provided, X, y have to be chosen based on full parameter: 
            full = True (self.X, self.y when best model fit is important)
            full = False (self.X_train, self.y_train when conclusions about % explained variance are important)
        otherwise model is fit to provided X, y
        '''
        match X, y:
            case None, None:
                if full:
                    self.model.fit(X = self.X, y = self.y)
                else:
                    self.model.fit(X = self.Xtrain, y = self.ytrain)
            case X, y:
                self.model.fit(X, y)


    def predict(self, full:bool = True,
                X:np.ndarray | None = None) -> np.ndarray:
        '''
        if X is not provided, X is chosen based on full parameter: 
            full = True (self.X when best model fit is important)
            full = False (self.X_train when conclusions about % explained variance are important)
        otherwise model calculates predictions for provided X
        '''
        match X:
            case None:
                if full:
                    return self.model.predict(X = self.X)
                else:
                    return self.model.predict(X = self.Xtest)
            case X:
                return self.model.predict(X)
    

    def plotGLM(self,
                ypred:np.ndarray, 
                model_label:str = 'full model', 
                col_pred:str = 'magenta', 
                plot_original:bool = True,
                ax_full = None,
                ax_bottom = None,
                fig = None):
        '''
        Produces a plot with continuous trace on top, as well as 4 average subplots below
        (one for Vtrials, one for Atrials, one for AV+ trials, one for AV- trials)
        
        For this structure, assume pltGrid.GridSpec has nrows = 2, ncols = 4.
        '''
        if ax_full is None or ax_bottom is None:
            fig = plt.figure(constrained_layout=True, figsize=(15, 6))
            grid  = fig.add_gridspec(nrows=2, ncols=4)

            # 2.  Create the axes
            ax_full     = fig.add_subplot(grid[0, :]) 
            ax_bottom   = [fig.add_subplot(grid[1, c]) for c in range(4)]

        # Continuous full trace (top row in grid[0,:])
        if plot_original:
            ax_full.plot(self.y, color = 'k', alpha = 0.8, label = 'dF/F data')
        
        ax_full.plot(ypred, color = col_pred, label = model_label)
        
        # 4 bottom
        trial_groups = [(6,7), (0,3), (1,5), (2,4)]
        trial_group_labels = ['V', 'A', 'AV+', 'AV-']

        ypred_trial_locked = ypred.reshape((len(ypred)//self.trial_size, self.trial_size))
        ypred_TT:dict[int:np.ndarray] = general_separate_signal(sig=ypred_trial_locked,
                                                                trial_types=self.trial_types,
                                                                trial_type_combinations=trial_groups,
                                                                separation_labels=trial_group_labels)
        
        if plot_original:
            y_TT = general_separate_signal(sig=self.yTRIAL,
                                        trial_types=self.trial_types,
                                        trial_type_combinations=trial_groups,
                                        separation_labels=trial_group_labels)
        
        for i, (ttname, pred_ttsignal) in enumerate(ypred_TT.items()):
            if plot_original:
                ax_bottom[i].plot(np.mean(y_TT[ttname], axis = 0), color = 'k', alpha = 0.8, label = 'dF/F data')
            
            ax_bottom[i].plot(np.mean(pred_ttsignal, axis = 0), color = col_pred, label = model_label)
            ax_bottom[i].set_title(ttname)
            ax_bottom[i].legend(loc = 2)

        return fig, ax_full, ax_bottom


# --------------------------- DESIGN MATRIX SECTION ---------------------------------
# TODO: z-score each analog column
# TODO: do not convolve bases with continous signals
def design_matrix(pre_post: Literal['pre', 'post', 'both'] = 'pre',
                  show:bool = False)-> dict[str:dict[str:np.ndarray]]:
    '''
    Builds design matrix for GLM
    '''
    AVs = load_in_data(pre_post=pre_post)
    
    output = defaultdict(lambda : defaultdict(dict))
    
    for AV in AVs:
        n_sessions = len(AV.session_neurons)
        str_tbs = AV.trials_apply_map(AV.trials, AV.int_to_str_trials_map)

        # useful for fair train-test split
        trial_types = np.transpose(AV.trials, (1, 0))

        # Trial column (drift over session)
        trials_column = np.tile(np.arange(AV.signal.shape[1]), reps=AV.signal.shape[0] # ntrials
                                ) + (np.repeat(np.arange(AV.signal.shape[0]), 
                                               AV.signal.shape[1]) * AV.signal.shape[1]) # timestemps
        # min-max scaling (between 0-1)
        trial_ramp = (trials_column - trials_column.min()) / (trials_column.max()-trials_column.min())
        trial_ramp = np.dstack([trial_ramp[:, np.newaxis] for _ in range(n_sessions)])

        # Stimulus kernels - same for each session [SESSION-level]
        stim_kernels, stim_col_names = stimulus_kernels(tbs=str_tbs, 
                                                        nts=AV.signal.shape[1],
                                                        SF = AV.SF,
                                                        trial_frames=AV.TRIAL,
                                                        n_basis=9,
                                                        basis_window=(0,1),
                                                        basis_width=0.3,
                                                        plot_bases=show)
        # Stimulus design matrix
        if show:
            plt.imshow(stim_kernels[:180, :, 0])
            plt.show()

        # Behavior kernels - only for sessions with behavioral information [SESSION-level]
        behav_kernels, beh_col_names = behavior_kernels(sessions=AV.sessions,
                                                        nts=AV.signal.shape[1],
                                                        SF = AV.SF,
                                                        trial_frames=AV.TRIAL,
                                                        n_basis=9,
                                                        basis_window=(-0.2,0.3),
                                                        basis_width=0.2,
                                                        plot_bases=show)
        # Behavioral design matrix
        if show:
            plt.imshow(behav_kernels[:180, :, 7])
            plt.show()

        # TODO? [NEURON-level]
        # Spike history

        # TODO? [Population-level]
        # (COUPLED) Other neurons' spike history
        
        # Save output
        # 3D - ts, predictor, session
        X = np.column_stack((trial_ramp,
                             stim_kernels,
                             behav_kernels,
                             ))
        
        Xcolnames = ['trial'] + stim_col_names + beh_col_names
        assert len(Xcolnames) == X.shape[1], f'Mismatch between number of column names:{len(Xcolnames)} and X columns:{X.shape[1]}'

        # Full design matrix
        if show:
            plt.imshow(X[:180, :, 7])
            plt.show()
        
        # 3D - trial, ts, neuron
        y = AV.baseline_correct_signal(AV.zsig)
        # 4D - (trial, ts, neuron), session
        y = [y[:,:,start:stop]
             for start, stop in AV.session_neurons]

        output[AV.NAME]['X'] = X
        output[AV.NAME]['y'] = y

        output[AV.NAME]['Xcolnames'] = Xcolnames
        
        output[AV.NAME]['trial_types'] = trial_types
        output[AV.NAME]['trial_size'] = AV.signal.shape[1]
        output[AV.NAME]['stimulus_window'] = AV.TRIAL
    
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
        # Column names
        if isess == nsessions -1:
            XcolNames = [
                # Visual stimulus columns 0-4
                'Vpresent', 'Vdirection', 'Vposition', 'Vonset', 'Voffset',
                # Auditory stimulus columns 5-9
                'Apresent', 'Adirection', 'Aposition', 'Aonset', 'Aoffset',]

        trials = trs[:,isess]
        # diff features as columns of bigger matrix
        Xstim = np.zeros(shape=(ntrials*nts, 
                                (5 * 2) # 2 modalities of stimuli
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
        
        # All trials done, convolve with bases to account for lags
        xbases = [getBases(Xcol=Xstim[:,ic], Bases=CosineBases, lags=frame_lags)
                  for ic in range(Xstim.shape[1])]
        
        # Get column names for predictor convolved with each basis
        if isess == nsessions - 1:
            for icol, xcol_bases in enumerate(xbases):
                for ibase in range(xcol_bases.shape[1]):
                    XcolNames.append(f'{XcolNames[icol]}_basis{ibase}')

        Xbases = np.column_stack(xbases)
        X_sess = np.column_stack([Xstim, Xbases])
        all_session_Xs.append(X_sess)
    X = np.dstack(all_session_Xs)
    assert X.shape[1] == len(XcolNames), f'Mismatch between number of column names:{len(XcolNames)} and X columns:{X.shape[1]}'
    return X, XcolNames


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
        if isess == len(behaviors) -1:
            XcolNames = []

        Xbeh = np.zeros(shape=(ntrials*nts, 3))
        for ib, bname in enumerate(['running', 'whisker', 'pupil']):
            if isess == len(behaviors) -1:
                XcolNames.append(bname)

            if hasattr(BS, bname):
                behavior = getattr(BS, bname)
                Xbeh[:,ib] = behavior.flatten()
        
        # have all behaviors, convolve with bases
        xbases = [getBases(Xcol=Xbeh[:,ic], Bases=CosineBases, lags=frame_lags,
                           trial_size=nts, trial_level=True)
                  for ic in range(Xbeh.shape[1])]
        
        # Get column names for predictor convolved with each basis
        if isess == len(behaviors) -1:
            for icol, xcol_bases in enumerate(xbases):
                for ibase in range(xcol_bases.shape[1]):
                    XcolNames.append(f'{XcolNames[icol]}_basis{ibase}')

        Xbases = np.column_stack(xbases)
        X_sess = np.column_stack([Xbeh, Xbases])
        all_session_Xb.append(X_sess)
    X = np.dstack(all_session_Xb)
    assert X.shape[1] == len(XcolNames), f'Mismatch between number of column names:{len(XcolNames)} and X columns:{X.shape[1]}'
    return X, XcolNames


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
             trial_size:int = 47, 
             trial_level:bool = False) -> np.ndarray:
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
        if trial_level:
            Xcol_conv_trimmed = []
            for it in np.arange(stop=n_ts, step=trial_size):
                # Convolution with the given basis
                Xcol_conv = np.convolve(Xcol[it:it+trial_size], basis)
                Xcol_conv_trimmed.append(Xcol_conv[lag_offset:lag_offset+trial_size])
            Xcol_conv_trimmed = np.concatenate(Xcol_conv_trimmed)
        
        else:
            # Convolution with the given basis
            Xcol_conv = np.convolve(Xcol, basis)
            # align around trial start
            Xcol_conv_trimmed = Xcol_conv[lag_offset:lag_offset+n_ts]
        
        basesCols.append(Xcol_conv_trimmed)
    
    Xcol_bases = np.column_stack(basesCols)
    assert Xcol_bases.shape == (Xcol.size, Bases.shape[0])
    return Xcol_bases


# -------------- running Models -------------------
# TODO: in each subplot (Original signal, 
# full model, Stimulus predictors (V, A, AV), Behavioral predictors (Whisk, Run, Pupil)
def decompose(Model:EncodingModel, motor_together:bool = True
              )->dict[str:np.ndarray]:
    '''
    This function needs to be modified based on the 
    desired decomposition of design matrix in Model
    NOTE: assumes linear model which can be decomposed easily!
    '''
    decomposed_drives = dict()
    if motor_together:
        components = ['V', 'A', 'AV', 'Motor']
    else:
        components = ['V', 'A', 'AV', 'Running', 'Whisker', 'Pupil']
    
    for component in components:
        match component:
            case 'V':
                components_i = [True if 'V' in colName 
                                else False for colName in Model.colnames]
            case 'A':
                components_i = [True if 'A' in colName 
                                else False for colName in Model.colnames]
            case 'AV':
                components_i = [True if ('V' in colName or 
                                        'A' in colName) 
                                else False for colName in Model.colnames]
            case 'Motor':
                components_i = [True if ('running' in colName or 
                                        'whisker' in colName or 
                                        'pupil' in colName) 
                                else False for colName in Model.colnames]
            
            # for direct behavioral components, the colnames include component name
            case _:
                components_i = [True if component.lower() in colName 
                                else False for colName in Model.colnames]
        
        X_component = Model.X[:, components_i]
        coefficients_component = Model.model.coef_[components_i]
        component_drive = X_component @ coefficients_component
        decomposed_drives[component] = component_drive
    
    return decomposed_drives


def evaluate_encoding_models(gXY, 
                             yTYPE:Literal['neuron', 'population'] = 'population'):
    for ig, (groupName, groupDict) in enumerate(gXY.items()):
        Xall, yall = groupDict['X'], groupDict['y']
        TTall, trial_size, stimulus_window = (groupDict['trial_types'], 
                                              groupDict['trial_size'], 
                                              groupDict['stimulus_window'])
        Xcolnames = groupDict['Xcolnames']

        assert isinstance(yall, list) and len(yall) == Xall.shape[-1], 'y should be a list of (trials x ts x neurons) arrays for each session'
        assert all(Xall.shape[0] == yall[i].shape[0] * yall[i].shape[1] for i in range(len(yall))
                   ), 'Entries of y should still be in shape (all_trials, n_ts, neurons)'
        
        for isess in range(len(yall)):
            X = Xall[:,:,isess]
            ysession = yall[isess]
            TTsession = TTall[:,isess]

            if yTYPE == 'neuron':
                for ineuron in range(ysession.shape[-1]):
                    y = ysession[:,:,ineuron].flatten()
                    SessionModel = EncodingModel(X = X, y = y, 
                                                 Xcolumn_names=Xcolnames,
                                                 trial_types=TTsession, 
                                                 trial_size=trial_size, stimulus_window=stimulus_window,
                                                 params=None)
                    SessionModel.fit(full=True)
                    predicted = SessionModel.predict(full=True)
    
                    pred_components:dict[str:np.ndarray] = decompose(Model=SessionModel,
                                                                 motor_together=True)
                    component_colors = ['dodgerblue', 'red', 'goldenrod', 'green']

                    f, axfull, axbottom = SessionModel.plotGLM(ypred=predicted, plot_original=True)
                    for ic, (component_name, component_signal) in enumerate(pred_components.items()):
                        SessionModel.plotGLM(ypred=component_signal, plot_original=False,
                                            model_label=component_name, 
                                            col_pred=component_colors[ic],
                                            ax_full=axfull, ax_bottom=axbottom, fig = f)
                    
                    f.tight_layout()
                    plt.show()
                    plt.close()

            else: # population-level encoding model
                y = np.nanmean(ysession, axis = 2).flatten()
                SessionModel = EncodingModel(X = X, y = y,
                                             Xcolumn_names=Xcolnames,
                                             trial_types=TTsession, 
                                             trial_size=trial_size, stimulus_window=stimulus_window,
                                             params=None)
                SessionModel.fit(full=True)
                predicted = SessionModel.predict(full=True)
                pred_components:dict[str:np.ndarray] = decompose(Model=SessionModel,
                                                                 motor_together=True)
                component_colors = ['dodgerblue', 'red', 'goldenrod', 'green']

                f, axfull, axbottom = SessionModel.plotGLM(ypred=predicted, plot_original=True)
                for ic, (component_name, component_signal) in enumerate(pred_components.items()):
                    SessionModel.plotGLM(ypred=component_signal, plot_original=False,
                                        model_label=component_name, 
                                        col_pred=component_colors[ic],
                                        ax_full=axfull, ax_bottom=axbottom, fig = f)
                
                f.tight_layout()
                plt.show()
                plt.close()



if __name__ == '__main__':
    gXY = design_matrix()
    evaluate_encoding_models(gXY=gXY, yTYPE='neuron')
    
