import src.matlab as mlb
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from src.GLM_new import behavior_kernels, rcb, getBases

def standardize_behaviors(MAT:mlb.MatFile, bsl_frames:int,
                          time_slice:list[int, int]|None = None
                          )->mlb.MatFile:
    '''
    Assume structure observed in matlab files:
        (timestamp, trial)
    '''
    new_MAT = mlb.MatFile()
    if time_slice is not None:
        # slice excludes upper bound
        time_slice[1] += 1

    for b in ['running', 'whisker', 'pupil']:
        B = getattr(MAT, b)
        if time_slice is not None:
            B = B[slice(*time_slice), :]
        # Z-scoring on global baseline µ and std
        Bmean_global = np.mean(B[:bsl_frames, ...])
        Bstd_global = np.std(B[:bsl_frames, ...])
        Bz_global = (B - Bmean_global) / Bstd_global
        # Baseline-correction on trial-level
        B_trial_evoked = Bz_global - np.mean(Bz_global[:bsl_frames, ...], 
                                             axis = 0, keepdims=True)
        # XXX transpose to work with my other GLM code expecting (trials, timestamps) matrices
        new_MAT.__setattr__(b, B_trial_evoked.T) 
    return new_MAT

def tt_to_freq_db(tt:int, TTexpl:list[str]
                  )->tuple[int, int]:
    '''
    Returns (X Hz, Y db) tuple for each tt 
    '''
    # get tt'th TT description
    description = TTexpl[tt]
    # regex pattern matching
    match = re.match(r"^PureTone(\d+)Hz_octaveStep(\d)_(\d+)dB$", description)
    assert match, f'No match for Frequency, Octave Step and Intensity found in {description}'

    Hz, octStep, dB = match.groups()

    # Octave = 2 x frequency; Half an octave = sqrt(2) x frequency
    half_octave = 2**(1/2)
    Hz = float(Hz) * (half_octave**(float(octStep)))
    return (int(Hz), int(dB))

def tone_predictors(tbs:np.ndarray, nts:int, SF:float,
                    trial_frames:tuple[int, int],
                    n_basis:int,
                    basis_window:tuple[float, float],
                    basis_width:float,
                    plot_bases:bool = False)->np.ndarray:
    ''''
    2 s baseline (30 frames)
    0.5 s stimulus (8 frames)
    2.5 s post-stim (40 frames)

    Equalize pre & post stimuli: 2s pre; 0.5 stim, 1.5 post
    '''
    ntrials, ntrial_features = tbs.shape
    trial_frames = np.array(trial_frames)
    nstim_ts = trial_frames[1] - trial_frames[0]
    combined_stim_response_window = nstim_ts + n_basis
    stim_responses = np.eye(combined_stim_response_window)
    CosineBases, frame_lags = rcb(n_basis=combined_stim_response_window, 
                                  window_s=basis_window, width_s=basis_width,
                                  dt = 1/SF, plot=plot_bases)
    response_matrix = getBases(Xcol = stim_responses[:,0], Bases = CosineBases, lags = frame_lags, 
                               trial_size = nts, trial_level = False)
    if plot_bases: plt.imshow(response_matrix);plt.show()
    frequency_blocks = np.unique(tbs[:, 0])
    STIMcolnames = [f'{i}_Tone_{f}HZ' for f in frequency_blocks for i in range(combined_stim_response_window)]
    # stimulus predictor block (0.5s stimulus presentation, 1.5s offset)
    STIM = np.zeros((nts*ntrials, frequency_blocks.size*(combined_stim_response_window)))
    for trial_i, (hz , db) in enumerate(tbs):
        t = int(trial_i * nts) # time index
        p = int(hz * combined_stim_response_window) # predictor block start
        STIM[t+trial_frames[0] : t+nts, 
             p : p+combined_stim_response_window] = response_matrix * (db+1) #NOTE if categorical then dB+1
    
    assert len(STIMcolnames) == STIM.shape[1]
    if plot_bases: plt.imshow(STIM[:1000,:]); plt.show()
    return STIM, STIMcolnames

def tone_tts_to_grid(tts_hz_db:np.ndarray)->dict[tuple:list[int]]:
    out = dict()
    for itt, (hz, db) in enumerate(tts_hz_db):
        if (db, hz) in out:
            out[(db, hz)] = np.append(out[(db, hz)], itt)
        else:
            out[(db, hz)] = np.array([itt,])
    return out

def design_matrix(MAT:mlb.MatFile,
                  SF:float = 15.4570,
                  trial_duration_sec:float = 0.5,
                  window_sec:tuple[float, float] = (-2, 2.07),
                  plot:bool = False)->tuple[np.ndarray]:
    '''
    Design matrix for one session
    '''
    trial_ts, trial_N, neuron_N = MAT.dF.shape
    pre_stim = ((MAT.time >= window_sec[0]) & (MAT.time < 0))
    pre_stim_ts = (min(np.where(pre_stim)[0]), max(np.where(pre_stim)[0]))
    stim = ((MAT.time >= 0) & (MAT.time <= .5))
    stim_ts = (min(np.where(stim)[0]), max(np.where(stim)[0]))
    post_stim = ((MAT.time > trial_duration_sec) & (MAT.time <= window_sec[1]))
    post_stim_ts = (min(np.where(post_stim)[0]), max(np.where(post_stim)[0]))
    TRIAL_FRAMES = (pre_stim.sum(), (pre_stim | stim).sum())
    trial_ts = (pre_stim | stim | post_stim).sum()
    
    # XXX REDO predictor block
    TTexp = MAT.TT_explain
    tt_num = (MAT.TTs - 1).astype(int) # correct for matlab 1 indexing
    # this contains the actual Hz and dB of each pure tone in each trial
    tts_Hz_dB = np.array([tt_to_freq_db(tt, TTexp) for tt in tt_num]) # (trials, [Hz, dB])
    
    # since frequencies will be different predictors, more useful to split into 12 categories
    Hz = tts_Hz_dB[:,0]
    used_Hz_scale= np.unique(Hz)
    Hz_cat = np.searchsorted(used_Hz_scale, Hz)
    # replace Hz with frequency predictor
    tts_Hz_dB[:,0] = Hz_cat

    # also categories for dB
    dB = tts_Hz_dB[:,1].copy()
    used_dB_scale = np.unique(dB)
    dB_cat = np.searchsorted(used_dB_scale, dB)
    # replace dB with loudness predictor
    tts_Hz_dB[:, 1] = dB_cat

    # trials grid, rows:intensity, cols:frequency
    # TODO: when plotting tts_Hz_dB[:, ::-1]
    # ax[tts_Hz_dB[trial, ::-1]]
    grid_tts = tone_tts_to_grid(tts_Hz_dB)
    # tts_Hz_dB[:, 1] = dB
    
    STIM, STIMcolnames = tone_predictors(tbs=tts_Hz_dB,
                                         nts = trial_ts,
                                         SF = SF,
                                         trial_frames=TRIAL_FRAMES,
                                         n_basis=post_stim.sum(),
                                         basis_window=(0, window_sec[1]),
                                         basis_width=0.35,
                                         plot_bases=plot)

    # behavior predictor block (both causal and anticausal lag)
    MAT = standardize_behaviors(MAT, pre_stim.sum(), 
                                time_slice=[pre_stim_ts[0], post_stim_ts[1]])
    # XXX use behavior kernels from AV
    BEH, BEHcolnames = behavior_kernels(sessions = {0:{'behavior':MAT}},
                                        nts=trial_ts,  
                                        SF = SF,
                                        trial_frames=TRIAL_FRAMES,
                                        n_onset_basis= (stim | post_stim).sum(),
                                        n_continuous_lags=10,
                                        basis_window=(-0.35, 1.05),
                                        onset_basis_width=0.35,
                                        cont_basis_width=0.5,
                                        THRESHOLD=2,
                                        plot_bases=plot)
    if plot:
        plt.imshow(BEH[:200, :, 0]); plt.show()
    
    X = np.hstack([STIM, BEH.squeeze()])
    Xcolnames = STIMcolnames + BEHcolnames
    if plot:
        plt.imshow(X[1952:2390, :]); plt.show()

    return X, Xcolnames, tts_Hz_dB, grid_tts

def runTonesGLM(M:mlb.MatFile,
                window_sec:tuple[float, float],
                SF:float = 15.4570,
                trial_dur:float = 0.5
                ):
    trial_ts, trial_N, neuron_N = M.dF.shape
    X, colnames, tts_Hz_dB, grid_tt = design_matrix(MAT = M, 
                                                    trial_duration_sec=trial_dur,
                                                    window_sec=window_sec,
                                                    plot=False)
    startSec, endSec = window_sec
    rows, cols = len(np.unique(tts_Hz_dB[:, 1])), len(np.unique(tts_Hz_dB[:, 0]))

    tone_cols  = np.array([True if 'Tone' in col else False for col in colnames])
    beh_cols = ~tone_cols
    time_bad = M.time
    time_good_idx = np.where(((startSec <= M.time) & (M.time <= endSec)))[0]
    fullsess_time_good = np.tile(time_good_idx, (trial_N, 1))
    trials = np.arange(trial_N) * len(time_good_idx)
    fullsess_time_good += trials[..., None]
    fullsess_time_good = fullsess_time_good.ravel()

    bsl = (M.time[time_good_idx] < 0).sum()
    bsl_idx = np.where(((startSec <= M.time) & (M.time <= 0)))[0]
    # baseline-correct
    dF_blc = M.dF - (M.dF[bsl_idx, ...].mean(axis = 0, keepdims=True))
    
    f, ax = plt.subplots(nrows=rows, ncols = cols, figsize = (24,8))
    for tt, tttrials in grid_tt.items():
        # average over trials for all neurons
        neurons_by_tt = dF_blc[time_good_idx[:, None], tttrials[None, :], :].mean(axis = 1)
        ax[tt].plot(neurons_by_tt) 
        ax[tt].set_ylim(-1,2)

    f.tight_layout(); plt.show()
        
    # average over neurons for all trials
    # plt.plot(dF_blc[time_good_idx, ...].mean(axis = 2)); plt.show()

    GLM = RidgeCV(alphas = np.logspace(-.2,4,30), fit_intercept=True) # cross-validate alpha
    
    for n in range(neuron_N):
        # per single neuron
        Y = dF_blc[..., n].T.ravel()

        GLM.fit(X, Y[fullsess_time_good])
        coefs = GLM.coef_
        print(X.shape, tone_cols.shape, coefs.shape)
        tone_coef = coefs[tone_cols]
        Xtone = X[:, np.where(tone_cols)[0][None, :]]

        beh_coef = coefs[beh_cols]
        Xbeh = X[:, np.where(beh_cols)[0][None, :]]

        Ypred = GLM.predict(X)
        Ybeh = Xbeh @ beh_coef
        Ytone = Xtone @ tone_coef

        f, ax = plt.subplots(nrows=rows, ncols = cols, figsize = (24,8))
        # ax[0].plot(Y[fullsess_time_good], color = 'k')
        # ax[0].plot(Ypred, color = 'magenta')
        # ax[0].plot(Ybeh, color = 'green')
        # ax[0].plot(Ytone, color = 'red')

        good_ts = len(time_good_idx)
        y = Y[fullsess_time_good].reshape((trial_N, good_ts))
        ypred = Ypred.reshape((trial_N, good_ts))
        ybeh = Ybeh.reshape((trial_N, good_ts))
        ytone = Ytone.reshape((trial_N, good_ts))
        
        for tt, tttrials in grid_tt.items():
            ax[tt].plot(y[tttrials, :].mean(axis=0), color='k')
            ax[tt].plot(ypred[tttrials, :].mean(axis=0), color='magenta')
            ax[tt].plot(ybeh[tttrials, :].mean(axis=0), color='green')
            ax[tt].plot(ytone[tttrials, :].mean(axis=0), color='red')
            ax[tt].set_ylabel('Ylabel')
            ax[tt].set_xlabel('Xlabel')

        f.tight_layout()
        plt.show()

if __name__ == '__main__':
    mats = mlb.find_mat(mlb.TEST)
    M = mlb.matgrab(mats[0], fields=mlb.FIELDS)
    runTonesGLM(M, window_sec=(-0.5, 1.0))

    