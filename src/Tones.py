import src.matlab as mlb
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, RidgeCV

mats = mlb.find_mat(mlb.TEST)
M = mlb.matgrab(mats[0], fields=mlb.FIELDS)

def standardize_behaviors(MAT:mlb.MatFile, bsl_frames:int
                          )->mlb.MatFile:
    '''
    Assume structure observed in matlab files:
        (timestamp, trial)
    '''
    new_MAT = mlb.MatFile()
    for b in ['running', 'whisker', 'pupil']:
        B = getattr(MAT, b)
        # Z-scoring on global baseline µ and std
        Bmean_global = np.mean(B[:bsl_frames, ...])
        Bstd_global = np.std(B[:bsl_frames, ...])
        Bz_global = (B - Bmean_global) / Bstd_global
        # Baseline-correction on trial-level
        B_trial_evoked = Bz_global - np.mean(Bz_global[:bsl_frames, ...], 
                                             axis = 0, keepdims=True)
        new_MAT.__setattr__(b, B_trial_evoked)
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

def make_toeplitz(v:np.ndarray, ncols:int)->np.ndarray:
    r = np.zeros(ncols)
    r[0] = v[0]
    return toeplitz(v, r)

def make_behavior_predictors(MAT:mlb.MatFile, bsl_frames:int, stim_frames:int,
                             SF:float = 15.4570)-> tuple[np.ndarray, list[str]]:
    '''
    Returns behavior predictors matrix + colnames
    '''
    nts, ntrials = MAT.running.shape
    basic_features  = 12
    features_per_beh = 4*stim_frames
    feature_names = ['running', 'running2', 'running_derivative', 'running_onset', 
                     'whisker', 'whisker2',  'whisker_derivative', 'whisker_onset', 
                     'pupil', 'pupil2', 'pupil_derivative', 'pupil_onset']
    
    BEH = np.zeros((nts*ntrials, basic_features*stim_frames))
    for trl in range(ntrials):
        for ib, b in enumerate(['running', 'whisker', 'pupil']):
            time = trl * nts
            i = ib * features_per_beh
            # grab relevant vector
            # Feature 0 - raw behavior
            beh = getattr(MAT, b)[:, trl]
            BEH[time:time+nts, i:i+stim_frames] = make_toeplitz(beh, stim_frames)
            
            # Feature 1 - squared behavior
            BEH[time:time+nts, i+stim_frames:i+2*stim_frames] = make_toeplitz(np.pow(beh, 2), 
                                                                          stim_frames)
            # Feature 2 - behavior first derivative
            BEH[time:time+nts, i+2*stim_frames:i+3*stim_frames] = make_toeplitz(
                np.diff(beh, prepend=0) * SF, 
                stim_frames)
            
            # Feature 3 - behavior ONSET
            absbeh = np.abs(beh)
            absbeh[:bsl_frames] = 0
            where_max = np.argmax(absbeh)
            max_threshold = int(np.max(absbeh) > 2)
            onset = np.zeros_like(beh)
            onset[where_max] = max_threshold
            BEH[time:time+nts, i+3*stim_frames:i+4*stim_frames] = make_toeplitz(onset, stim_frames)
    
    BEHcolnames = [f'{i}_{fn}'for fn in feature_names for i in range(4)]
    return BEH, BEHcolnames


def design_matrix(MAT:mlb.MatFile)->dict[str:np.ndarray]:
    '''
    Design matrix for one session
    '''
    trial_ts, trial_N, neuron_N = MAT.MLspike.shape
    stim_ts = ((MAT.time >= 0) & (MAT.time <= .5)).sum()
    pre_stim = (MAT.time < 0).sum()

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
    
    # stimulus predictor block (0.5s stimulus presentation, 0.5s offset)
    on_off_ts = 2*stim_ts
    STIM = np.zeros((trial_ts*trial_N, used_Hz_scale.size*(on_off_ts)))
    for trial_i, (hz , db) in enumerate(tts_Hz_dB):
        t = trial_i * trial_ts # time index
        p = hz * on_off_ts # predictor block start
        STIM[t+pre_stim : t+pre_stim+on_off_ts, 
             p : p+on_off_ts] = np.eye(on_off_ts) * db
    
    STIM = (STIM - np.mean(STIM, axis = 0)) / np.std(STIM, axis = 0)
    STIMcolnames = [f'{i}_Tone_{f}HZ' for f in used_Hz_scale for i in range(on_off_ts)]

    # behavior predictor block (both causal and anticausal lag)
    MAT = standardize_behaviors(MAT, pre_stim)
    BEH, BEHcolnames = make_behavior_predictors(MAT, pre_stim, on_off_ts)

    X = np.hstack([STIM, BEH])
    Xcolnames = STIMcolnames + BEHcolnames
    
    # plt.imshow(X[1952:2390, :])    
    # plt.show()

    return X, Xcolnames

        
trial_ts, trial_N, neuron_N = M.MLspike.shape
X, colnames = design_matrix(MAT = M)
time_bad = M.time
time_good = np.where(((-0.5 <= M.time) & (M.time <= 1)))[0]
fullsess_time_good = np.tile(time_good, (trial_N, 1))
trials = np.arange(trial_N) * trial_ts
fullsess_time_good += trials[..., None]
fullsess_time_good = fullsess_time_good.ravel()

bsl = (M.time < 0).sum()

# PGLM = TweedieRegressor(power = 1.5, alpha = 0, link='log', tol = 1e-4) # cross-validate alpha
# ML_blc = M.MLspike - np.mean(M.MLspike[:bsl, ...], axis = 0).astype(int)
# Y = ML_blc[:,:,190].T.ravel()

PGLM = RidgeCV(alphas = np.logspace(-.2,4,30), fit_intercept=True) # cross-validate alpha
dF_blc = M.dF - np.mean(M.dF[:bsl, ...], axis = 0)
Y = dF_blc[:,:,:].mean(axis=2).T.ravel()

PGLM.fit(X[fullsess_time_good, :], Y[fullsess_time_good])
Ypred = PGLM.predict(X[fullsess_time_good, :])

f, ax = plt.subplots()
ax.plot(Y[fullsess_time_good], color = 'k')
# ax.plot(Ypred * np.max(Y) / np.max(Ypred), color = 'r')
ax.plot(Ypred, color = 'g')
f.tight_layout()

plt.show()
