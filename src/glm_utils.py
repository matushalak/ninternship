import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import atexit
from tqdm import tqdm

import multiprocessing as MP
from joblib import Parallel, delayed
from multiprocessing import shared_memory

from src.analysis_utils import general_separate_signal
from src.AUDVIS import Behavior, AUDVIS

from dataclasses import dataclass
from typing import Literal

import src.GLM as glm
from src import PYDATA


# TODO: implement EV shuffling and saving weights / neuron
# here we only care about held-out EV distributions
# X in shared memory
# y has to be unique since shuffled

# TODO: integrate loader in other GLM functions (not necessary, just more elegant)
# QUITE A MESS NOW
def loader_(group_name:str | None = None, 
            pre_post: Literal['pre', 'post', 'both'] = 'pre',
            yTYPE:Literal['neuron', 'population'] = 'neuron',
            clean_signal:bool = False,
            exportDrives:bool = False,
            exportWeights:bool = False,
            EV:bool = False,
            redo:bool = False,
            storage_folder:str = PYDATA,
            ):
    '''
    This function tries to load and return the requested files if possible, 
    otherwise returns False
    '''
    match clean_signal, EV:
        # clean_signal
        case True, False: 
            if os.path.exists(sigpath := os.path.join(storage_folder, 
                                        f'{group_name}_{yTYPE}_signal_GLMclean.npy')
                            ) and not redo:
                if not exportDrives:    
                    SIG_CLEAN = np.load(sigpath)
                    print(f'Loaded signal from {sigpath}')
                    return SIG_CLEAN
                else:
                    drive_files = [f for f in os.listdir(storage_folder) 
                                if 'drive_GLMclean.npy' in f]
                    # assume naming convention here (that I implement when saving files below)
                    driveDict = dict()
                    for drive_file in drive_files:
                        allnameparts = drive_file.split('_')
                        component_name = allnameparts[2]
                        driveDict[component_name] = np.load(drive_file)

                    print(f'Loaded drive signal dictionary from from {drive_files} (run function separately without export_drives to load only cleaned signal)')
                    return driveDict
            else:
                print(sigpath, 'does not exist, running clean_signal!')
                return None
        
        # collect explained variance results
        case False, True:
            savepathEV=os.path.join(storage_folder, 
                                    f'GLM_ev_results_{yTYPE}.csv')
            # zipped compressed numpy file
            savepathWEIGHTS=os.path.join(storage_folder, 
                                         f'GLM_shuffle_weights_{yTYPE}.npz')

            if not exportWeights:
                if os.path.exists(savepathEV) and not redo:
                    print(f'Returning explained variance results stored in :{savepathEV}!')
                    return pd.read_csv(savepathEV, index_col=False)
                else:
                    print('No explained variance results, running explain_variance!')
                    return None
            
            else:
                if os.path.exists(savepathWEIGHTS) and not redo:
                    print(f'Returning shuffled weights stored in :{savepathWEIGHTS}!')
                    # This is a dictionary with shuffled data for each group, index accordingly
                    weights_dict:np.lib.npyio.NpzFile = np.load(savepathWEIGHTS)
                    if group_name is None:
                        return [weights_dict[g] for g in weights_dict.keys()]
                    else:
                        return weights_dict[group_name]
                else:
                    print('No shuffled weights file, running explain_variance (will also produce shuffled weights file)!')
                    return None
        
        case _:
            raise AssertionError('Choose whether you want to export clean signals / drives OR if you want to export the EV dataframe / shuffled weights')


@dataclass
class ShuffleData:
    Xtrain:tuple
    Xtest:tuple
    Ytrain:tuple
    Ytest:tuple
    TT:tuple
    Xcolnames:list[str]
    trial_size:int
    stimulus_window:tuple[int,int]
    sess_i:int
    group_i:int


# TODO: generalize for clean signal and EV
def encodingLOOP_(groupName:str, groupDict:dict, 
                  storage_folder:str,
                  n_shuffles:int = 1000,
                  redo:bool = False):
    shuffle_file = os.path.join(storage_folder, f'{groupName}_EVshuffle_distribution.npy')
    if os.path.exists(shuffle_file) and not redo:
        print(f'Loading {shuffle_file}!')
        return np.load(shuffle_file)
    
    Xall, yall = groupDict['X'], groupDict['y']
    TTall, trial_size, stimulus_window = (groupDict['trial_types'], 
                                            groupDict['trial_size'], 
                                            groupDict['stimulus_window'])
    Xcolnames = groupDict['Xcolnames']

    assert isinstance(yall, list) and len(yall) == Xall.shape[-1], 'y should be a list of (trials x ts x neurons) arrays for each session'
    assert all(Xall.shape[0] == yall[i].shape[0] * yall[i].shape[1] for i in range(len(yall))
                ), 'Entries of y should still be in shape (all_trials, n_ts, neurons)'
    
    component_colors = {'V':'dodgerblue', 'A':'red', 
                        'AV':'goldenrod', 'Motor':'green'}
    
    GROUP_RESULTS = []
    for isess in range(len(yall)):
        # get session data
        X = Xall[:,:,isess]
        ysession = yall[isess]
        TTsession = TTall[:,isess]

        print(f'Processing group {groupName} session {isess} ...')
        # make one model for random neuron to extract the train / test split
        TRAIN, TEST = glm.EncodingModel.train_test_split(
            X = X, y = ysession[:,:,0].flatten(), 
            split_proportion=0.8,
            trial_size=trial_size,
            trial_types=TTsession,
            )
        
        XTRAIN, _, TRAINtrials = TRAIN
        XTEST, _, TESTtrials = TEST
        
        # Shared memory design matrix
        shmXtrain = shared_memory.SharedMemory(create=True, size = XTRAIN.nbytes)
        Xtrainsh = np.ndarray(XTRAIN.shape, dtype=XTRAIN.dtype, buffer = shmXtrain.buf)
        Xtrainsh[:] = XTRAIN

        shmXtest = shared_memory.SharedMemory(create=True, size = XTEST.nbytes)
        Xtestsh = np.ndarray(XTEST.shape, dtype=XTEST.dtype, buffer = shmXtest.buf)
        Xtestsh[:] = XTEST

        # Shared memory neural signals
        YTRAIN, YTEST = ysession[TRAINtrials, :, :], ysession[TESTtrials, :, :]
        shmYtrain = shared_memory.SharedMemory(create=True, size = YTRAIN.nbytes)
        Ytrainsh = np.ndarray(YTRAIN.shape, dtype=YTRAIN.dtype, buffer = shmYtrain.buf)
        Ytrainsh[:] = YTRAIN

        shmYtest = shared_memory.SharedMemory(create=True, size = YTEST.nbytes)
        Ytestsh = np.ndarray(YTEST.shape, dtype=YTEST.dtype, buffer = shmYtest.buf)
        Ytestsh[:] = YTEST
        
        TTtest = TTsession[TESTtrials]
        shmTT = shared_memory.SharedMemory(create=True, size = TTtest.nbytes)
        TTsessionsh = np.ndarray(TTtest.shape, dtype=TTtest.dtype, buffer = shmTT.buf)
        TTsessionsh[:] = TTtest

        # if program crashes no leaked memory
        atexit.register(lambda: glm.safeMPexit([shmXtrain, shmXtest, shmYtest, shmYtrain, shmTT]))

        SD = ShuffleData((shmXtrain.name, Xtrainsh.shape, Xtrainsh.dtype), 
                        (shmXtest.name, Xtestsh.shape, Xtestsh.dtype), 
                        (shmYtrain.name, Ytrainsh.shape, Ytrainsh.dtype), 
                        (shmYtest.name, Ytestsh.shape, Ytestsh.dtype),
                        (shmTT.name, TTsessionsh.shape, TTsessionsh.dtype),
                        Xcolnames, trial_size, stimulus_window,
                        isess, group_i=groupName)
        
        neuron_indices = np.arange(ysession.shape[-1])
        # TODO: consider batched if single 1000 model fits too fast
        neuron_batches = np.array_split(neuron_indices, MP.cpu_count())
        neuron_batches = [b for b in neuron_batches if len(b) != 0]
        worker_args = [(SD, nb, i, n_shuffles) 
                       for i, nb in enumerate(neuron_batches)]
        
        # use get shuffled_dist here
        session_res = Parallel(n_jobs=MP.cpu_count(), backend='threading'
                               )(delayed(lightweight_EV_worker)(*wa) 
                                 for wa in worker_args)
        
        GROUP_RESULTS.append(np.concatenate(session_res, axis = 3))

        shmXtrain.close(); shmXtest.close(); shmYtest.close(); shmYtrain.close(); shmTT.close()
        shmXtrain.unlink(); shmXtest.unlink(); shmYtest.unlink(); shmYtrain.unlink(); shmTT.unlink()
    
    # Final group output
    GROUP_RESULTS = np.concatenate(GROUP_RESULTS, axis = 3)
    print(f'Saved shuffle distribution in {shuffle_file}')
    np.save(shuffle_file, GROUP_RESULTS)
    return GROUP_RESULTS


def lightweight_EV_worker(sess:ShuffleData, # FIXED, shared between workers,
                          neuron_batch:np.ndarray,
                          ibatch:int, 
                          nshuffles:int
                          ):
    '''
    NOTE:
    Here I implement the 'fixed weights' shuffle test, where I only shuffle the test signal
    and keep the weights from the original fit to the real Ytrain. 
    This is computationally efficient and produces a null distribution of the EV metric, 
    for each predictor block, which we care about.
    A more stringent way is to refit the model at every shuffle, by shuffling Ytrain and keeping Ytest intact. 
    This is very slow and would take > 5 days to run for 4000 neurons and 1000 shuffles.
    '''
    # unpack shared memory inside each worker
    (Xtrainname, Xtrainshape, Xtraindtype) = sess.Xtrain
    (Xtestname, Xtestshape, Xtestdtype) = sess.Xtest 
    (Ytrainname, Ytrainshape, Ytraindtype) = sess.Ytrain
    (Ytestname, Ytestshape, Ytestdtype) = sess.Ytest 
    (shmTTname, TTshape, TTdtype) = sess.TT

    shmXtrain = shared_memory.SharedMemory(name=Xtrainname)
    shmXtest = shared_memory.SharedMemory(name=Xtestname)
    shmYtrain = shared_memory.SharedMemory(name=Ytrainname)
    shmYtest = shared_memory.SharedMemory(name=Ytestname)
    shmTT = shared_memory.SharedMemory(name=shmTTname)

    XTRAIN = np.ndarray(shape=Xtrainshape, dtype=Xtraindtype, buffer=shmXtrain.buf)
    XTEST = np.ndarray(shape=Xtestshape, dtype=Xtestdtype, buffer=shmXtest.buf)
    YTRAIN = np.ndarray(shape=Ytrainshape, dtype=Ytraindtype, buffer=shmYtrain.buf)
    YTEST = np.ndarray(shape=Ytestshape, dtype=Ytestdtype, buffer=shmYtest.buf)
    TT = np.ndarray(shape=TTshape, dtype=TTdtype, buffer=shmTT.buf) # only test-set trial-types

    res_collection = []
    for neuron in tqdm(neuron_batch, position=ibatch):    
        # Expensive 1 real fit per neuron
        LightWeightModel = glm.EncodingModel(smart=False)
        # Train on shuffled Ytrain
        LightWeightModel.fit(X = XTRAIN, y = YTRAIN[:,:,neuron].flatten())
        # Test on unshuffled Ytest
        # because of this, what comes out of decompose will be drives on test data
        LightWeightModel.X = XTEST 
        drives = glm.decompose(LightWeightModel, colnames=sess.Xcolnames)
        drives['Model'] = LightWeightModel.predict(X = XTEST) # Full model

        # Fast EV calculation for 1000 shuffles
        # Explained variance metric
        EV : function = lambda ypred, truth: 1 - (np.var(truth - ypred)/np.var(truth))
        
        # NOTE: This has to be changed based on experiment!!!
        trial_groups = [(6,7), (0,3), (1,5), (2,4)]
        trial_group_labels = ['V', 'A', 'AV+', 'AV-']
        
        # random number generator with a reproducible seed
        RNG : np.random.Generator = np.random.default_rng(seed = 2025)
        # Get the random circular shuffles
        flat = YTEST[:,:,neuron].flatten()
        shuffles = RNG.choice(a = flat.shape[0], size=nshuffles)

        # 1000 shuffles (reshaped into trial-locked)
        YTESTshuf = np.column_stack([np.roll(flat, shift=s) for s in shuffles]
                                    ).reshape((YTEST.shape[0], YTEST.shape[1], -1))
        
        # calculate explained variance and return 3D numpy array (tt x predictors x shuffles)
        # + generates and saves explanation dictionary as pickle file
        res = glm.explained_variance(target_trial_locked=YTESTshuf, pred_components=drives,
                                    TTsession=TT, trial_size=sess.trial_size,
                                    exportAS=np.ndarray)
        
        res_collection.append(res)
    
    # 4D numpy array (separate 3D array for each neuron)
    # (tt x predictors x shuffles x neurons)
    out = np.stack(res_collection, axis=3)
    return out


def use_encoding_model(group_name:str = 'both',
                       pre_post: Literal['pre', 'post', 'both'] = 'pre',
                       yTYPE:Literal['neuron', 'population'] = 'neuron',
                       clean_signal:bool = False,
                       exportDrives:bool = False,
                       exportWeights:bool = False,
                       EV:bool = False,
                       redo:bool = False,
                       storage_folder:str = PYDATA,
                       n_shuffles:int = 1000
                       ):
    # If used to Load already saved files
    loaded = loader_(group_name, pre_post, yTYPE, 
                    clean_signal, exportDrives, exportWeights, 
                    EV, redo, storage_folder)
    
    if loaded is not None:
        return loaded
    
    # Otherwise start use of encoding model process
    # 1) design matrix (containing signal) for chosen group
    gXY:dict = glm.design_matrix(pre_post=pre_post, group=group_name)

    for ig, (groupName, groupDict) in enumerate(gXY.items()):
        groupRES = encodingLOOP_(groupName, groupDict, 
                                 storage_folder, 
                                 n_shuffles=n_shuffles,
                                 redo=redo)


def drives_loader(group_name:str, storage_folder:str = PYDATA,):
    drive_files = [f for f in os.listdir(storage_folder) 
                    if 'drive_GLMclean.npy' in f and group_name in f]
    # assume naming convention here (that I implement when saving files in GLM.py)
    driveDict = dict()
    for drive_file in drive_files:
        allnameparts = drive_file.split('_')
        component_name = allnameparts[2]
        driveDict[component_name] = np.load(os.path.join(storage_folder, drive_file))

    return driveDict


if __name__ == '__main__':
    use_encoding_model(EV = True, redo=True)