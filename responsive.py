# @matushalak
# For parallelized execution of Zeta test to determine responsive units
from SPSIG import SPSIG, Dict_to_Class
from AUDVIS import Behavior
from utils import group_condition_key, progress_bar
from matplotlib_venn import venn3
from collections import defaultdict
from pandas import DataFrame
from zetapy import zetatstest, zetatstest2 # test for neuronal responsiveness by Montijn et al.
import numpy as np
import multiprocessing as mp
import os
import pickle
import argparse

def run_ZETA(signals:np.ndarray,
             frame_times_corrected:np.ndarray,
             event_IDs:np.ndarray,
             event_times:np.ndarray,
             maxDur:float = 1.4):
    '''
    inputs:
    Runs zeta test for each ROI for each trial type in a session
        signals : ndarray (n_neurons, all_times) =  contains the full-recording signals for all neurons 
        frame_times_corrected : ndarray (n_neurons, all_times) = contains corrected frame times for each ROI
        event_IDs : ndarray (n_trials, ) = IDs for all trials
        event_times : ndarray (n_trials, ) = event times for each trial
    ---------------
    outputs:
    results : list[int:dict] = list of dictionaries with Zeta test results for each neuron for each condition
    '''
    results = [dict() for _ in range(signals.shape[0])]
    ttwhere = {tt : np.where(event_IDs == tt)
                for tt in np.unique(event_IDs)}
    
    # progress bar to not go crazy
    total_iterations = len(list(ttwhere)) * signals.shape[0]
    iteration = 0
    for tt in list(ttwhere):
        for neuron in range(signals.shape[0]):  
            _, res = zetatstest(vecTime = frame_times_corrected[neuron,:],
                                vecValue = signals[neuron,:],
                                arrEventTimes = event_times[ttwhere[tt]],
                                dblUseMaxDur=maxDur)
            # get rid of most returned arrays that is just memory BLOAT
            Res_stat = {'zetaP':res['dblZetaP'],
                        'ZETA':res['dblZETA']}
            results[neuron][tt] = Res_stat # results for that trial type
            progress_bar(iteration, total_iterations)
            iteration += 1
    print(f'Zeta test for {signals.shape[0]} neurons and {len(list(ttwhere))} trial types.')
    return results
    
def prepare_zeta_params(spsig_file:str,
                        regress_OUT : bool = True
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Prepares parameters for zeta test for all ROIs in a given session
    '''
    res_file = spsig_file[:-4] + '_Res.mat'
    SPSG = SPSIG(spsig_file)
    RES = SPSIG(res_file)
    resres : Dict_to_Class = RES.Res
    # all sessions should have running speed and only 3 should lack facemap
    print(res_file.split('/')[-1], 'facemap', hasattr(resres, 'facemap'), 'speed', hasattr(resres, 'speed'))
    
    # get RAW running speed and whisker movements
    if regress_OUT:
        parts_of_file = spsig_file.split('/')[-1].split('_')
        quadrature_file = '_'.join(parts_of_file[:3]) + '_quadrature.mat'
        assert os.path.exists(running_raw_file := os.path.join(spsig_file.removesuffix(spsig_file.split('/')[-1]
                                                              ),quadrature_file)), 'Quadrature file (running speed) should exist for each session'
        running : np.ndarray = running_wheel_to_speed(quadrature = SPSIG(running_raw_file).quad_data)
        if hasattr(SPSG, 'facemapTraces'):
            whisker : np.ndarray = SPSG.facemapTraces.motion
        else:
            whisker = None

    # Get arrays important for zeta test
    signal : np.ndarray = SPSG.sigCorrected.T # (neurons, all_times)
    event_times : np.ndarray = RES.info.StimTimes[:720] # (n_trials, )
    event_IDs_str : np.ndarray = np.array(RES.info.Stim.log.stim[:720]) # (n_trials, )
    event_IDs : np.ndarray = trials_apply_map(event_IDs_str, 
                                              {s:i for i, s in enumerate(np.unique(event_IDs_str))})
    assert event_IDs.shape == event_times.shape and all(isinstance(ID, np.int64) for ID in event_IDs
                                                        ), f'Mismatch in event times {event_times.shape} & IDs {event_IDs.shape} or event IDs not correctly mapped to ints {np.unique(event_IDs)}'

    # correct frame times for each ROI
    frame_times_1D :np.ndarray = RES.info.Frametimes # (all_times, )
    rois : DataFrame = DataFrame(RES.info.rois)
    ms_delays_1D : np.ndarray = np.array([*rois['msdelay']]) * 1e-3
    assert np.size(ms_delays_1D) == signal.shape[0], f'Mismatch between ROI info {ms_delays_1D.shape} and signal array {signal.shape}'
    frame_times_corrected_2D : np.ndarray = frame_times_1D[np.newaxis, :] + ms_delays_1D[:, np.newaxis] # using broadcasting, get corrected frame times for all neurons
    assert frame_times_corrected_2D.shape == signal.shape, f'Corrected frame times {frame_times_corrected_2D.shape} dont match signal array {signal.shape}'

    # Regress out z-whisker movement and z-running speed
    if regress_OUT:
        signal, discard_trials = regress_out_raw(signal=signal, running=running, whisker=whisker)
        frame_times_corrected_2D  = frame_times_corrected_2D[:discard_trials]
        # for purpose of zeta test, throw out those trials where we didn't regress out behavior
        keep_events = event_times < frame_times_corrected_2D[:, discard_trials].min()
        event_IDs = event_IDs[keep_events]
        event_times = event_times[keep_events]

    # print(spsig_file, 'zeta parameters prepared!')
    return (signal, frame_times_corrected_2D,
            event_IDs, event_times)

def responsive_zeta (RUN:bool = False, savedir:str = 'pydata', SPECIFIEDcond : str | None = None
                     ) -> dict[int:np.ndarray]:
    # runs zeta test and saves results in pickle file
    if RUN:
        if not os.path.exists(zeta_path := savedir):
            os.makedirs(zeta_path)
            print('Created directory for output:', zeta_path)

        # get files (without specifying root will popup asking for directory)
        g1spsig_files, g2spsig_files = group_condition_key(root = '/Volumes/my_SSD/NiNdata/data',
                                                           raw=True)
        
        g1pre, g1post = g1spsig_files['pre'], g1spsig_files['post']
        g2pre, g2post = g2spsig_files['pre'], g2spsig_files['post']
        sessions = [g1pre, g1post, g2pre, g2post]
        session_ranges = []
        indx = 0
        for session in sessions:
            session_ranges.append((indx, indx := indx + len(session)))
        all_sessions = g1pre + g1post + g2pre + g2post
        print('preparing parameters to run zeta test')
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            session_params = pool.map(prepare_zeta_params, all_sessions)
        print('\nRunning Zeta test on all sessions!\n')
        
        # takes really long time
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(run_ZETA, session_params) 
        print('Zeta tests finished!')
        
        # save    
        with open(os.path.join(zeta_path, 'zeta_results.pkl'), 'wb') as zetaRES:
            pickle.dump(results, zetaRES)
    

    # preprocess results of existing zeta test results file
    else:
        results = process_zeta_output(savedir, SPECIFIEDcond)
    
    return results
        

def process_zeta_output(savedir:str = 'pydata', SPECIFIEDcond : str | None = None):
    print('Fetching ZETA results:')
    conditions = ('g1pre', 'g1post', 'g2pre', 'g2post')
    # zeta res just contains sessions
    assert os.path.exists(zeta_file := os.path.join(savedir, 'zeta_results.pkl')), 'Need zeta results file to preprocess them'
    assert all(os.path.exists(os.path.join(savedir, f'{condition}_indexing.pkl')) 
                for condition in conditions), 'Need session indexing files for each condition to correctly split zeta results'

    # load files
    with open(zeta_file, 'rb') as zet:
        zeta_res = pickle.load(zet)
    cond_info_files = []
    for cond in conditions:
        with open(os.path.join(savedir, f'{cond}_indexing.pkl'), 'rb') as cond_info:
            COND_info = pickle.load(cond_info)
        cond_info_files.append(COND_info['session_index'])

    # dimensions per condition
    dims = {g_name:0 for g_name in conditions}
    for i in range(len(conditions)):
        condition_all_sessions_info = cond_info_files[i]
        for s in list(condition_all_sessions_info):
            dims[conditions[i]] += condition_all_sessions_info[s]['n_neurons']
    drr = 0
    # find out which neurons belong to which session
    dim_ranges = [(drr, drr := drr + c_range) for c_range in list(dims.values())]
    # prepare output dictionary
    # for each condition have dictionary with tt : (p-values and zeta scores) for all neurons in that condition
    preprocessed_zeta = {cond:{tt : [] for tt in range(len(zeta_res[0][0]))} 
                            for cond in conditions}
    if SPECIFIEDcond is not None: # return preprocessed zeta for CHOSEN condition
        assert SPECIFIEDcond in conditions, f'Your specified condition is NOT in {conditions}'
    
    # flatten zeta output
    flat_zeta = []
    for ss in range(len(zeta_res)) : flat_zeta = flat_zeta + [*zeta_res[ss]]
    iteration = 0 # for progress bar
    # loop over dimension ranges for each condition
    for icond,  (start, end) in enumerate(dim_ranges):
        cond_zeta = flat_zeta[start:end]            
        # loop over neurons and add their info to the TTL organized DF
        for ineuron, neuron in enumerate(cond_zeta):
            for tt in neuron:
                preprocessed_zeta[conditions[icond]][tt].append((neuron[tt]['zetaP'], 
                                                    neuron[tt]['ZETA']))
                progress_bar(current_iteration = (iteration := iteration + 1),
                                total_iterations = sum(dims.values()) * len(neuron))
    
    cond_message =  f' for {SPECIFIEDcond}' if SPECIFIEDcond is not None else ''
    print(f'\nDone preprocessing zeta{cond_message}!')
    
    return preprocessed_zeta[SPECIFIEDcond] if SPECIFIEDcond is not None else preprocessed_zeta



#---------------------------------- helper functions -------------------------------   
def trials_apply_map(trials_array:np.ndarray[str|int],
                    tmap:dict[str|int:int|str]) -> np.ndarray[int|str]:
        dtype = int if isinstance(list(tmap)[0], str) else str # depending on which map used want to map from str->int or from int->str
        
        if len(trials_array.shape) == 1:
            return np.fromiter(map(lambda s: tmap[s], trials_array), dtype = dtype)
        else:
            return np.array([np.fromiter(map(lambda s: tmap[s], trials_array[session, :]), dtype = dtype) 
                            for session in range(trials_array.shape[0])])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_zeta', type = str, default = 'no')
    parser.add_argument('-savedir', type = str, default = '/Volumes/my_SSD/NiNdata/zeta')
    return parser.parse_args()

def regress_out_raw(signal:np.ndarray,
                    running:np.ndarray | None = None,
                    whisker:np.ndarray | None = None) -> tuple[np.ndarray, int | None]:
    '''
    Returns: 
        1) regressed-out signal np.ndarray, 
        2) the trials to be discarded for the zeta calculation (frame index) | None if no trials discarded
    '''
    # make sure everything has correct shape
    assert signal.shape[0] < signal.shape[1], 'signal array must be (neurons, all_times) shaped'
    # regress out at once for all neurons z-scored running speed and whisker energy
    match running, whisker:
        case None, None:
            return signal, None
        case running, None:
            running_z = (running - np.nanmean(running)) / np.nanstd(running)
            X_design : np.ndarray = running.reshape(-1, 1)
            X_design = np.column_stack((np.ones(X_design.shape[0]), X_design)) # add intercept
            assert X_design.shape == (running_z.shape[0], 2), f'Wrong shape of design matrix {X_design.shape} should be {(running_z.shape[0], 2)}'
        case running, whisker:
            running_z = (running - np.nanmean(running)) / np.nanstd(running)
            whisker_z = (whisker - np.nanmean(whisker)) / np.nanstd(whisker)
            X_design = np.column_stack((np.ones(running_z.shape[0]), running_z, whisker_z)) # add intercept
            assert X_design.shape == (running_z.shape[0], 3), f'Wrong shape of design matrix {X_design.shape} should be {(running_z.shape[0], 3)}'

    # exclude timestamps with nan behavior    
    nan_behavior = np.any(np.isnan(X_design), axis = 1)
    X = X_design[~nan_behavior, :] 
    SIG = signal[:, ~nan_behavior]
    # X_design is (timepoints, predictors)
    t, p = X.shape
    # signal is (neurons, timepoints)
    n, ts = SIG.shape
    assert t == ts, 'Time dimensions of signal ({}) and design matrix ({}) do not match!'.format(ts, t)

    # regress out simultaneously for all neurons 
    piv = np.linalg.pinv(X)
    assert piv.shape == (p, t), 'Error with pseudo-inverse calculation, shape should be {}'.format((p,t))
    # beta coefficients for all neurons
    beta_neurons = np.einsum('pt, nt -> np', piv, SIG)
    assert beta_neurons.shape == (n, p), 'Error with beta coefficients calculation, shape should be {}'.format((n,p))
    # predicted signal for all neurons
    predicted_neurons = np.einsum('tp, np -> nt', X, beta_neurons)
    assert predicted_neurons.shape == (n, t), 'Error with predicted signal calculation, shape should be {}'.format((n,t))
    # residual
    residual_signal : np.ndarray = SIG - predicted_neurons

    return residual_signal, np.where(np.any(np.isnan(X_design), axis = 1))[0][0] # from which trial onwards nans

def running_wheel_to_speed(quadrature : np.ndarray) -> np.ndarray:
    # % Mouse speed
    #         % quad_data is delta angular value per second
    #         % full rotation is 1000 (for 360 degrees)
    HeadPosRadius = 8.5 # in cms
    Speed = 2* np.pi* HeadPosRadius * np.array(quadrature, dtype=float) / 1024
    return Speed


if __name__ == '__main__':
    args = parse_args()
    if args.run_zeta.lower() in ('y', 'yes', 'true', 't'):
        resp_indices = responsive_zeta(RUN = True)#, savedir=args.savedir)
        print('Zeta test finished, results saved!')
    else:
        neuron_significance_by_group_and_TT = responsive_zeta(SPECIFIEDcond='g2pre')
        # breakpoint()