# @matushalak
# For parallelized execution of Zeta test to determine responsive units
from SPSIG import SPSIG
from utils import group_condition_key, progress_bar
from matplotlib_venn import venn3
from collections import defaultdict
from pandas import DataFrame
from zetapy import zetatstest, zetatstest2 # test for neuronal responsiveness by Montijn et al.
import numpy as np
import multiprocessing as mp
import os
import pickle


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
            results[neuron][tt] = res # results for that trial type
            progress_bar(iteration, total_iterations)
            iteration += 1
    print(f'Zeta test for {signals.shape[0]} neurons and {len(list(ttwhere))} trial types.')
    return results
    
def prepare_zeta_params(spsig_file:str
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Prepares parameters for zeta test for all ROIs in a given session
    '''
    res_file = spsig_file[:-4] + '_Res.mat'

    SPSG = SPSIG(spsig_file)
    RES = SPSIG(res_file)

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

    print(spsig_file, 'zeta parameters prepared!')
    return (signal, frame_times_corrected_2D,
            event_IDs, event_times)

def responsive_zeta () -> dict[str:dict[int:np.ndarray]]:
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
    
    if not os.path.exists('pydata'):
        os.makedirs('pydata')

    # save    
    with open(os.path.join('pydata', 'zeta_results.pkl'), 'wb') as zetaRES:
        pickle.dump(results, zetaRES)

#---------------------------------- helper functions -------------------------------   
def trials_apply_map(trials_array:np.ndarray[str|int],
                    tmap:dict[str|int:int|str]) -> np.ndarray[int|str]:
        dtype = int if isinstance(list(tmap)[0], str) else str # depending on which map used want to map from str->int or from int->str
        
        if len(trials_array.shape) == 1:
            return np.fromiter(map(lambda s: tmap[s], trials_array), dtype = dtype)
        else:
            return np.array([np.fromiter(map(lambda s: tmap[s], trials_array[session, :]), dtype = dtype) 
                            for session in range(trials_array.shape[0])])


if __name__ == '__main__':
    resp_indices = responsive_zeta()