from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.VisualAreas import Areas
import src.utils as utl

import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from collections import defaultdict

from src import PYDATA, PLOTSDIR

def rawsession_loader(AV:AUDVIS, session_number:int|None = None,
                      region:Literal['V1', 'AM/PM', 'A/RL/AL', 'LM'] | None = None
                      )->dict:
    '''
    Loads data for a session. Decides based on (trial-locked) AUDVIS object, but actually
    returns RAW dF/F data, as well as, RAW MLspike data with deconvolved spikes
    
    Returns everything together with other important information in a dictionary that can
    be used to instatiate the Model
    '''
    AR = Areas(AV, get_indices=True)
    sess_overview:dict[int:str] = dict()
    region_overview:dict[str:list[int]] = defaultdict(list)
    # annotate sessions with appropriate brain regions
    for isess, (sessneurstart, sessneurstop) in enumerate(AV.session_neurons):
        regions_in_session = AR.dfROI.iloc[sessneurstart : sessneurstop
                                           ]['Region'].value_counts()
        percentage_regions_in_session = regions_in_session / regions_in_session.sum()
        if percentage_regions_in_session.max() > 0.9:
            sess_region = percentage_regions_in_session.idxmax()
        else:
            sess_region = '_'.join(percentage_regions_in_session.index.tolist())
        
        sess_overview[isess] = sess_region
        region_overview[sess_region].append(isess)
    
    # session overview
    print(f'{AV.NAME} sessions overview:', sess_overview)

    if region is not None and session_number is None:
        # return first session with that value
        session_number = region_overview[region][0] # return first session for that region
    
    # by now session number is established
    firstneur, lastneur = AV.session_neurons[session_number]

    g = f'{AV.NAME[:2]}'
    animal, date, _ = AV.sessions[session_number]['session'].split('_')
    nneur = AV.sessions[session_number]['n_neurons']
    chosen_sess =  f'{g}_{'_'.join([animal, date])}_{nneur}'
    RAW_SESS_OVERVIEW = utl.get_sessions_overview()
    CHOSEN_RAW_FILE = RAW_SESS_OVERVIEW[g][chosen_sess]

    print(f'Using raw file for session {session_number} from {sess_overview[session_number]}! ({CHOSEN_RAW_FILE})')
    
    spikes:np.ndarray = utl.get_SPSIGvars(spsgpath=CHOSEN_RAW_FILE, vars=['deconCorrected'], npsave=True)
    calcium:np.ndarray = utl.get_SPSIGvars(spsgpath=CHOSEN_RAW_FILE, vars=['sigCorrected'], npsave=True)

    # Model input
    MODEL_PARAMS = {
        'spikes' : spikes,
        'calcium' : calcium,
        'SFreal' : AV.SF,
        'adjM' : AR.adjacencyMATRIX(indices=np.arange(firstneur, lastneur),
                                    signal=calcium),
        'session_name': chosen_sess,
        'session_number' : session_number, 
        # If at some point want to do rasterplot and PSTH afterwards, 
        # need to write trial-locking msDelays in Cython using:
        'frametimes' : AV.sessions[session_number]['frame_times'],
        'event_times' : AV.sessions[session_number]['event_times'],
        'msdelays' : np.array(AV.rois.msdelay.iloc[firstneur:lastneur].to_numpy(), dtype=float)
    }
    return MODEL_PARAMS
    

if __name__ == '__main__':
    # exemplary sessions for G1pre: Epsilon (2), Eta(3), Zeta2(8)
    # exemplary sessions for G2pre: Dieciceis (0), Diez(2), Nueve(3)
    avs = load_in_data(pre_post='pre')
    # for av in avs:
    MP:dict = rawsession_loader(avs[1], region='V1')
