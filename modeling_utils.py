from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
from AUDVIS import AUDVIS, Behavior, load_in_data
from VisualAreas import Areas
from GLM import clean_group_signal

import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from collections import defaultdict

#TODO: consider only including responsive neurons (know for sure those are neurons)
def session_loader(AV:AUDVIS, 
                   session_number:int|None = None,
                   region:Literal['V1', 'AM/PM', 'A/RL/AL', 'LM'] | None = None
                   ):
    '''
    Loads data for a session from provided AUDVIS object
    
    Later try with clean_group_signal(group_name=av.NAME)
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
    
    if session_number is not None:
        sn, en = AV.session_neurons[session_number]
        sig = AV.baseline_correct_signal(AV.zsig[:,:,sn:en], AV.TRIAL[0])
        beh:Behavior = AV.sessions[session_number]['behavior']

    print(f'Using session {session_number} from {sess_overview[session_number]}!')
    return sig, beh, session_number
    

# TODO: what is the kernel that's being convolved?
def get_spikes(S:np.ndarray)->np.ndarray:
    for i in range(S.shape[-1]):
        denoised, spikes, fb, params, lam = deconvolve(y = S[:,:,i].flatten())
        c, t = oasisAR1(S[:,:,i].flatten(), .95, s_min = .55)
        breakpoint()
        f, ax = plt.subplots()
        ax.plot(S[:,:,i].flatten())
        # ax.plot(denoised + fb)
        ax.plot(spikes)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    # exemplary sessions for G1pre: Epsilon (2), Eta(3), Zeta2(8)
    # exemplary sessions for G2pre: Dieciceis (0), Diez(2), Nueve(3)
    avs = load_in_data(pre_post='pre')
    # for av in avs:
    S, B, sess_i = session_loader(avs[1], region='V1')

    # OASIS fast spike deconvolution
    get_spikes(S)