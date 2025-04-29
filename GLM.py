import pandas as pd
import pyglmnet as glm
import numpy as np
import scipy as sp
import sklearn as skl

from AUDVIS import Behavior, AUDVIS, load_in_data
from typing import Literal


def inspect(pre_post: Literal['pre', 'post', 'both'] = 'pre'):
    AVs = load_in_data(pre_post=pre_post)
    
    for AV in AVs:
        breakpoint()


def stimulus_kernels(tbs:np.ndarray)->np.ndarray:
    '''
    Generates stimulus kernels for GLM (5 for bars), (5 for sounds)
        (for M in (bar, sound)
        M_present: (0,1)), 
        M_direction: rightward(+1, -1)leftward
        M_absolute_position: left(0,...,1)right, 
        onset_L_kernel,
        onset_R_kernel

    
    INPUT:
        tbs: trials by session
    '''
    # trials as rows
    tr = tbs.transpose(1,0) if tbs.shape[0] < tbs.shape[1] else tbs
    # diff features as columns of bigger matrix
    

if __name__ == '__main__':
    inspect()
