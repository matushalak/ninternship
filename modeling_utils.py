from AUDVIS import AUDVIS, Behavior, load_in_data
from VisualAreas import Areas

import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from collections import defaultdict

import timeit
from tqdm import tqdm

### --- Wrappers for Cython models ---


### --- Optimizer utils ---
def loss(ypred:np.ndarray, y:np.ndarray
         )->float:
    '''
    Coincidence factor, used by Clopath, Jolivet, Gerstener et al.
    '''
    pass


def parameter_ranges():
    ''''
    9 intrinsic model parameters
    ----------------------------
    C      = (30, 300)
    gL     = (1.5, 15)
    EL     = (-70, -59)
    VT     = (-60, -42)
    DeltaT = (0.6, 6)
    tauw   = (16, 720)
    a      = (-12, 80)
    Vreset = (-80, -40) 
    b      = (0, 400)
    g_s    = (0, 1) # overall synaptic conductance
    
    N synaptic weights
    ----------------------------
    A      = synaptic weight (-10, 10)
        - 10 inhibitory, +10 excitatory
    
    '''

    # Initial population: 
    # adapting, bursting, fast, slow tonic spiking
    # a) Tonic spiking
    tonic_params = (200,10,)

### --- Other model utils ---