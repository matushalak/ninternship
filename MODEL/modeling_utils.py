import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from collections import defaultdict

import timeit
from tqdm import tqdm

from MODEL import PYDATA
import os

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
def upsampling_worker(sigBATCH:np.ndarray, MODELts:np.ndarray,
                      batchSize:int,
                      SPACING:float, MODELsize:int,
                      gaussSD:float, 
                      bindelaysBATCH:np.ndarray, additionalMSoffset:float,
                      ):
    pid = os.getpid()
    print(f"[upsampling_worker STARTED, PID={pid}, batchSize={batchSize}, SPACING={SPACING}]", flush=True)
    BATCH = np.zeros((MODELsize, batchSize), dtype=float)
    for ni in range(batchSize):
        # Linear interpolation to get higher sampling rate
        out = np.interp(x=MODELts, xp = MODELts[::SPACING], fp = sigBATCH[:,ni])
        # Convolve with gaussian to smooth over linear interpolation artifacts
        symmgaussrange = np.linspace(-MODELsize//2, MODELsize//2, MODELsize)
        gaussian = np.exp(-(symmgaussrange/gaussSD)**2/2)
        gaussian = gaussian[gaussian > 0]
        out = np.convolve(out, gaussian, mode='same')
        # 0 pad from left to shift by few ms (first to correct for imaging offset)
        # + to allow for some "causality" in which signal came first
        lag = round(bindelaysBATCH[ni] + additionalMSoffset)
        out = np.roll(out, shift = lag)
        out[:lag] = 0
        BATCH[:,ni] = out
        print(lag, 'done', flush=True)
    return BATCH


def dummy_worker(sigBATCH, MODELts, batchSize, SPACING, MODELsize, gaussSD, bindelaysBATCH, additionalMSoffset):
    import os, time
    pid = os.getpid()
    print(f"[dummy_worker PID={pid} batchSize={batchSize}, SPACING={SPACING}]", flush=True)
    time.sleep(0.5)
    return np.zeros((MODELsize, batchSize), dtype=float)


def slow_square(x):
    pid = os.getpid()
    y = x
    for i in range(100000000):
        x = x**2
        x = y
    print(f"[Worker {pid}] squared {x}")
    return x * x