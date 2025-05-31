import numpy as np
import matplotlib.pyplot as plt
from joblib import cpu_count, delayed, Parallel
import MODEL.adEx_models as adEx
from MODEL.adEx_utils import run_experiment
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


def __runWRAP__(ineuron:int, 
                model:adEx, 
                INPUT_highHZ:np.ndarray,
                adjROW:np.ndarray,
                model_params:dict,
                MODELts:np.ndarray,
                Tmaxmodel:float, DTmodel:float, 
                plot:bool):
    adjROW[ineuron] = 0
    eval_score = run_experiment(adExModel= model,
                                Tmax=Tmaxmodel, dt = DTmodel,
                                model_params=model_params,
                                Iapp=INPUT_highHZ, 
                                adjM=adjROW,
                                ineuron=ineuron,
                                plot=plot,
                                ts = MODELts)

### --- Other model utils ---
def upsampling_worker(sigBATCH:np.ndarray, MODELts:np.ndarray,
                      batchSize:int,
                      SIGsize:float, MODELsize:int,
                      gaussSD:float, 
                      bindelaysBATCH:np.ndarray, additionalMSoffset:float,
                      ibatch:int,
                      ):
    pid = os.getpid()
    print(f"[upsampling_worker STARTED, PID={pid}, batchSize={batchSize}]", flush=True)
    BATCH = np.zeros((MODELsize, batchSize), dtype=float, order = 'C')
    MODELdownsampledTS = np.linspace(MODELts[0], MODELts[-1], SIGsize)
    for ni in tqdm(range(batchSize), position=ibatch):
        # Linear interpolation to get higher sampling rate
        out = np.interp(x=MODELts, xp = MODELdownsampledTS, fp = sigBATCH[:,ni])
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
        # print('Neuron w lag: ',lag, 'done', flush=True)
    return BATCH

def upsample_memory_optimized(sig: np.ndarray,
                             MODELts: np.ndarray,
                             MODELsize: int,
                             SIGsize: int,
                             ITERneuron: np.ndarray,
                             msdelays: np.ndarray,
                             gaussSD: float = 20,
                             additionalMSoffset: float = 5,
                             ) -> tuple[np.ndarray, np.ndarray]:
    
    # Force single-threaded NumPy to avoid conflicts
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    bindelays = np.round(msdelays)
    
    # Use fewer workers for memory-intensive tasks
    n_workers = min(cpu_count() // 2, 8)
    batches = np.array_split(ITERneuron, n_workers)
    
    print(f"Memory-optimized version using {n_workers} workers")
    
    worker_args = [
        [sig[:, batch].copy(),  # Make explicit copies to avoid sharing issues
         MODELts.copy(),
         batch.size,
         SIGsize, MODELsize,
         gaussSD,
         bindelays[batch].copy(), additionalMSoffset,
         ibatch
        ]
        for ibatch, batch in enumerate(batches)
    ]
    
    # Try different multiprocessing backends
    for backend in ['loky', 'threading', 'multiprocessing']:
        try:
            print(f"Trying backend: {backend}")
            outList = Parallel(n_jobs=n_workers, verbose=10, backend=backend)(
                delayed(upsampling_worker)(*wa) for wa in worker_args
            )
            print(f"Success with {backend} backend!")
            break
        except Exception as e:
            print(f"Failed with {backend}: {e}")
            continue
    else:
        print("All backends failed, falling back to sequential processing")
        # single core also good for debugging
        outList = [upsampling_worker(*wa) for wa in worker_args]
    
    return np.column_stack(outList)