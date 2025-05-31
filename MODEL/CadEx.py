#@matushalak
import numpy as np
import os
import scipy.integrate as integr
import scipy.optimize as optim
import scipy as sp
import matplotlib.pyplot as plt
from joblib import delayed, Parallel, cpu_count

from tqdm import tqdm
from time import time

from src.AUDVIS import load_in_data, Behavior, AUDVIS
from src.Raw import rawsession_loader
import MODEL.adEx_utils as adEx_utils
import MODEL.adEx_models as adEx
import MODEL.modeling_utils as modutl
import MODEL.debug as dbg
from MODEL import PYDATA
import pickle

class CadEx:
    ''''
    Adaptive exponential integrate and fire model for
    in-vivo 2-photon calcium imaging data (CadEx)

    Input
    -----

    Explanation:
    """
    Clock-driven forward Euler integration for Calcium imaging AdEx neuron, 
    returns numpy arrays for outputs.

    !!!IMPORTANT NOTE!!!
    in principle this could be applied to a network, but currently the intent
    is to optimize parameters on each neuron separately. So we fit the entire
    model and optimize parameters to real spike train from neuron_i and then
    move onto neuron_i+1, and start from the beginning.

    equation:
    ## VOLTAGE EQUATION
    # classic adEx model without input current
    C * dV_i/dt = -gL (V_i - EL) + gL*∆T*exp((V_i-VT)/∆T) - w_i 
    # synapses, (those are the "input current") 
                 +g_s * ∑_j A_ji*(dF/F0 signal neuron j)
    # where
        - A_ji is a adjacency matrix with SIGNED synaptic weight 
        of synapse from neuron j to currently modelled neuron i
        - dF/F0 signal neuron j is continuously fluctuating calcium signal
        of neuron j. When j is active, dF/F0 increases. Because multiplied by
        synaptic weight, + synaptic weights will translate the signal directly,
        while negative synaptic weights will flip the signal.   
        NOTE: dF/F is Treated as 
        'effective current' units already pre-multipled by ~100. This multiplication
        should be optimized to results in A_ji that does not consistently 
        hit the bounds -10:+10, but rather stays around the middle
    
    ## ADAPTATION EQUATION
    tauw * dw_i/dt = a(V_i - EL) - w_i
    """
    '''
    def __init__(self,
                 SFmodel:float,
                 model_runtime_sec:float,
                 spikes:np.ndarray,
                 calcium:np.ndarray,
                 SFreal:float,
                 adjM:np.ndarray, 
                 session_name:str,
                 session_number:int | None = None,
                 frametimes:np.ndarray | None = None,
                 event_times:np.ndarray | None = None,
                 msdelays:np.ndarray | None = None,
                 fixed_params: dict[str:float] | None = None
                 ):
        # convert to milliseconds
        self.DTmodel = 1000 / SFmodel
        self.SFmodel, self.SFreal = SFmodel, SFreal
        self.Tmaxmodel = model_runtime_sec * 1000
        self.msdelays = msdelays
        # how many more samples we need based on how much higher sampling freq. is
        self.MODELsize = round(max(calcium.shape) * (self.SFmodel // self.SFreal))
        self.MODELts = np.linspace(0, self.Tmaxmodel, num = self.MODELsize, dtype=float)
        self.SIGsize = max(calcium.shape)
        self.N_neurons = msdelays.size
        self.ITERneuron = np.arange(self.N_neurons, dtype=int)

        # signals
        self.calcium = calcium

        _, _, self.model_params, _, _ = adEx_utils.define_experiment(
            dt = self.DTmodel, Tmax = self.Tmaxmodel)

    def upsample(NEURON):
        return dbg.upsample_memory_optimized(
                    sig=NEURON.calcium,
                    MODELts=NEURON.MODELts,
                    MODELsize=NEURON.MODELsize,
                    SIGsize=NEURON.SIGsize,
                    ITERneuron=NEURON.ITERneuron,
                    msdelays=NEURON.msdelays,
                    gaussSD=20,
                    additionalMSoffset=5)


    def run(self, INPUT_highHZ:np.ndarray,
            model:adEx = adEx.euler_nosynapse_cython,):
        start = time()
        adEx_utils.run_experiment(adExModel= model,
                                  Tmax=self.Tmaxmodel, dt = self.DTmodel,
                                  model_params=self.model_params,
                                  # *100 correction for dF/F amplitude relative to current
                                  Iapp=INPUT_highHZ * 100, 
                                  plot=True)
        print(f'{time()-start} s')

    def train_test(self, split:float
                   )->tuple[np.ndarray, np.ndarray]:
        pass

def upsample(sig:np.ndarray,
             MODELts:np.ndarray,
             MODELsize:int,
             SIGsize:int,
             ITERneuron:np.ndarray,
             msdelays:np.ndarray,
             gaussSD:float = 20,
             additionalMSoffset: float = 5,
            )->tuple[np.ndarray, np.ndarray]:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # collection array
    # Account for msdelays based on location within imaging frame
    bindelays = np.round(msdelays)

    SPACING = MODELsize // SIGsize
    batches = np.array_split(ITERneuron, 5)
    worker_args= [
        [sig[:,batch].copy(), MODELts.copy(), 
        batch.size, 
        SPACING, MODELsize,
        gaussSD, 
        bindelays[batch].copy(), additionalMSoffset
        ]
        for batch in batches]
    
    outList = Parallel(njobs = 5, verbose=14, backend='loky')(
        delayed(modutl.upsampling_worker)(*wa) for wa in worker_args
        )
    
    # not parallelized version (just as fast with 1 worker)
    # outList = [modutl.upsampling_worker(*wa) for wa in worker_args]
    
    return np.column_stack(outList)

def upsamp(SFmodel:float,
        model_runtime_sec:float,
        spikes:np.ndarray,
        calcium:np.ndarray,
        SFreal:float,
        adjM:np.ndarray, 
        session_name:str,
        session_number:int | None = None,
        frametimes:np.ndarray | None = None,
        event_times:np.ndarray | None = None,
        msdelays:np.ndarray | None = None,
        fixed_params: dict[str:float] | None = None):
    # convert to milliseconds
    DTmodel = 1000 / SFmodel
    SFmodel, SFreal = SFmodel, SFreal
    Tmaxmodel = model_runtime_sec * 1000
    msdelays = msdelays
    # how many more samples we need based on how much higher sampling freq. is
    MODELsize = round(max(calcium.shape) * (SFmodel // SFreal))
    MODELts = np.linspace(0, Tmaxmodel, num = MODELsize, dtype=float)
    SIGsize = max(calcium.shape)
    N_neurons = msdelays.size
    ITERneuron = np.arange(N_neurons, dtype=int)

    _, _, model_params, _, _ = adEx_utils.define_experiment(
        dt = DTmodel, Tmax = Tmaxmodel)
    return calcium, MODELts, MODELsize, SIGsize, ITERneuron, msdelays
          
def rundum():
    arr = list(range(20))
    start = time()
    results = Parallel(n_jobs=cpu_count(), verbose=10)(
        delayed(modutl.slow_square)(i) for i in arr
    )
    print("Results (slice):", results[:5], "… total time:", time() - start)

if __name__ == '__main__':
    # Dummy test (WORKS)
    # arr = list(range(20))
    # start = time()
    # results = Parallel(n_jobs=cpu_count(), verbose=10)(
    #     delayed(modutl.slow_square)(i) for i in arr
    # )
    # print("Results (slice):", results[:5], "… total time:", time() - start)
    
    # rundum()
    
    # exemplary sessions for G1pre: Epsilon (2), Eta(3), Zeta2(8)
    # exemplary sessions for G2pre: Dieciceis (0), Diez(2), Nueve(3)
    avs = load_in_data(pre_post='pre')
    # for av in avs:
    MP:dict = rawsession_loader(avs[1], region='V1')

    # model class
    NEURON = CadEx(SFmodel=1000, model_runtime_sec=2500, 
                   **MP)
    
    start = time()
    # sig, MODELts, MODELsize, SIGsize, ITERneuron, msdelays = upsamp(SFmodel=1000, model_runtime_sec=2500, 
    #                                                               **MP)
    # HIGHHz = upsample(sig=sig, 
    #                 MODELts=MODELts,
    #                 MODELsize=MODELsize,
    #                 SIGsize=SIGsize,
    #                 ITERneuron=ITERneuron,
    #                 msdelays=msdelays,
    #                 gaussSD=20,
    #                 additionalMSoffset=5
    # )
    # REAL DATA, DOESNT WORK!
    HIGHHz = NEURON.upsample()
    # HIGHHz = dbg.upsample_memory_optimized(sig=sig, 
    #                 MODELts=MODELts,
    #                 MODELsize=MODELsize,
    #                 SIGsize=SIGsize,
    #                 ITERneuron=ITERneuron,
    #                 msdelays=msdelays,
    #                 gaussSD=20,
    #                 additionalMSoffset=5)
    print(f'{time()-start} s')

    NEURON.run(INPUT_highHZ=HIGHHz[:,0])