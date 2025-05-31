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
                 fixed_params: dict[str:float] = {'':None}
                 ):
        # convert to milliseconds
        self.DTmodel = 1000 / SFmodel # in milliseconds
        self.SFmodel, self.SFreal = SFmodel, SFreal
        self.Tmaxmodel = model_runtime_sec * 1000 # in milliseconds
        self.recordingFRAMES = max(calcium.shape)
        self.msdelays = msdelays # in milliseconds
        # only take model_runtime_sec slice of the data
        self.SIGsize = round(model_runtime_sec * self.SFreal) # frames from original signal
        # how many more samples we need based on how much higher sampling freq. is
        self.MODELsize = round(self.SIGsize * (self.SFmodel / self.SFreal))
        self.MODELts = np.linspace(0, self.Tmaxmodel, num = self.MODELsize, dtype=float)
        self.N_neurons = msdelays.size
        self.ITERneuron = np.arange(self.N_neurons, dtype=int)

        # signals
        self.calcium = calcium[:self.SIGsize, :]
        self.spikes = spikes[:self.SIGsize, :]
        self.frametimes = frametimes[:self.SIGsize]
        self.calciumMODEL = self.upsample()

        # adjacency matrix (neurons x neurons x 2) 
        # first layer = physical distances, second layer = dF/F correlation
        # use correlations as hot start for connectivity
        self.adjM = adjM

        _, _, self.model_params, _, _ = adEx_utils.define_experiment(
            dt = self.DTmodel, Tmax = self.Tmaxmodel, **fixed_params)

    def upsample(NEURON):
        return modutl.upsample_memory_optimized(
                    sig=NEURON.calcium,
                    MODELts=NEURON.MODELts,
                    MODELsize=NEURON.MODELsize,
                    SIGsize=NEURON.SIGsize,
                    ITERneuron=NEURON.ITERneuron,
                    msdelays=NEURON.msdelays,
                    gaussSD=20,
                    additionalMSoffset=5)


    def run(self, INPUT_highHZ:np.ndarray,
            ineuron:int = 0,
            model:adEx = adEx.nosynapse_euler_cython,
            # *100 correction for dF/F amplitude relative to current
            dFSCALER:float = 100,
            plot:bool = True):
        start = time()
        # For synaptic model include adjMatrix and evaluation array
        if model in (adEx.synapse_euler, adEx.synapse_euler_cython):
            # second depth slice is cross-correlations of signals
            adjROW = self.adjM[ineuron, :, 1]
            adjROW = np.ascontiguousarray(adjROW)
            modutl.__runWRAP__(ineuron = ineuron, 
                            model = model, 
                            INPUT_highHZ = INPUT_highHZ * dFSCALER,
                            adjROW = adjROW,
                            model_params = self.model_params,
                            MODELts = self.MODELts,
                            Tmaxmodel = self.Tmaxmodel, DTmodel = self.DTmodel, 
                            plot = plot)
            
        else:
            adEx_utils.run_experiment(adExModel= model,
                                    Tmax=self.Tmaxmodel, dt = self.DTmodel,
                                    model_params=self.model_params,
                                    Iapp=INPUT_highHZ * dFSCALER,
                                    ineuron=ineuron, 
                                    plot=plot,
                                    ts = self.MODELts)
        print(f'{time()-start} s', flush=True)

    def train_test(self, split:float
                   )->tuple[np.ndarray, np.ndarray]:
        pass


if __name__ == '__main__':
    # exemplary sessions for G1pre: Epsilon (2), Eta(3), Zeta2(8)
    # exemplary sessions for G2pre: Dieciceis (0), Diez(2), Nueve(3)
    avs = load_in_data(pre_post='pre')
    # for av in avs:
    MP:dict = rawsession_loader(avs[1], region='V1')

    # model class
    start = time()
    NEURON = CadEx(SFmodel=1000, 
                   # 3600 1h, 1800 30 min, 900 15 min, 600 10 min, 300 5 min,
                   model_runtime_sec=300, 
                   **MP)
    print(f'{time()-start} s')

    NEURON.run(INPUT_highHZ=NEURON.calciumMODEL,
               ineuron=0,
               model=adEx.synapse_euler_cython,
               dFSCALER=10,
               plot=False)