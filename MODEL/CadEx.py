#@matushalak
import numpy as np
import scipy.integrate as integr
import scipy.optimize as optim
import scipy as sp
import matplotlib.pyplot as plt

import timeit
from tqdm import tqdm

from src.AUDVIS import load_in_data, Behavior, AUDVIS
from src.Raw import rawsession_loader
import MODEL.adEx_utils as adEx_utils
import MODEL.adEx_models as adEx
from MODEL import PYDATA

class CadEx:
    ''''
    Adaptive exponential integrate and fire model for
    in-vivo 2-photon calcium imaging data

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
        _, _, model_params, _, _ = adEx_utils.define_experiment(dt = self.DTmodel, 
                                                                Tmax = self.Tmaxmodel)
        adEx_utils.run_experiment(adExModel=adEx.euler_nosynapse_cython,
                                  Tmax=self.Tmaxmodel, dt = self.DTmodel,
                                  model_params=model_params,
                                  # correction for dF/F amplitude relative to current
                                  Iapp=self.upsample(calcium[:,0]) * 125, 
                                  plot=True)
    

    def train_test(self, split:float
                   )->tuple[np.ndarray, np.ndarray]:
        pass


    def upsample(self, sig
                 )->tuple[np.ndarray, np.ndarray]:
        FULLsize = round(sig.size * (self.SFmodel // self.SFreal))
        FULLts = np.arange(FULLsize)
        DIM = np.zeros(round(sig.size * (self.SFmodel // self.SFreal)), dtype=float)
        # technically should use frametimes

        sigdim = sig.size
        SPACING = DIM.size // sigdim
        DIM[::SPACING] = sig

        out = np.interp(x=FULLts, xp = FULLts[::SPACING], fp = sig)
        symmgaussrange = np.linspace(-FULLsize//2, FULLsize//2, FULLts.size)
        gaussSD = 20
        gaussian = np.exp(-(symmgaussrange/gaussSD)**2/2)
        gaussian = gaussian[gaussian > 0]

        return np.convolve(out, gaussian, mode='same')


if __name__ == '__main__':
    # exemplary sessions for G1pre: Epsilon (2), Eta(3), Zeta2(8)
    # exemplary sessions for G2pre: Dieciceis (0), Diez(2), Nueve(3)
    avs = load_in_data(pre_post='pre')
    # for av in avs:
    MP:dict = rawsession_loader(avs[1], region='V1')

    NEURON = CadEx(SFmodel=1000, model_runtime_sec=60, 
                   **MP)