#@matushalak
import numpy as np
import scipy.integrate as integr
import scipy.optimize as optim
import scipy as sp

import timeit
from tqdm import tqdm

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
        pass
    

    def train_test(self, split:float
                   )->tuple[np.ndarray, np.ndarray]:
        pass


    def upsample(self
                 )->tuple[np.ndarray, np.ndarray]:
        pass


