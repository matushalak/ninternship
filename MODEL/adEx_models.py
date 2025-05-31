# @matushalak
import numpy as np
import numba as nb
from MODEL.adex_euler import forward_euler_cython, forward_euler_synapse_cython

#%%---BASIC Adaptive exponential and fire model definitions---
def nosynapse(t:float, y:tuple[float, float], 
            I:float, params:dict[str:float]):
    '''
    Brette & Gerstner 2005 adEx neuron WITHOUT synapses
    responding to input current
    '''
    V, w = y
    C = params['C']     # pF
    gL = params['gL']     # nS
    EL = params['EL']    # mV
    VT = params['VT']    # mV
    DeltaT = params['DeltaT']  # mV SLOPE
    tauw = params['tauw']  # ms
    a = params['a']       # nS

    dVdt = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + I) / C
    dwdt = (a * (V - EL) - w) / tauw
    return dVdt, dwdt


#%%--- CYTHON fast implementations ----
def nosynapse_euler_cython(*args, **kwargs):
    '''
    ## CYTHON IMPLEMENTATION
    Wrapper for Cython code running the whole forward euler method
    + the adEx model WITHOUT synapses
    '''
    return forward_euler_cython(*args, **kwargs)

def synapse_euler_cython(*args, **kwargs):
    '''
    ## CYTHON IMPLEMENTATION
    Wrapper for Cython code running the whole forward euler method
    + the adEx model WITH synapses

    #### 3x faster than the numba synapse_euler implementation
    '''
    return forward_euler_synapse_cython(*args, **kwargs)


#%%------- NUMBA compiled files -------
# With extremely large arrays O(10^6) elements, as fast as cython version
# With smaller arrays, (for us we use O(10^3) cython_euler is MUCH faster
@nb.njit(fastmath = True)
def nosynapse_euler(y0:tuple[float, float], dt:float, 
                    model_params:tuple[float,...], 
                    reset_params:tuple[float,...],
                    currents:np.ndarray):
    '''
    # NUMBA IMPLEMENTATION
    Runs the entire forward euler integration method on calssical adEX model (Brette & Gerstner 2005)
    '''
    V, w = y0
    nsamples = currents.size
    Vout = np.zeros(nsamples, dtype=np.float32)
    wout = np.zeros(nsamples, dtype=np.float32)
    spts = np.zeros(nsamples, dtype=np.int8)

    # unpack reset parameters
    Vpeak, Vreset, b = reset_params
    C, gL, EL, VT, DeltaT, tauw, a = model_params
    for i in range(nsamples):
        '''
        Brette & Gerstner 2005 adEx neuron WITHOUT synapses
        responding to input current
        '''
        dV = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + currents[i]) / C
        dw = (a * (V - EL) - w) / tauw

        # need to get change PER dt
        V2, w2 = V + dV*dt, w+dw*dt
        # spike
        if V2 >= Vpeak:
            V2 = Vreset
            w2 = w + b
            # Count spike occured at this time
            spts[i] = 1
            Vout[i-1] = Vpeak
        
        # save results
        V, w = V2, w2
        Vout[i] = V
        wout[i] = w
    
    return Vout, wout, spts


@nb.njit(fastmath = True)
def synapse_euler(y0:tuple[float, float], dt:float, 
                  model_params:tuple[float,...], 
                  reset_params:tuple[float,...],
                  adjM:np.ndarray,
                  currents:np.ndarray):
    '''
    # NUMBA IMPLEMENTATION
    Runs the entire forward euler integration method on synaptic CadEx model:
    ------------------------
    ### VOLTAGE EQUATION
    C * dV_i/dt = -gL (V_i - EL) + gL*∆T*exp((V_i-VT)/∆T) - w_i 
    #### synapses, (those are the "input current") 
                 +g_s * ∑_j A_ji*(dF/F0 signal neuron j)
    #### where
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
    
    ### ADAPTATION EQUATION
    tauw * dw_i/dt = a(V_i - EL) - w_i

    INPUT:
    adjM : np.ndarray shape (nneurons); weight of neuron to itself is strictly set to 0 
        (not autoregressive model)
    currents : np.ndarray shape (nts, nneurons)
    '''
    V, w = y0
    nsamples = currents.shape[0]
    Vout = np.zeros(nsamples, dtype=np.float64)
    wout = np.zeros(nsamples, dtype=np.float64)
    spts = np.zeros(nsamples, dtype=np.bool)

    # unpack reset parameters
    Vpeak, Vreset, b = reset_params
    C, gL, EL, VT, DeltaT, tauw, a, gs = model_params
    for i in range(nsamples):
        allinputs = np.ascontiguousarray(currents[i,:])
        dV = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + 
              # synapses overall conductance gs * (dot product between "synaptic" dF/F and signed synaptic weighs)
              (gs * np.dot(allinputs, adjM))) / C
        dw = (a * (V - EL) - w) / tauw

        # need to get change PER dt
        V2, w2 = V + dV*dt, w+dw*dt
        # spike
        if V2 >= Vpeak:
            V2 = Vreset
            w2 = w + b
            # Count spike occured at this time
            spts[i] = 1
            Vout[i-1] = Vpeak
        
        # save results
        V, w = V2, w2
        Vout[i] = V
        wout[i] = w
    
    return Vout, wout, spts
