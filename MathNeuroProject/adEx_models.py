# @matushalak
import numpy as np
import numba as nb
from adex_euler import forward_euler_cython

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
def euler_nosynapse_cython(*args, **kwargs):
    '''
    Wrapper for Cython code running the whole forward euler method
    + the adEx model WITHOUT synapses
    '''
    return forward_euler_cython(*args, **kwargs)


#%%------- NUMBA compiled files -------
@nb.njit
def nosynapse_fast(t:float, y:tuple[float, float], I:float, 
                   params:tuple[float,...]):
    '''
    Brette & Gerstner 2005 adEx neuron WITHOUT synapses
    responding to input current
    '''
    V, w = y
    C, gL, EL, VT, DeltaT, tauw, a = params
    dVdt = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + I) / C
    dwdt = (a * (V - EL) - w) / tauw
    return dVdt, dwdt


# With extremely large arrays O(10^6) elements, as fast as cython version
# With smaller arrays, (for us we use O(10^3) cython_euler is MUCH faster
@nb.njit(fastmath = True)
def nosynapse_euler(y0:tuple[float, float], dt:float, 
                    model_params:tuple[float,...], 
                    reset_params:tuple[float,...],
                    currents:np.ndarray):
    '''
    Runs the entire forward euler integration method
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
                  currents:np.ndarray):
    '''
    Runs the entire forward euler integration method
    '''
    V, w = y0
    nsamples = currents.size
    Vout = np.zeros(nsamples, dtype=np.float64)
    wout = np.zeros(nsamples, dtype=np.float64)
    spts = np.zeros(nsamples, dtype=np.bool)

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
            spts[i] = True
            Vout[i-1] = Vpeak
        
        # save results
        V, w = V2, w2
        Vout[i] = V
        wout[i] = w
    
    return Vout, wout, spts
