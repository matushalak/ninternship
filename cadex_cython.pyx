# adex_euler.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport exp
import numpy as np
cimport numpy as np
from cython cimport Py_ssize_t

# Use typed memoryviews for fast buffer access
def CadEXneuron(double[:] y0,
                double dt,
                tuple model_params,   # length 9: C, gL, EL, VT, DeltaT, tauw, a, Vreset, b
                double[::1] currents  # precomputed Iapp
                ):
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
        while negative synaptic weights will flip the signal
    
    ## ADAPTATION EQUATION
    tauw * dw_i/dt = a(V_i - EL) - w_i
    """
    # initial conditions
    cdef double V0 = y0[0]
    cdef double w0 = y0[1]

    cdef Py_ssize_t N = currents.shape[0]
    # Allocate output numpy arrays
    Vout_arr = np.empty(N, dtype=np.float64)
    wout_arr = np.empty(N, dtype=np.float64)
    spts_arr = np.zeros(N, dtype=np.int64)

    # Create memoryviews for C-level indexing
    cdef double[::1] Vout = Vout_arr
    cdef double[::1] wout = wout_arr
    cdef long[::1]   spts = spts_arr

    # Fixed parameters
    cdef double Vpeak  = 0
    # only the C/gL ratio matters, so we can fix gL
    # cdef double gL     = 30
    # shifting EL and Vt by same amount = shift in voltage scale (don't care about voltage)
    # cdef double EL     = -70.6
    
    # Unpack 9 optimized model parameters into C locals
    cdef double C      = model_params[0]
    cdef double gL     = model_params[1]
    cdef double EL     = model_params[2]
    cdef double VT     = model_params[3]
    cdef double DeltaT = model_params[4]
    cdef double tauw   = model_params[5]
    cdef double a      = model_params[6]
    cdef double Vreset = model_params[7]
    cdef double b      = model_params[8]

    # Unpack optimized synaptic weights (vector)

    cdef double V = V0
    cdef double w = w0
    cdef double dV, dw, V2, w2
    cdef Py_ssize_t i

    for i in range(N):
        # compute derivatives
        dV = (-gL * (V - EL)
              + gL * DeltaT * exp((V - VT)/DeltaT)
              - w + currents[i]) / C
        dw = (a * (V - EL) - w) / tauw

        # forward Euler step
        V2 = V + dV * dt
        w2 = w + dw * dt

        # spike reset
        if V2 >= Vpeak:
            V2 = Vreset
            w2 = w + b
            # Count spike occured at this time
            spts[i] = 1
            Vout[i-1] = Vpeak

        # save and advance
        V = V2
        w = w2
        Vout[i] = V
        wout[i] = w

    return Vout_arr, wout_arr, spts_arr
