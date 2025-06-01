# Cython implementations
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport exp
import numpy as np
cimport numpy as np
from cython cimport Py_ssize_t

# Use typed memoryviews for fast buffer access
def forward_euler_cython(double[:] y0,
                         double dt,
                         tuple model_params,   # length 7: C, gL, EL, VT, DeltaT, tauw, a
                         tuple reset_params,   # length 3: Vpeak, Vreset, b
                         double[::1] currents        # precomputed Iapp
                         ):
    """
    Clock-driven forward Euler for AdEx, returns numpy arrays for outputs.
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

    # Unpack parameters into C locals
    cdef double C      = model_params[0]
    cdef double gL     = model_params[1]
    cdef double EL     = model_params[2]
    cdef double VT     = model_params[3]
    cdef double DeltaT = model_params[4]
    cdef double tauw   = model_params[5]
    cdef double a      = model_params[6]

    cdef double Vpeak  = reset_params[0]
    cdef double Vreset = reset_params[1]
    cdef double b      = reset_params[2]

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


# this is the synaptic model
def forward_euler_synapse_cython(double[:] y0,
                                double dt,
                                tuple model_params,   # length 8: C, gL, EL, VT, DeltaT, tauw, a, gs
                                tuple reset_params,   # length 3: Vpeak, Vreset, b
                                double[::1] adjM,           # adjacency matrix row for this neuron
                                double[:, ::1] currents     # shape (nsamples, nneurons)
                                ):
    """
    Clock-driven forward Euler for synaptic AdEx model.
    
    Parameters:
    -----------
    y0 : initial conditions [V0, w0]
    dt : time step
    model_params : (C, gL, EL, VT, DeltaT, tauw, a, gs)
    reset_params : (Vpeak, Vreset, b)
    adjM : adjacency matrix row for target neuron (synaptic weights)
    currents : calcium signals for all neurons (nsamples x nneurons)
    """
    # Initial conditions
    cdef double V0 = y0[0]
    cdef double w0 = y0[1]

    cdef Py_ssize_t N = currents.shape[0]  # number of time samples
    cdef Py_ssize_t M = currents.shape[1]  # number of neurons
    
    # Verify adjacency matrix size matches number of neurons
    if adjM.shape[0] != M:
        raise ValueError(f"adjM size {adjM.shape[0]} doesn't match number of neurons {M}")

    # Allocate output numpy arrays
    Vout_arr = np.empty(N, dtype=np.float64)
    wout_arr = np.empty(N, dtype=np.float64)
    spts_arr = np.zeros(N, dtype=np.int64)

    # Create memoryviews for C-level indexing
    cdef double[::1] Vout = Vout_arr
    cdef double[::1] wout = wout_arr
    cdef long[::1]   spts = spts_arr

    # Unpack model parameters into C locals
    cdef double C      = model_params[0]
    cdef double gL     = model_params[1]
    cdef double EL     = model_params[2]
    cdef double VT     = model_params[3]
    cdef double DeltaT = model_params[4]
    cdef double tauw   = model_params[5]
    cdef double a      = model_params[6]
    cdef double gs     = model_params[7]  # synaptic conductance

    # Unpack reset parameters
    cdef double Vpeak  = reset_params[0]
    cdef double Vreset = reset_params[1]
    cdef double b      = reset_params[2]

    # State variables
    cdef double V = V0
    cdef double w = w0
    cdef double dV, dw, V2, w2
    cdef double synaptic_input
    cdef double exp_arg, exp_term
    
    # Loop indices
    cdef Py_ssize_t i, j

    # Main integration loop
    for i in range(N):
        # Calculate synaptic input: gs * sum(adjM[j] * currents[i, j])
        synaptic_input = 0.0
        for j in range(M):
            synaptic_input += adjM[j] * currents[i, j]
        synaptic_input *= gs
        
        exp_arg = (V - VT) / DeltaT
        # Prevent exponential overflow
        #if exp_arg > 20.0:
        #    exp_term = exp(20.0)
        #else:
        exp_term = exp(exp_arg)
        
        # Compute derivatives
        dV = (-gL * (V - EL) 
              + gL * DeltaT * exp_term 
              - w 
              + synaptic_input) / C
        dw = (a * (V - EL) - w) / tauw

        # Forward Euler step
        V2 = V + dV * dt
        w2 = w + dw * dt

        # Spike detection and reset
        if V2 >= Vpeak:
            V2 = Vreset
            w2 = w + b
            spts[i] = 1
            # Set previous voltage to peak for visualization
            if i > 0:
                Vout[i-1] = Vpeak

        # Save state and advance
        V = V2
        w = w2
        Vout[i] = V
        wout[i] = w

    return Vout_arr, wout_arr, spts_arr
