#@matushalak
import numpy as np
import scipy.integrate as integr
import scipy.optimize as optim
import scipy as sp

# NOTE: first without synapses
# later add synaptic activity
# Excitatory Ee, AMPA
# Ee = 0

# Inhibitory Ei, GABA_A
# Ei = -75

# NOTE: need to interrupt integration
# NOTE: scale is incredibly important!
def adEX(t:float, vars:list[float, float],
         I:callable
        # params:dict
        ):
    V, w = vars
    
    C = 281
    gl = 30
    El = -70
    Vt = -50.4
    slope = 2
    tauw = 144
    a = 4
    b = 80.5
    Vpeak = -20

    dvdt = (-gl * (V-El) + gl*slope*np.exp((V-Vt)/slope) - w + I(t)) / C
    dwdt = (a*(V-El) - w) / tauw

    return np.array([dvdt, dwdt])


vstart, wstart = -70, 0
ini = [vstart, wstart]
t_interval = [0,1000]
