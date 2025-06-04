import numpy as np
import matplotlib.pyplot as plt
from time import time
import MODEL.adEx_utils as adEx_utils
import MODEL.adEx_models as adEx
import MODEL.phase_plane as pp
'''
Default parameters Brette & Gerstner (2005)
5 SCALING parameters (Naud et al., 2008)
'C' : 281.0,     # pF
'gL' : 30.0,     # nS
'EL' : -70.6,    # mV
'VT' : -50.4,    # mV
'DeltaT' : 2.0,   # mV
'Vpeak' : 0, # mV - can't be more than 5 due to numerical overflow in scipy

4 bifurcation parameters ()
'Vreset' : -70.6, # mV
    low (typically reset to ~EL and model has to climb back up to spiking region)
    higher values induce bursting (if reset is > VT burst during input pulse)
Time constant of adaptation
    'tauw' : 144.0,  # ms
    How fast the is the evolution of the adaptation variable (w)
    Lower values make it faster, higher values make it slower
    at tauw = 1, there is no separation of timescales anymore
Subthreshold adaptation (regardless of spikes, purely to voltage changes)
    default 'a' : 4.0,       # nS 
    the lower this is, the more the adaptation variable (w) just decays to 0 after a spike
    the higher this is, the more the adaptation variable (w) responds to the input signal regardless of spikes as a "leak" current
Spike-triggered adaptation
    default 'b' : 80.5    # pA !!!
    how much adaptation variable (w) if offset after a spike
    higher values mean voltage must overcome that many more mV before next spike can be triggered
'''
### Basic model without synapses experiments
start = time()
# Custom input current
np.random.seed(1)
# Choose model
# Fastest
MODEL = adEx.nosynapse_euler_cython

# Choose plotting
PLOT = True

# Experiments
# Noisy pulsed
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(#EL = -70, #tauw = 50, a= 6, Vreset = -47,
#                                                                 # dt = 1,
#                                                                 # 40: 5 trials per stimulus, 80: 10 trials per stimulus, ... 720: 90 trials per stimulus
#                                                                 # Tmax = 40 * 47 * (1000/15.45), # ms
#                                                                 custom = lambda t: np.random.normal(loc = 800, scale = 500) 
#                                                                 if t % 47 < 15 and t > 15 else np.random.normal(loc = 200, scale = 500))
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# # Default experiment - pulsed current (not in original paper)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = 1) # returns the same as adEx.utils.default_experiment()
# adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)

# Figure 1C - Voltage response to small and large current
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Vpeak = -20, 
                                                                figure = 'small_large')
PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
                          plot=PLOT)
pp.AdExPP(*PP)

# # Figure 2C - Bursting Voltage response to small and large current when setting Vreset = -47
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Vreset = -47, 
#                                                                 figure = 'small_large')
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# Figure 2D - Postinhibitory Rebound Voltage response to hyperpolarization when setting EL = -60, a = 80, tauw = 720
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(EL = -60, Vreset = -60, a = 80, tauw = 720,
#                                                                 figure = 'hyperpol')
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# print(time()-start)


# # Bifurcation parameter experiments
# #### Vreset
# # Low Vr = default pulsed (typically reset to ~EL and model has to climb back up to spiking region)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)
# # High Vr (if reset is > VT burst during input pulse)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 Vreset = -45) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# #### tauw
# # # Low tauw (fast adaptation variable)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 tauw = 10) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# # High tauw (slow adaptation variable)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 tauw = 720) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)


# #### a
# # Low a (spiking decreases w)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 a = -80) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# # High a (spiking increases w)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 a = 80) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)


# #### b
# # Low b - small / no penalty after spike (can spike more)
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 b = 2) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)

# # High b - big penalty after spike 
# Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(dt = .1,
#                                                                 b = 800) # returns the same as adEx.utils.default_experiment()
# PP = adEx_utils.run_experiment(adExModel=MODEL, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp,
#                           plot=PLOT)
# pp.AdExPP(*PP)