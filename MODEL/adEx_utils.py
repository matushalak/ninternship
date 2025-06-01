#@matushalak
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Literal, Callable
import seaborn as sns
import MODEL.adEx_models as adEx

# Default Experimental parameters from original definition of adEx model!
# Parameters (from Brette & Gerstner 2005, Table 1, "regular spiking")
def default_experiment()->tuple[int, float, dict, dict, Callable]:
    Tmax = 1000
    dt = 0.1
    
    model_params = {
        'C' : 281.0,     # pF
        'gL' : 30.0,     # nS
        'EL' : -70.6,    # mV
        'VT' : -50.4,    # mV
        'DeltaT' : 2.0,   # mV
        'Vpeak' : 0.0, # mV - can't be more than 5 due to numerical overflow in scipy
        'tauw' : 144.0,  # ms
        'a' : 4.0,       # nS
        'gs' : .99, # overall synaptic conductance
        'Vreset' : -70.6, # mV
        'b' : 80.5,    # pA !!!!
    }
    experiment_settings = {
        'figure' : None,
        'custom' : None,
        'Istart' : 0, 'Istop' : Tmax, 
        'Iperiod' : 100, 'Iduration' : 60, 
        'Iamplitude' : 1000
        }
    
    # default periodically pulsed current
    current:Callable = lambda t: Iapp(t, **experiment_settings)
    
    return Tmax, dt, model_params, experiment_settings, current


def define_experiment(**kwargs)->tuple[int, float, dict, dict, Callable]:
    '''
    Only need to define deviations from default adEx values taken from
    Brette & Gerstner 2005, (Table 1, "regular spiking")
    '''
    Tmax, dt, model_params, experiment_settings, _ = default_experiment()
    # Update default model with user specifications
    for arg, argval in kwargs.items():
        # Default when no fixed args NEURON model definition
        if arg == '':
            continue
        if arg == 'Tmax':
            Tmax = argval
            # to maintain default intended behavior throughout recorded session
            if 'Istop' not in kwargs:
                experiment_settings['Istop'] = Tmax
        elif arg == 'dt':
            dt = argval
        elif arg in model_params:
            model_params[arg] = argval
        elif arg in experiment_settings:
            experiment_settings[arg] = argval
        else:
            raise KeyError(f'''{arg} is not a valid parameter name, see valid parameter names: 
                           Tmax, dt, 
                           model_params:{list(model_params)} 
                           experiment_settings:{list(experiment_settings)}''')
    
    # user-defined input current
    current:Callable = lambda t: Iapp(t, **experiment_settings)
    return Tmax, dt, model_params, experiment_settings, current


def Iapp(t:float, 
         figure:Literal['small_large', 'hyperpol'] | None,
         custom:Callable, 
         Istart:int, Istop:int, Iperiod:int, Iduration:int, 
         Iamplitude:float):
    '''
    Figure argument is used to reproduce figures from Brette & Gerstner (2005)
    '''
    # custom function supplied
    if custom is not None:
        return custom(t)
    if figure is None:
        # specified by parameters
        return Iamplitude if t >= Istart and t <= Istop and (t % Iperiod) < Iduration else 0
    elif figure ==  'small_large':
        # Fig 2C / 3C - small and large pulse with Vreset = EL OR Vreset = Vt + 3
        return 500 if t < 200 else(0 if t < 500 else 800)
    elif figure ==  'hyperpol':
        # Fig 3D - rebound spike
        return -800 if 10 < t < 410 else 0
    

def bin_spikes(spts_ms:np.ndarray, nbins:int, Tmax_ms:float,
               maxcount:int)->np.ndarray:
    """
    Bin high-res spiketimes into designated number of bins
    """
    bin_edges = np.linspace(0,Tmax_ms, nbins+1)
    spike_counts, _ = np.histogram(spts_ms, bins = bin_edges)
    return np.clip(spike_counts, 0,maxcount)


def run_experiment(adExModel:adEx, Tmax:float, dt:float, model_params:dict, Iapp:Callable|np.ndarray, 
                   ineuron:int = 0,
                   ts:np.ndarray | None = None, 
                   adjM:np.ndarray | None = None,
                   evaluate:bool = False,
                   eval_array:np.ndarray | None = None,
                   plot:bool = False)-> np.ndarray | None:
    '''
    Runs entire experiment for adEx model of choice with model and experimental parameters of choice
    '''        
    y0 = np.array([model_params['EL'], 0.0], dtype=float)
    if ts is None:
        ts = np.arange(0, Tmax, dt, dtype=float)

    # Clock-based simulator (constant step sizes), simple forward euler method
    reset_tuple = tuple([model_params[par] for par in ['Vpeak', 'Vreset', 'b']])
    
    # Run spiking model
    if adjM is None:
        model_tuple = tuple([model_params[par] for par in ['C', 'gL', 'EL', 'VT', 'DeltaT', 'tauw', 'a']])
        # without synapses / 1 synapse
        if isinstance(Iapp, np.ndarray):
            assert max(Iapp.shape) == ts.size, f'Iapp shape {Iapp.shape} must match ts shape {ts.shape} on 1 dimension!'
            # slicing breaks contiguos arrays!!! important for C / Cython that array is C-contigous (by row)
            all_curr = Iapp[:, ineuron].copy()
        else:
            all_curr = np.array([Iapp(t) for t in ts], dtype=float)

        V_all, w_all, spikes = adExModel(y0 = y0,dt=dt,
                                        model_params=model_tuple, 
                                        reset_params=reset_tuple,
                                        currents=all_curr)
    else:
        model_tuple = tuple([model_params[par] for par in ['C', 'gL', 'EL', 'VT', 'DeltaT', 'tauw', 'a', 'gs']])
        # WITH (many) synapses, weights in adjM
        assert isinstance(Iapp, np.ndarray), f'Synaptic model only works with arrays of inputs, not {type(Iapp)}'
        assert max(Iapp.shape) == ts.size, f'Iapp shape {Iapp.shape} must match ts shape {ts.shape} on 1 dimension!'
        # model must be synaptic model, won't work with the other ones
        assert adExModel in (adEx.synapse_euler, adEx.synapse_euler_cython), 'When providing adjacency matrix, must choose synaptic model'
        assert adjM.size == Iapp.shape[1], f'The number of entries in the adjacency matrix row {adjM.size} dont match the number of neurons {Iapp.shape[1]}'
        adjM[ineuron] = 0.0
        assert adjM[ineuron] == 0, 'Synaptic weight for autosynapse must be 0'
        assert adjM.flags['C_CONTIGUOUS'], 'Synaptic weights must be C-contiguous array'
        V_all, w_all, spikes = adExModel(y0 = y0,dt=dt,
                                        model_params=model_tuple, 
                                        reset_params=reset_tuple,
                                        adjM=adjM,
                                        currents=Iapp,
                                        )
    
    t_all = ts
    spike_times = t_all[spikes.astype(bool)]
    # return spike times
    if evaluate and not plot:
        return spike_times
    
    if plot:
        ninputs = len(Iapp.shape)
        ROWS = 3 if not evaluate else 4
        HR = [1,.2,.5] if not evaluate else [.8,.3,.5, .75]
        f, ax = plt.subplots(nrows=ROWS, figsize = (12, 2.5*ROWS), 
                             sharex='all',
                             gridspec_kw={'height_ratios': HR})
        ax[0].plot(t_all, V_all, color = 'k')
        ax[0].set_ylabel('Membrane potential (mV)')

        ax[1].plot(t_all, w_all, color = 'crimson')
        ax[1].set_ylabel('Adaptation (w)')

        if ninputs > 1:
            # show only top 10 most strongly connected
            if adjM is not None:
                neurons_sorted = np.argsort(adjM)[-10:]
            else:
                neurons_sorted = np.arange(ineuron-5, ineuron+5)
            hm = sns.heatmap(data = Iapp.T[neurons_sorted,:], ax = ax[-1], 
                             cbar_kws={'location':'top'})
            ax[-1].set_yticks(ticks = np.arange(neurons_sorted.size)[1::2], labels = neurons_sorted[1::2])
        else:
            ax[-1].plot(t_all, all_curr, # *1000 conversion to nA for plotting if applied current in nA!!! 
                    color = 'b')
        ax[-1].set_ylabel('Synaptic Input')
        ax[-1].set_xlabel('Time (ms)')

        if evaluate:
            assert eval_array is not None, 'To evaluate AND plot, need eval_array of MLspike output!'
            pred_array = bin_spikes(spts_ms=spike_times, nbins=eval_array.size, Tmax_ms=Tmax, maxcount=3)
            BINS = np.linspace(0, Tmax, eval_array.size + 1)
            bin_centers = (BINS[:-1] + BINS[1:]) / 2   
            ax[2].bar(bin_centers, eval_array, alpha = 0.7, color = 'blue', label = 'Target MLspike output',
                      width = 120
                      )
            ax[2].bar(bin_centers, pred_array, alpha = 0.7, color = 'orange', label = 'Fitted Model output',
                      width = 120
                      )
            ax[2].legend(loc = 1)
            ax[2].set_ylabel('Spikes per imaging frame')
        
        plt.tight_layout()
        plt.show()

    print(f"Total spikes: {len(spike_times)}", flush=True)
