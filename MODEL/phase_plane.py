#@matushalak
import numpy as np
import matplotlib.pyplot as plt

def vnull(params:dict,V:np.ndarray, I:float)->np.ndarray:
    gL = params['gL']     # nS
    EL = params['EL']    # mV
    VT = params['VT']    # mV
    DeltaT = params['DeltaT']  # mV SLOPE

    return -gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) + I
    

def wnull(params:dict, V:np.ndarray)->np.ndarray:
    return params['a'] * (V - params['EL'])


def AdExPP(V_all:np.ndarray, w_all:np.ndarray, 
           currents:np.ndarray,
           params:dict,
           I_baseline:float = 0.0,
           ):
    '''
    Depiction of solution in AdEx phase-plane
    '''
    f,ax = plt.subplots(figsize = (5,5))

    nullclinesgrid = np.linspace(np.min(V_all) + .05*np.min(V_all), 
                                 -40, 
                                 1000)
    # V nullcline
    VNULLmin = vnull(params=params, V=nullclinesgrid, I = I_baseline)
    extreme = np.argmax(currents.__abs__())
    VNULLmax = vnull(params=params, V=nullclinesgrid, I = currents[extreme])
    # w nullcline
    WNULL = wnull(params=params, V=nullclinesgrid)
    
    spike_indices = np.where(V_all >= params['Vpeak']-1e-3)[0]+1 # get the after spike_resets
    # plot nullclines
    ax.plot(nullclinesgrid, VNULLmin, color = 'grey', linestyle = '--', label = f'V-nullcline (I = {np.min(1e-3*currents).round(2)} nA)')
    ax.plot(nullclinesgrid, VNULLmax, color = 'k',  label = f'V-nullcline (I = {np.max((1e-3)*currents).round(2)} nA)')
    ax.plot(nullclinesgrid, WNULL, color = 'r', label = 'w-nullcline')

    # plot solutions
    # plot initial trajectory
    # starting point=stable equilibrium
    
    # spike_indices = spike_indices[spike_indices < len(V_all)//2]
    spike_indices = spike_indices.tolist() + [len(V_all)]
    alpharange = np.linspace(0.3, 1, len(spike_indices))
    
    ax.scatter(V_all[0], w_all[0], color = 'blue', marker='x')
    ax.plot(V_all[:spike_indices[0]-2], w_all[:spike_indices[0]-2], color = 'green', alpha = alpharange[0])
    
    for i, (si, si2) in enumerate(zip(spike_indices, spike_indices[1:])):
        ALPHA = alpharange[i]
        ax.scatter(V_all[si], w_all[si], color = 'green', marker='s', alpha=ALPHA)
        ax.plot(V_all[si:si2-2], w_all[si:si2-2], color = 'green', alpha = ALPHA)

    ax.set_ylim(min(np.min(w_all), np.min(VNULLmin), np.min(VNULLmax)), 
                max(np.max(w_all) + 100, 3*np.min(VNULLmin), 3*np.min(VNULLmax)))
    ax.set_xlim(np.min(V_all) + .025*np.min(V_all), -40)
    ax.set_xlabel('V (mV)')
    ax.set_ylabel('w')
    # ax.legend(loc = 1)
    f.tight_layout()
    plt.savefig(f'PHASE PLANE {params}.png', dpi = 300)
    plt.show()

