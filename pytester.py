import numpy as np
import matplotlib.pyplot as plt

screen = ears = time = 16

for linear in [np.linspace(0, 1, screen), np.linspace(1, 0, screen)]:
    exponential = np.exp(-5*linear)
    time = np.linspace(1, 0, screen)
    ident = np.eye(screen)

    barstim = ident
    audstim = (exponential + exponential[::-1] ) * ident
    
    f, ax = plt.subplots(nrows=2, ncols=2, figsize = (9, 9))
    # How stimulus moves
    ax[0,0].plot(linear[::-1], time, color = 'k')
    ax[0,0].set_xlabel('Bar location (0:left, 1:right)')
    ax[0,0].set_ylabel('Time since trial start (s)')
    ax[0,0].invert_yaxis()
    # How modelled with matrix
    ax[0,1].imshow(barstim if linear[0] == 0 else barstim[::-1, :])
    ax[0,1].set_xlabel('Visual location predictors (0-15)(left - right)')
    ax[0,1].set_ylabel('Frames since trial start')

    # How sound moves
    ax[1,0].plot(exponential, time, label = 'right speaker')
    ax[1,0].plot(exponential[::-1], time, label = 'left speaker')
    ax[1,0].plot(exponential + exponential[::-1], time, label = 'overall intensity')
    ax[1,0].set_xlabel('Tone intensity (0:min, 1:max)')
    ax[1,0].set_ylabel('Time since trial start (s)')
    ax[1,0].legend(loc=5)
    ax[1,0].invert_yaxis()
    # How modelled with matrix
    ax[1,1].imshow(audstim if linear[0] == 0 else audstim[::-1, :])
    ax[1,1].set_xlabel('Auditory location predictors (0-15)(left - right)')
    ax[1,1].set_ylabel('Frames since trial start')
    plt.tight_layout()
    plt.show()