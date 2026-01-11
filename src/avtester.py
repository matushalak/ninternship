import numpy as np
import matplotlib.pyplot as plt
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.VisualAreas import Areas
import seaborn as sns

#%% general tester
avs:list[AUDVIS] = load_in_data('pre')
for av in avs:
    AR = Areas(av)
    # Quick distance - signal correlation analysis
    A = AR.adjacencyMATRIX(signal=av.zsig)

    # flat
    f, ax = plt.subplots(
        # nrows=len(av.session_neurons), 
        ncols=3,
        figsize = (18,6))
    for si, (start, stop) in enumerate(av.session_neurons):
        session = slice(start, stop)
        Asess = A[session, session, :]
        
        # XXX show correlations
        hm = sns.heatmap(Asess[:,:,1])
        plt.show()
        
        Asess = np.tril(Asess) # get rid of reduntant upper triangle
        Aflat = Asess.reshape(-1, 2)
        dists, corrs = Aflat[:,0], Aflat[:, 1]
        sns.histplot(x=dists, ax = ax[0], kde=True)
        ax[0].set_xlabel('Distance (a.u.)')
        sns.histplot(y=corrs, ax = ax[1], kde=True)
        ax[1].set_ylabel('Correlation')
        sns.histplot(x = dists, y = corrs, ax = ax[2], fill=True)
        ax[2].set_xlabel('Distance (a.u.)')
        ax[2].set_ylabel('Correlation')
    plt.tight_layout()
    plt.show()

#%% Quick script to check adaptation or any systematic trends in signal over the session
avs:list[AUDVIS] = load_in_data('both')
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (20,10))

for iax, ax in enumerate(axs.flatten()):
    AV  = avs[iax]
    ax.set_title(AV.NAME)
    # get signal for all neurons
    tsig = AV.baseline_correct_signal(AV.signal)
    # FRs
    blFRs = tsig[:,16:32,:].max(axis = 1)
    
    session_indices = AV.session_neurons
    sess_names = [AV.sessions[sess]['session'] for sess in AV.sessions]
    
    print(iax)
    # range of trials
    x = np.arange(tsig.shape[0])
    
    # go through sessions
    for (sstart, sstop), sname in zip(session_indices, sess_names):
        meanFR = blFRs[:,sstart:sstop].mean(axis = 1)
        minFR = blFRs[:,sstart:sstop].min(axis = 1)
        maxFR = blFRs[:,sstart:sstop].max(axis = 1)
        sdFR = blFRs[:,sstart:sstop].std(axis = 1)

        erb = sdFR
        # erb = np.array([minFR.__abs__(), maxFR])
        ax.errorbar(x = x, y = meanFR, yerr = erb ,label = sname,
                    errorevery=(sess_names.index(sname)*3, 35), 
                    linestyle = 'None')
        ax.scatter(x = x[sess_names.index(sname)*3::35], y = meanFR[sess_names.index(sname)*3::35], marker='o', s = 10)
    
    ax.set_ylabel('Mean FR (SD) across neurons in a session / trial')
    ax.set_xlabel('Trial from first (0) to last (720)')
    ax.legend(loc = 1, fontsize = 4)

plt.tight_layout()
plt.savefig('trials_check.png', dpi = 300)
plt.show()
