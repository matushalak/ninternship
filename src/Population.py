import sklearn.decomposition as decomp
from src.AUDVIS import Behavior, AUDVIS, load_in_data
from src.analysis_utils import general_separate_signal, trialsSessionToNeurons
import matplotlib.pyplot as plt
import numpy as np

AV = load_in_data('pre')[0]

def organize_by_trials(df:np.ndarray,
                       trial_types:np.ndarray|list,
                       trial_type_combinations:list[tuple]|None = None
                       )->np.ndarray:
    separated_dict:dict = general_separate_signal(df, trial_types, trial_type_combinations)
    pass

trial_groups:list[tuple[int]] = [(6,7), (0,3), (1,5), (2,4)],
trial_group_labels:list[str] = ['V', 'A', 'AV+', 'AV-'],
print(AV.trials.shape, AV.session_neurons)
tsn = trialsSessionToNeurons(AV.trials, AV.session_neurons)
print(tsn.shape)

x = AV.baseline_correct_signal(AV.zsig) # (trials, ts, neurons)
neurx = x.transpose((2,0,1)) # (neurons, trials, ts)
# neurons are individual samples
neurons, trials, ts = neurx.shape
neurx = neurx.reshape((neurons, -1)) # (neurons, trials*ts)
kPCA = decomp.KernelPCA(n_components=ts, kernel='rbf')
PCA = decomp.PCA(n_components=3)
# Xtransformed = kPCA.fit(X = neurx).transform(neurx)
Xtransformed = PCA.fit(X = neurx).transform(neurx)
# plt.plot(Xtransformed.T); plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')  
ax.scatter(Xtransformed[:, 0], Xtransformed[:, 1], Xtransformed[:, 2]); plt.show()
print(Xtransformed.shape)
