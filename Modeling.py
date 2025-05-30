#@matushalak
import numpy as np
import scipy.integrate as integr
import scipy.optimize as optim
import scipy as sp

import timeit
from tqdm import tqdm

class CadEx:
    ''''
    Adaptive exponential integrate and fire model for
    in-vivo 2-photon calcium imaging data

    Input
    '''
    def __init__(self,
                 SFmodel:float,
                 model_runtime_sec:float,
                 spikes:np.ndarray,
                 calcium:np.ndarray,
                 SFreal:float,
                 session_name:str,
                 session_number:int | None = None,
                 frametimes:np.ndarray | None = None,
                 event_times:np.ndarray | None = None,
                 msdelays:np.ndarray | None = None,
                 fixed_params: dict[str:float] | None = None
                 ):
        pass
    

    def train_test(self, split:float
                   )->tuple[np.ndarray, np.ndarray]:
        pass


    def upsample(self
                 )->tuple[np.ndarray, np.ndarray]:
        pass


