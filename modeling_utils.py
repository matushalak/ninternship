from AUDVIS import AUDVIS, Behavior, load_in_data
from VisualAreas import Areas

import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from collections import defaultdict

import timeit
from tqdm import tqdm

### --- Wrappers for Cython models ---


### --- Optimizer utils ---
def loss(ypred:np.ndarray, y:np.ndarray
         )->float:
    '''
    Coincidence factor, used by Clopath, Jolivet, Gerstener et al.
    
    '''
    pass

### --- Other model utils ---