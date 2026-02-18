from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from typing import Literal

from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.Analyze import Analyze
from src.VisualAreas import Areas

    
class Decoder:
    def __init__(self):
        pass
    
def decode(pre_post: Literal['pre', 'post', 'both'] = 'pre',
           decoder:object = LinearDiscriminantAnalysis):
    # Load in all data and perform necessary initial calculations
    AVS : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]
    breakpoint()

    # TEst
    an = ANS[0]
    # trial labels
    y = an.allTrials # (ntrials, nsessions)


    DC = Decoder()

if __name__ == '__main__':
    decode()