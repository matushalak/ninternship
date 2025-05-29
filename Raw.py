from SPSIG import SPSIG
import utils as utl
import numpy as np
from joblib import Parallel, delayed

def load_SPSIGvars(spsgpath:str, vars:list[str]|None = None
                   )->SPSIG:
    SPSG = SPSIG(spsgpath)
    print(spsgpath)
    print(f'MLspike: {hasattr(SPSG, 'deconCorrected')}, {np.unique(SPSG.deconCorrected)}')
    print(f'FaceMap: {hasattr(SPSG, 'facemapTraces')}, {dir(SPSG.facemapTraces) 
                                                        if hasattr(SPSG, 'facemapTraces') else ''}')


g1spsig_files, g2spsig_files = utl.group_condition_key(root = '/Volumes/my_SSD/NiNdata/data',
                                                        raw=True)
g1pre, g1post = g1spsig_files['pre'], g1spsig_files['post']
g2pre, g2post = g2spsig_files['pre'], g2spsig_files['post']
sessions = [g1pre, g1post, g2pre, g2post]
session_ranges = []
indx = 0
for session in sessions:
    session_ranges.append((indx, indx := indx + len(session)))

all_sessions = g1pre + g1post + g2pre + g2post

res = Parallel(n_jobs=-1)(delayed(load_SPSIGvars)(ses) 
                          for ses in all_sessions)