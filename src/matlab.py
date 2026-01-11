# Functions for light-weight modification of matlab files
from src.SPSIG import SPSIG, Dict_to_Class
from glob import glob
import os
from joblib import Parallel, delayed
from scipy.io.matlab import matfile_version
import h5py
import numpy as np

# Datasets
ALL = '/Volumes/my_SSD/darkRearingData'
TONES = '/Volumes/my_SSD/darkRearingData/pureTones'
GRATINGS = '/Volumes/my_SSD/darkRearingData/gratingWarped'
TEST = '/Volumes/my_SSD/test'

# Data fields of interest
FIELDS = {
          'dF':('Res', 'CaSigCorrected'),
          'running':('Res', 'speed'),
          'whisker':('Res', 'facemap', 'motion'),
          'pupil':('Res', 'facemap', 'eyeArea'),
          'time':('Res', 'ax'),
          'TTs':('info', 'Stim', 'log'),
          'TT_explain':('info', 'Stim', 'Parameters', 'stimuli')
          }

WRITEFIELDS = {'dFClean':'CaSigCorrectedGLMclean'}

class MatFile:
    pass

def find_mat(root:str = '/Volumes/my_SSD/darkRearingData/pureTones/darkReared_baseline', 
             file_ending:str = 'SPSIG_Res.mat')->list[str]:
    # find all files with specified ending within root directory
    allResMats = glob(os.path.join(root,  '**/*' + file_ending), recursive=True)
    print(len(allResMats))
    return allResMats

def matgrab(matfile:str, fields:dict[str, tuple[str]]
            )->MatFile:
    '''
    Grabs a desired matlab variables and stores them in an object

    args:
        matfile:str - path to .mat file
        fields:dict - fields to grab 
            {new_field_name : (outer_struct, inner_struct, field), 
            another_field_name : ...}
    
    returns:
        object: DataClass - attributes = new fields
    '''
    # load full mat file
    full = SPSIG(matfile)
    # storage class
    out = MatFile()
    for name, pth in fields.items():
        data = full
        for struct in pth:
            assert hasattr(data, struct), (
                f'The file {matfile} does not contain this structure leading to a field {pth}')
            data = getattr(data, struct)
        
        if name == 'dF':
            # Z-score (per neuron)
            mean = np.mean(data, axis=(0,1), keepdims=True)
            std = np.std(data, axis=(0,1), keepdims=True)
            data = (data - mean) / std
        setattr(out, name, data)
    return out

def matwrite(matfile:str, fields:dict, obj:MatFile):
    '''
    Writes desired object attributes into a matlab file
    '''
    version, _ = matfile_version(matfile)
    if version == 2: 
        print(".mat version > V7.3: hdf5 format, safe to append computed properties")
    else: 
        print("UNSAFE! .mat version <V7.3: proprietary .mat format, appending new values requires rewriting whole file")
        return
    
    # append to existing hdf5 file
    with h5py.File(matfile, 'a') as mf:
        for attr, fieldname in fields.items():
            arr = np.asarray(getattr(obj, attr), dtype=np.float64)  # MATLAB "double"
            if fieldname in mf:
                del mf[fieldname]
            d = mf.create_dataset(fieldname, data=arr)
            d.attrs['MATLAB_class'] = np.bytes_('double')
    
    print(f'Write into {matfile} completed!')

if __name__ == '__main__':
    mats = find_mat(TEST)
    M = matgrab(mats[0], fields=FIELDS)
    M.dFClean = M.dF.mean(axis = 1)
    print(M.__dict__.keys())
    matwrite(mats[0], WRITEFIELDS, M)

    # for later
    # Parallel(n_jobs=-1, backend='threading')(delayed(SPSIG)(f, True)for f in mats)