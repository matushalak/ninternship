from mat73 import loadmat as hdf_loadmat
from scipy.io.matlab import loadmat as old_loadmat
from scipy.io.matlab import matfile_version

class Dict_to_Class:
    'recursively gets rid of dictionaries'
    def __init__(self, attrs:dict):
        for att_k, att_v in attrs.items():
            if isinstance(att_v, dict):
                setattr(self, att_k, Dict_to_Class(att_v))
            else:
                setattr(self, att_k, att_v)

class SPSIG:
    ''''
    Works for both SPSIG and SPSIG_Res
    Turns SPSIG.mat file into
    '''
    def __init__(self,
                 SPSIG_mat_path:str): # path to ..._SPSIG.mat file
        version, _ = matfile_version(SPSIG_mat_path)
        if version in (0,1):
            # old version < v7.3 mat file
            SPSIG_dict = old_loadmat(SPSIG_mat_path, simplify_cells = True)
            
        else:
            # Load > v7.3 .mat hdf5 file
            SPSIG_dict = hdf_loadmat(SPSIG_mat_path)
        
        # set attributes to keys of SPSIG_dict
        for k, v in SPSIG_dict.items():
            if k.startswith('__'):
                continue
            
            elif isinstance(v, dict):
                setattr(self, k, Dict_to_Class(v))
            else:
                setattr(self, k, v)

        
# Test out on MAT file of choice
if __name__ == '__main__':
    testSPSIG_mat = input('Enter path to your ..._SPSIG.mat file:\t')

    MySPSIG = SPSIG(SPSIG_mat_path=testSPSIG_mat)
    breakpoint()
