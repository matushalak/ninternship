#@matushalak
# main class that holds and organizes data for the Audio-visual task of Dark-rearing experiment at Levelt Lab
# requires python > 3.8, preferably newer for := to function
# aggregates all neurons across recording sessions of same group
from utils import group_condition_key
from collections import defaultdict
from SPSIG import SPSIG, Dict_to_Class
from numpy import ndarray, array, stack, save, dstack, zeros_like, zeros
from pandas import DataFrame, concat
import pickle
from multiprocessing import Pool, cpu_count

class AUDVIS:
    def __init__(self,
                 neuron_indexing:dict,
                 session_indexing:dict,
                 ROIs:DataFrame,
                 SIG:ndarray,
                 Z:ndarray):
        ABA_regions = ['VISpm', 'VISam', 'VISa', 'VISrl', 'VISal', 'VISl', 'VISp']
        pass

# necessary for defining the default dict because lambdas can't be pickled
def default_neuron_index():
    return {'overall_id':0,
            'specific_id':('', 0),
            'region':0,
            'trialIDs':array([]),
            'n_trials':0,
            'facemap_corrected':False}

def default_session_index():
    return {'n_neurons':0,
            'trialIDs':array([]),
            'n_trials':0,
            'behavior':None, #instance of Behavior class
            'session':''}

class CreateAUDVIS:
    def __init__(self,
                 files:list[str],
                 name:str):
        self.NAME = name
        # SPSIG or SPSIG_Res files to be combined across sessions in same group and condition
        self.files = sorted(files)
        # Neuron index has the follwing structure:
        # - overall neuronID int (0:n_neurons in given condition across animals)
        #   - 'overall_id'  : int (0:n_neurons in given condition across animals)
        #   - 'specific_id' : tuple(str, int) ({RecordingName}, {neuronID}) (1:n_neurons in given recording in given animal)
        #   - 'region' : int | str (based on AllenAllign.m script gives the area in Visual system in which this neuron resides)
        #   - 'trialIDs' : ndarray (indicates which trials correspond to what condition for this particular neuron)
        #   - 'n_trials': int (indicates number of trials that this particular neuron was involved in)
        #   - 'facemap_corrected' : bool (indicates whether the signals of this neuron have been corrected with facemap information)
        self.neuron_index = defaultdict(default_neuron_index)
        # TODO: explanation for  session index
        self.session_index = defaultdict(default_session_index) 
        
        # main step in CreateAUDVIS class
        self.ROIs, self.SIG, self.Z = self.update_index_and_data()

        # save files that we will use when loading AUDVIS
        self.ROIs.to_pickle(f'{self.NAME}_rois.pkl')
        save(f'{self.NAME}_sig.npy', self.SIG)
        save(f'{self.NAME}_zsig.npy', self.Z)
        
        with open(f'{self.NAME}_indexing.pkl', 'wb') as indx_f:
            pickle.dump({'neuron_index':self.neuron_index,
                         'session_index':self.session_index}, indx_f)

    def update_index_and_data(self):
        roi_info = [] # collect dataframes here
        signal_corrected = [] # collect neuropil corrected ∆F/F signal for all sessions here
        signal_z = [] # collect z-score (over entire session) of neuropil corrected ∆F/F signal for all sessions here

        for session_i, file in enumerate(self.files):
            sp = SPSIG(file)

            # this is consistent across all neurons in one recording
            recording_name = sp.info.strfp.split('\\')[-1]
            n_neurons = sp.Res.CaSig.shape[-1]
            trialIDs = sp.info.Stim.log.stim
            n_trials = len(trialIDs)
            behavior = Behavior(sp.Res.speed, sp.Res.facemap) if hasattr(sp.Res, 'facemap') else (Behavior(sp.Res.speed) if hasattr(sp.Res, 'speed') else None)
            
            # session index
            self.session_index[session_i]['n_neurons'] = n_neurons
            self.session_index[session_i]['trialIDs'] = trialIDs[:720] # hardcoded because one session had more somehow
            self.session_index[session_i]['n_trials'] = n_trials
            self.session_index[session_i]['behavior'] = behavior
            self.session_index[session_i]['session'] = recording_name

            # this is per-neuron
            roi_info.append(roiDF := DataFrame(sp.info.rois))
            signal_corrected.append(sig := sp.Res.CaSigCorrected[:,:720,:])
            
            if hasattr(sp.Res, 'CaSigCorrected_Z'):        
                signal_z.append(sp.Res.CaSigCorrected_Z[:, :720, :])
            else:
                signal_z.append((sig - sig.mean(axis = (0,1)))/sig.std(axis = (0,1))) # Z-score calculation

            for neuron_i_session in range(n_neurons):
                neuron_i_overall = neuron_i_session if session_i == 0 else max(list(self.neuron_index)) + 1
                self.neuron_index[neuron_i_overall]['overall_id'] = neuron_i_overall
                self.neuron_index[neuron_i_overall]['specific_id'] = (recording_name, neuron_i_session)
                self.neuron_index[neuron_i_overall]['region'] = roiDF.ABAregion[neuron_i_session]
                self.neuron_index[neuron_i_overall]['trialIDs'] = trialIDs
                self.neuron_index[neuron_i_overall]['n_trials'] = n_trials
            
            print(f'Session {recording_name}, with {n_neurons} ROIs DONE!')
        
        # create the IMPORTANT DFs & Arrays
        return (concat(roi_info), # DataFrane with ABA info & contours & locations
                dstack(signal_corrected), # ndarray Neuropil corrected ∆F/F
                dstack(signal_z)) # ndarray Z-Scored (over entire signal) Neuropil corrected ∆F/F

        

class Behavior:
    def __init__(self,speed:ndarray, 
                 facemap:Dict_to_Class|None = None):
        self.running = speed
        if facemap: # some sessions don't have video & facemap
            self.whisker = facemap.motion
            self.blink = facemap.blink
            self.pupil = facemap.eyeArea
    
    # TODO
    def regress_out(signal:ndarray)->ndarray:
        pass


# for multiprocessing
def run_CreateAUDVIS(args):
    g, g_name = args
    return CreateAUDVIS(g, g_name)


# if ran as a script, initializes 4 AUDVIS classes with loaded data and saves them
if __name__ == '__main__':
    answ = input('Do you wish to create (and overwrite) the .npy and .pkl files for your data? (Y/N)   ')
    
    if answ.lower() in ('y', 'yes', 'true'):
        print('Please select the root directory with your folders containing data to analyze')
        # will ask for directory from which to load
        # g1_files, g2_files = group_condition_key() # general
        g1_files, g2_files = group_condition_key(root = '/Volumes/my_SSD/NiNdata/data')

        # these will be the 4 instances of the AUDVIS class
        g1pre, g1post = g1_files['pre'], g1_files['post']
        g2pre, g2post = g2_files['pre'], g2_files['post']
        names = ['g1pre', 'g1post', 'g2pre', 'g2post']

        # 1 core
        # for g, g_name in zip((g1pre, g1post, g2pre, g2post), names):
        #     av = CreateAUDVIS(g, g_name)

        # multiprocessing on 4 cores
        params = list(zip([g1pre, g1post, g2pre, g2post], names))
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(run_CreateAUDVIS, params)

    else:
        print('Okay, not changing anything ;)')