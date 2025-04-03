# @matushalak
# contains calculations to quantify multisensory enhancement / suppression, as well as, direction selectivity
import numpy as np
from collections import defaultdict


### ---------- MULTISENSORY INTEGRATION CALCULATIONS ---------------
def direction_selectivity(FLUORO_RESP: np.ndarray):
    '''
    Finds direction selectivity and appropriate comparison conditions 
    from Fluorescence responses of neurons to all trial types

    Returns:
    ----------
        (nneurons, 
        nstats {2; mean, std}, 
        ntts {4; pref, orthogonal, MSTcongruent, MSTincongruent}, 
        nvisaud {2; vis, aud}), 4D np.ndarray with fluorescence response information
    -----------
    NOTE: according to Meijer (2017) also neurons that were 
    only direction selective in AV condition should be considered 
    in direction selectivity calculation (not the case now!)
    '''
    # setup trials overview
    vis = (6, 7) # L, R
    aud = (0, 3) # L, R
    
    # preference to congruent trials maps
    vis_map = {
        # (Vl) left preference
        0:{'congruent':1, # AlVl
            'incongruent':4}, # ArVl
        # (Vr) right preference
        1:{'congruent':5, # ArVr
            'incongruent':2} # AlVr
            }
    aud_map = {
        # (Al) left preference
        0:{'congruent':1, # AlVl
            'incongruent':2}, # AlVr
        # (Ar) right preference
        1:{'congruent':5, # ArVr
            'incongruent':4} # ArVl
            }
    modality_maps = (vis_map, aud_map)

    # Prepare results array
    results = np.empty((FLUORO_RESP.shape[0], 2, 4, 2))
    
    # Get preferred trial type
    for imodality, modality_tts in enumerate([vis, aud]):
        # only the trials for xthat modality
        sense_tt = FLUORO_RESP[:,:,modality_tts] 
        # find preference within that modality (take absolute value, if stronger inhibitory response)
        sense_pref = np.argmax(sense_tt[:,0,:].__abs__(), axis = 1) # 0 is LEFT, 1 is RIGHT
        # preferred direction (0 in 3rd dimension)
        results[:, :, 0, imodality] = np.take_along_axis(sense_tt, 
                                                         # broadcasting and fancy indexing
                                                         sense_pref[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        # opposite direction (1 in 3rd dimension)
        results[:, :, 1, imodality] = np.take_along_axis(sense_tt, 
                                                         # broadcasting and fancy indexing
                                                         1 - sense_pref[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        
        # choose the proper direction -> congruent / incongruent map for each modality
        mod_map = modality_maps[imodality]
        # lookup tables
        congruent_lookup = np.array([mod_map[0]['congruent'], mod_map[1]['congruent']])
        incongruent_lookup = np.array([mod_map[0]['incongruent'], mod_map[1]['incongruent']])
        
        # congruent MST trials (2 in 3rd dimension)
        congruent_tts = congruent_lookup[sense_pref]
        results[:, :, 2, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         congruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        # incongruent MST trials (3 in 3rd dimension)
        incongruent_tts = incongruent_lookup[sense_pref]
        results[:, :, 3, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         incongruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
    
    assert not np.isnan(results).any(), 'Output array should not contain NaNs!'
    return results

def DSI(neuron_preferred: np.ndarray, neuron_orth:np.ndarray,
        window : tuple[int, int] = (16, 32)) -> float:
    '''
    Calculates Direction-selectivity index (DSI) for one neuron
    defined as in Meijer et al. (2017):
        DSI = (µ_pref - µorth) / √[(sigma_pref + sigma_orth)/2]
    '''
    pass


def RCI(preferred_AV:np.ndarray, preferred_1MOD:np.ndarray,
        window : tuple[int, int] = (16, 32)) -> float:
    '''
    Calculates Response-change index (RSI) for one neurondefined as in Meijer et al. (2017):
        RCI = (Fav - Fv) / (Fav + Fv) for with preferred visual direction
        RCI = (Fav - Fa) / (Fav + Fa) for with preferred auditory direction
    
    Fx is defined as the average fluorescence response to that stimulus over stimulus bins and across all trials
    '''
    pass


# TODO: if we want to say that they are direction TUNED (significantly) [and only look at those neurons]
def DSI_threshold(trial_labels : np.ndarray | list, all_signals) -> float:
    '''
    Average 99th percentile of trial-label shuffled DSI distribution as discussed in Meijer et al. (2017)
    '''
    pass