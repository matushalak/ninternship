# @matushalak
# contains calculations to quantify multisensory enhancement / suppression, as well as, direction selectivity
import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import artist
from matplotlib.patches import Patch
from src.utils import get_sig_label, add_sig
from typing import Literal
import re
from src import PYDATA, MIPLOTSDIR

### ---------- MULTISENSORY INTEGRATION CALCULATIONS ---------------
def direction_selectivity(FLUORO_RESP: np.ndarray):
    '''
    Finds direction selectivity and appropriate comparison conditions 
    from Fluorescence responses of neurons to all trial types

    Returns:
    ----------
        (nneurons, 
        nstats {2; mean, std}, 
        ntts {6; pref, orthogonal, prefMSTcongruent, prefMSTincongruent , nonprefMSTcongruent, nonprefMSTincongruent}, 
        nvisaud {2; vis, aud}), 4D np.ndarray with fluorescence response information
    -----------
    NOTE: according to Meijer (2017) also neurons that were 
    only direction selective in AV condition should be considered 
    in direction selectivity calculation (not the case now!)
    '''
    FLUORO_RESP = FLUORO_RESP[:,:2,:]
    # setup trials overview
    vis = (6, 7) # L, R
    aud = (0, 3) # L, R
    mod_trials = (vis, aud)
    
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
    results = np.full((FLUORO_RESP.shape[0], 2, 6, 2), np.nan)
    index_results = {'VIS':dict(),
                     'AUD':dict()}
    modnames = ['VIS', 'AUD']
    # Get preferred trial type
    for imodality, modality_tts in enumerate(mod_trials):
        # only the trials for xthat modality
        F_sense_tt = FLUORO_RESP[:,:,modality_tts] 
        # absolute value of the MEAN fluorescence response for each neuron
        F_abs_sense_tt = F_sense_tt[:,0,:].__abs__()
        # Create a boolean mask identifying rows that are NOT all NaN
        non_nan_mask = ~np.all(np.isnan(F_abs_sense_tt), axis=1)
        # Prepare sense_pref as an integer index array of length nneurons.
        sense_pref = np.zeros(F_abs_sense_tt.shape[0], dtype=int)
        # find preference within that modality (take absolute value, if stronger inhibitory response)
        sense_pref[non_nan_mask] = np.nanargmax(F_abs_sense_tt[non_nan_mask], axis = 1) # 0 is LEFT, 1 is RIGHT
        # preferred direction (0 in 3rd dimension)
        results[:, :, 0, imodality] = np.take_along_axis(F_sense_tt, 
                                                         # broadcasting and fancy indexing
                                                         sense_pref[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        # opposite direction (1 in 3rd dimension)
        results[:, :, 1, imodality] = np.take_along_axis(F_sense_tt, 
                                                         # broadcasting and fancy indexing
                                                         # 1 - sense_pref is nonpref
                                                         1 - sense_pref[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        
        # choose the proper direction -> congruent / incongruent map for each modality
        mod_map = modality_maps[imodality]
        # lookup tables for appropriate congruent / incongruent trials for each direction
        congruent_lookup = np.array([mod_map[0]['congruent'], mod_map[1]['congruent']])
        incongruent_lookup = np.array([mod_map[0]['incongruent'], mod_map[1]['incongruent']])
        
        # prefcongruent MST trials (2 in 3rd dimension)
        prefcongruent_tts = congruent_lookup[sense_pref]
        results[:, :, 2, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         prefcongruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        # prefincongruent MST trials (3 in 3rd dimension)
        prefincongruent_tts = incongruent_lookup[sense_pref]
        results[:, :, 3, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         prefincongruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        
        # nonprefcongruent MST trials (4 in 3rd dimension)
        nonprefcongruent_tts = congruent_lookup[1-sense_pref]
        results[:, :, 4, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         nonprefcongruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()
        # nonprefincongruent MST trials (5 in 3rd dimension)
        nonprefincongruent_tts = incongruent_lookup[1-sense_pref]
        results[:, :, 5, imodality] = np.take_along_axis(FLUORO_RESP, 
                                                         # broadcasting and fancy indexing
                                                         nonprefincongruent_tts[:, np.newaxis, np.newaxis],
                                                         axis = 2).squeeze()

        # get indices for preferred stimuli and corresponding congruent / incongruent trials
        preferences = sense_pref.copy()
        left_pref = (preferences == 0)
        right_pref = (preferences == 1)
        # 0 = 6 for visual and 0 = 0 for auditory
        preferences[left_pref] = mod_trials[imodality][0]
        # 1 = 7 for visual and 1 = 3 for auditory
        preferences[right_pref] = mod_trials[imodality][1]
        assert all(np.unique(preferences) == mod_trials[imodality]), 'Incorrect indices for preferred trial-type'
        
        # same for nonpreferred
        nonpreferences = (1-sense_pref).copy()
        nonpreferences[(nonpreferences == 0)] = mod_trials[imodality][0]
        nonpreferences[(nonpreferences == 1)] = mod_trials[imodality][1]
        assert all(np.unique(nonpreferences) == mod_trials[imodality]), 'Incorrect indices for preferred trial-type'
        
        # add index results
        modname = modnames[imodality]
        index_results[modname]['preferred'] = preferences
        index_results[modname]['prefcongruent'] = prefcongruent_tts
        index_results[modname]['prefincongruent'] = prefincongruent_tts
        index_results[modname]['nonpreferred'] = nonpreferences
        index_results[modname]['nonprefcongruent'] = nonprefcongruent_tts
        index_results[modname]['nonprefincongruent'] = nonprefincongruent_tts

    assert not np.isnan(results).any(), 'Output array should not contain NaNs!'
    return results, index_results

def DSI(neuron_preferred: np.ndarray, neuron_orth:np.ndarray) -> float:
    '''
    Calculates Direction-selectivity index (DSI) for one neuron
    defined as in Meijer et al. (2017) for Orientation selectivity (OSI):
        OSI = (µ_pref - µorth) / √[(sigma_pref + sigma_orth)/2]
    or in Olcese et al., (2013) for Direction selectivity (DSI):
        DSI = (µ_pref - µ_nonpref) / (µ_pref + µ_nonpref)
    '''
    return (neuron_preferred[:,0,:].__abs__() 
            - neuron_orth[:,0,:].__abs__()
            ) / np.sqrt((neuron_preferred[:,1,:]**2 + neuron_orth[:,1,:]**2) / 2) # Meijer 2017 OSI
            # ) / (neuron_preferred[:,0,:].__abs__() + neuron_orth[:,0,:].__abs__()) # Olcese 2013 DSI

def RCI(preferred_AV:np.ndarray, preferred_1MOD:np.ndarray) -> float:
    '''
    Calculates Response-change index (RSI) for one neurondefined as in Meijer et al. (2017):
        RCI = (Fav - Fv) / (Fav + Fv) for with preferred visual direction
        RCI = (Fav - Fa) / (Fav + Fa) for with preferred auditory direction
    
    Fx is defined as the average fluorescence response to that stimulus over stimulus bins and across all trials
    Only -1 to 1 if Fx is always non-negative
    '''
    return (preferred_AV[:,0,:].__abs__() 
            - preferred_1MOD[:,0,:].__abs__()
            ) / (preferred_AV[:,0,:].__abs__() + preferred_1MOD[:,0,:].__abs__())


def Quantification(MIdata:pd.DataFrame, 
                   X_VAR:str, Y_VAR:str, HUE_VAR:str,
                   kind: Literal['within', 'between'] = 'within'):
    ngroups = MIdata[HUE_VAR].nunique()
    comparison_values = []
    groups_list = []
    for ig, g in enumerate(MIdata[HUE_VAR].unique()):
        vecs = MIdata.loc[MIdata[HUE_VAR] == g][[X_VAR, Y_VAR]].to_numpy()
        # x and y values
        xVALS = vecs[:,0]
        yVALS = vecs[:,1]
        # For RCI comparisons, we take the absolute value for quantifications
        if 'RCI' in X_VAR or 'RCI' in Y_VAR:
            xVALS = xVALS.__abs__()
            yVALS = yVALS.__abs__()
        
        if kind == 'within':
            # values for (x, y)
            comparison_values.append((xVALS, yVALS))
            groups_list.append(g)
        else:
            if ig == 0:
                comparison_values.append([xVALS])
                comparison_values.append([yVALS])
                groups_list.append([g+X_VAR])
                groups_list.append([g+Y_VAR])
            else:
                comparison_values[0].append(xVALS)
                comparison_values[1].append(yVALS)
                groups_list[0].append(g+X_VAR)
                groups_list[1].append(g+Y_VAR)
    
    assert len(comparison_values) in (2,4, 2*4, 4*4), 'Current version compares between 2 or 4 groups (not according to region) or 8 or 16 groups (per region)'
    match kind:
        # means comparison
        # TODO: generalize code
        case 'within':
            (g1pref, g1MST), (g2pref, g2MST) = comparison_values
            # Within group (related test - WCOX)
            WCOX1 = spstats.wilcoxon(g1pref, g1MST, alternative='two-sided', nan_policy='omit')
            pval1 = WCOX1.pvalue
            g1pref_mean, g1MST_mean = np.nanmean(g1pref), np.nanmean(g1MST)
            g1pref_SEM, g1MST_SEM = spstats.sem(g1pref, nan_policy='omit'), spstats.sem(g1MST, nan_policy='omit')
            
            WCOX2 = spstats.wilcoxon(g2pref, g2MST, alternative='two-sided', nan_policy='omit')
            pval2 = WCOX2.pvalue
            g2pref_mean, g2MST_mean = np.nanmean(g2pref), np.nanmean(g2MST)
            g2pref_SEM, g2MST_SEM = spstats.sem(g2pref, nan_policy='omit'), spstats.sem(g2MST, nan_policy='omit')

            pvals = np.array((pval1, pval2))
            vars_list = [X_VAR, Y_VAR, X_VAR, Y_VAR]
            return (pvals * len(pvals), #bonferroni correction 
                    np.array((g1pref_mean, g1MST_mean, g2pref_mean, g2MST_mean)), 
                    np.array((g1pref_SEM, g1MST_SEM, g2pref_SEM, g2MST_SEM)),
                    (groups_list[0]+X_VAR, groups_list[0]+Y_VAR, groups_list[1]+X_VAR, groups_list[1]+Y_VAR))

        case 'between':
            (g1A, g2A), (g1B, g2B) = comparison_values
            MWU1 = spstats.mannwhitneyu(g1A, g2A, alternative='two-sided', nan_policy='omit')
            pval1 = MWU1.pvalue
            g1A_mean, g2A_mean = np.nanmean(g1A), np.nanmean(g2A)
            g1A_SEM, g2A_SEM = spstats.sem(g1A, nan_policy='omit'), spstats.sem(g2A, nan_policy='omit')
            
            MWU2 = spstats.mannwhitneyu(g1B, g2B, alternative='two-sided', nan_policy='omit')
            pval2 = MWU2.pvalue
            g1B_mean, g2B_mean = np.nanmean(g1B), np.nanmean(g2B)
            g1B_SEM, g2B_SEM = spstats.sem(g1B, nan_policy='omit'), spstats.sem(g2B, nan_policy='omit')
            
            pvals = np.array((pval1, pval2))
            return (pvals * len(pvals), 
                    np.array((g1A_mean, g2A_mean, g1B_mean, g2B_mean)), 
                    np.array((g1A_SEM, g2A_SEM, g1B_SEM, g2B_SEM)),
                    np.array(groups_list).flatten())
        

        # TODO: brain regions, all 4 groups, etc
        case _:
            raise NotImplementedError


# --------- prepare plotting dataframe ---------
def initializeMIdata()-> dict:
    MIdata_dict = {
        # general information
        'NeuronID':[],
        'Group':[],# collect group_names,
        'BrainRegion':[], # collect which brain region
        
        # collect DSIs
        'DSI (VIS)' : [], 'DSI (AUD)': [], 
        
        # collect RCIs
        'RCI (VISpref congruent)': [], 'RCI (VISpref incongruent)' : [],
        'RCI (AUDpref congruent)': [], 'RCI (AUDpref incongruent)' : [],
        'RCI (VISnonpref congruent)': [], 'RCI (VISnonpref incongruent)' : [],
        'RCI (AUDnonpref congruent)': [], 'RCI (AUDnonpref incongruent)' : [],
        
        # collect fluorescence responses
        # from VIS stimuli perspective
        'pref_VIS':[], 'prefVIS_MST+':[], 'prefVIS_MST-':[],
        'nonpref_VIS':[], 'nonprefVIS_MST+':[], 'nonprefVIS_MST-':[],
        # from AUD stimuli perspective
        'pref_AUD':[], 'prefAUD_MST+':[], 'prefAUD_MST-':[],
        'nonpref_AUD':[], 'nonprefAUD_MST+':[], 'nonprefAUD_MST-':[],
        
        # collect tt indices
        'pref_VIS_idx':[], 'prefVIS_MST+_idx':[], 'prefVIS_MST-_idx':[],
        'nonpref_VIS_idx':[], 'nonprefVIS_MST+_idx':[], 'nonprefVIS_MST-_idx':[],
        'pref_AUD_idx':[], 'prefAUD_MST+_idx':[], 'prefAUD_MST-_idx':[],
        'nonpref_AUD_idx':[], 'nonprefAUD_MST+_idx':[], 'nonprefAUD_MST-_idx':[],}
    return MIdata_dict

def getMIdata(FLUORO_RESP: np.ndarray, group_cond_name: str,
              MIdata_dict: dict[str: list]) -> dict[str: list]:
    '''
    Returns lists which can be iterated over and appended to from different conditions
    '''
    #1) get direction selectivity info
    FRs, index_infos = direction_selectivity(FLUORO_RESP)
    splitFRs = np.split(FRs, indices_or_sections=6, axis = 2)
    pref_stats, nonpref_stats, prefcongruent_stats, prefincongruent_stats = splitFRs[:4] 
    nonprefcongruent_stats, nonprefincongruent_stats = splitFRs[4:]
    
    #2) get direction selectivity index for visual and auditory separately
    DSI_vis, DSI_aud = np.split(DSI(pref_stats, nonpref_stats),
                                    indices_or_sections=2,
                                    axis = 2)
    
    MIdata_dict['Group'] += [group_cond_name] * DSI_vis.size
    MIdata_dict['DSI (VIS)'] += [*DSI_vis.squeeze()]
    MIdata_dict['DSI (AUD)'] += [*DSI_aud.squeeze()]

    #3) Get response change index for visual and auditory separately
    # first for preferred stimulus directions
    # for congruent MST
    RCI_vis_pref_congruent, RCI_aud_pref_congruent = np.split(
                                        RCI(prefcongruent_stats, pref_stats),
                                        indices_or_sections=2,
                                        axis = 2)
    MIdata_dict['RCI (VISpref congruent)'] += [*RCI_vis_pref_congruent.squeeze()]
    MIdata_dict['RCI (AUDpref congruent)'] += [*RCI_aud_pref_congruent.squeeze()]
    # for incongruent MST
    RCI_vis_pref_incongruent, RCI_aud_pref_incongruent = np.split(
                                        RCI(prefincongruent_stats, pref_stats),
                                        indices_or_sections=2,
                                        axis = 2)
    MIdata_dict['RCI (VISpref incongruent)'] += [*RCI_vis_pref_incongruent.squeeze()]
    MIdata_dict['RCI (AUDpref incongruent)'] += [*RCI_aud_pref_incongruent.squeeze()]

    # do the same for nonprefered stimulus directions
    # for congruent MST
    RCI_vis_nonpref_congruent, RCI_aud_nonpref_congruent = np.split(
                                        RCI(nonprefcongruent_stats, nonpref_stats),
                                        indices_or_sections=2,
                                        axis = 2)
    MIdata_dict['RCI (VISnonpref congruent)'] += [*RCI_vis_nonpref_congruent.squeeze()]
    MIdata_dict['RCI (AUDnonpref congruent)'] += [*RCI_aud_nonpref_congruent.squeeze()]
    # for incongruent MST
    RCI_vis_nonpref_incongruent, RCI_aud_nonpref_incongruent = np.split(
                                        RCI(nonprefincongruent_stats, nonpref_stats),
                                        indices_or_sections=2,
                                        axis = 2)
    MIdata_dict['RCI (VISnonpref incongruent)'] += [*RCI_vis_nonpref_incongruent.squeeze()]
    MIdata_dict['RCI (AUDnonpref incongruent)'] += [*RCI_aud_nonpref_incongruent.squeeze()]


    # 4) Save preferred direction and corresponding MST indices
    # Visual
    # preferred
    MIdata_dict['pref_VIS_idx'] += [*index_infos['VIS']['preferred']]
    MIdata_dict['prefVIS_MST+_idx'] += [*index_infos['VIS']['prefcongruent']]
    MIdata_dict['prefVIS_MST-_idx'] += [*index_infos['VIS']['prefincongruent']]
    # nonprefered
    MIdata_dict['nonpref_VIS_idx'] += [*index_infos['VIS']['nonpreferred']]
    MIdata_dict['nonprefVIS_MST+_idx'] += [*index_infos['VIS']['nonprefcongruent']]
    MIdata_dict['nonprefVIS_MST-_idx'] += [*index_infos['VIS']['nonprefincongruent']]
    # Auditory
    # preferred
    MIdata_dict['pref_AUD_idx'] += [*index_infos['AUD']['preferred']]
    MIdata_dict['prefAUD_MST+_idx'] += [*index_infos['AUD']['prefcongruent']]
    MIdata_dict['prefAUD_MST-_idx'] += [*index_infos['AUD']['prefincongruent']]
    # nonpreferred
    MIdata_dict['nonpref_AUD_idx'] += [*index_infos['AUD']['nonpreferred']]
    MIdata_dict['nonprefAUD_MST+_idx'] += [*index_infos['AUD']['nonprefcongruent']]
    MIdata_dict['nonprefAUD_MST-_idx'] += [*index_infos['AUD']['nonprefincongruent']]

    # 4) Save preferred direction and corresponding MST Fluorescence responses
    # Visual
    # preferred
    MIdata_dict['pref_VIS'] += [*pref_stats.squeeze()[:,0,0]]
    MIdata_dict['prefVIS_MST+'] += [*prefcongruent_stats.squeeze()[:,0,0]]
    MIdata_dict['prefVIS_MST-'] += [*prefincongruent_stats.squeeze()[:,0,0]]
    # nonpreferred
    MIdata_dict['nonpref_VIS'] += [*nonpref_stats.squeeze()[:,0,0]]
    MIdata_dict['nonprefVIS_MST+'] += [*nonprefcongruent_stats.squeeze()[:,0,0]]
    MIdata_dict['nonprefVIS_MST-'] += [*nonprefincongruent_stats.squeeze()[:,0,0]]
    # Auditory
    # preferred
    MIdata_dict['pref_AUD'] += [*pref_stats.squeeze()[:,0,1]]
    MIdata_dict['prefAUD_MST+'] += [*prefcongruent_stats.squeeze()[:,0,1]]
    MIdata_dict['prefAUD_MST-'] += [*prefincongruent_stats.squeeze()[:,0,1]]
    # nonpreferred
    MIdata_dict['nonpref_AUD'] += [*nonpref_stats.squeeze()[:,0,1]]
    MIdata_dict['nonprefAUD_MST+'] += [*nonprefcongruent_stats.squeeze()[:,0,1]]
    MIdata_dict['nonprefAUD_MST-'] += [*nonprefincongruent_stats.squeeze()[:,0,1]]

    return MIdata_dict


def prepare_long_format_Areas(out_size : int, 
                              Area_indices:dict[str: np.ndarray]) -> np.ndarray:
    regions = np.full(shape = out_size, fill_value='', dtype='U7')
    for region_name, where_region in Area_indices.items():
        regions[where_region] = str(region_name)
    return regions


def processFULLDFintoLONG(df: pd.DataFrame)->pd.DataFrame:
    # shared id-vars
    id_vars = ['NeuronID','Group','BrainRegion','NeuronType']
    # ---------------------------------------------------
    # 1) Drop all idx columns
    idx_cols = [c for c in df.columns if c.endswith('_idx')]
    df = df.drop(columns=idx_cols)
    # ---------------------------------------------------
    # 2) Melt DSI
    dsi = df.melt(
        id_vars=id_vars,
        value_vars=[c for c in df.columns if c.startswith('DSI')],
        var_name='tmp', value_name='DSI'
    )
    # extract VIS/AUD
    dsi['Modality'] = dsi['tmp'].str.extract(r'DSI \((VIS|AUD)\)')
    dsi = dsi.drop(columns='tmp')
    dsi['Preference'] = None
    dsi['Congruency'] = None
    # ---------------------------------------------------
    # 3) Melt RCI
    rci = df.melt(
        id_vars=id_vars,
        value_vars=[c for c in df.columns if c.startswith('RCI')],
        var_name='tmp', value_name='RCI'
    )
    # parse "RCI (VISpref congruent)"
    rci[['Modality','Preference','Congruency']] = (
        rci['tmp']
        .str.extract(r'RCI \((VIS|AUD)(pref|nonpref) (congruent|incongruent)\)')
    )
    rci = rci.drop(columns='tmp')
    # ---------------------------------------------------
    # 4) Melt FR
    # pick up both unimodal and MST columns
    fr_cols = []
    for c in df.columns:
        if re.match(r'^(pref_|nonpref_)(VIS|AUD)$', c):
            fr_cols.append(c)
        elif re.match(r'^(pref|nonpref)(VIS|AUD)_MST[+-]$', c):
            fr_cols.append(c)

    fr = df.melt(
        id_vars=id_vars,
        value_vars=fr_cols,
        var_name='tmp', value_name='FR'
    )
    # unimodal: "pref_VIS" / "nonpref_AUD"
    um = fr['tmp'].str.extract(r'^(pref|nonpref)_(VIS|AUD)$')
    mask_um = um[0].notna()
    fr.loc[mask_um, 'Preference'] = um.loc[mask_um, 0]
    fr.loc[mask_um, 'Modality']   = um.loc[mask_um, 1]
    fr.loc[mask_um, 'Congruency'] = 'Unimodal'

    # bimodal: "prefVIS_MST+" / "nonprefAUD_MST-"
    bm = fr['tmp'].str.extract(r'^(pref|nonpref)(VIS|AUD)_MST([+-])$')
    mask_bm = bm[0].notna()
    fr.loc[mask_bm, 'Preference'] = bm.loc[mask_bm, 0]
    fr.loc[mask_bm, 'Modality']   = bm.loc[mask_bm, 1]
    # map + → congruent, - → incongruent
    fr.loc[mask_bm, 'Congruency'] = bm.loc[mask_bm, 2].map({'+':'congruent','-':'incongruent'})

    fr = fr.drop(columns='tmp')
    # ---------------------------------------------------
    # 5) Merge everything
    df_long = (
        dsi
        .merge(rci, on=id_vars+['Modality','Preference','Congruency'], how='outer')
        .merge(fr,  on=id_vars+['Modality','Preference','Congruency'], how='outer')
    )
    # final columns
    df_long = df_long[
        ['NeuronID','Group','BrainRegion','NeuronType',
        'Modality','Preference','Congruency',
        'DSI','RCI','FR']
    ]
    return df_long

#------- plotting --------
# TODO: fix with annotator
def scatter_hist_reg_join(MIdata: pd.DataFrame,
                          NAME: str,
                          X_VAR: str, Y_VAR:str, HUE_VAR :str, 
                          kde:bool = False, reg:bool = False, square: bool = False,
                          savedir: str | None = None, 
                          statsmethod: Literal['within', 'between'] = 'within',
                          BrainArea: str | None = None,
                          colmap: dict[str:str] = {'g1pre' : 'royalblue', 
                                                   'g1post': 'teal', 
                                                   'g2pre' : 'tomato', 
                                                   'g2post': 'crimson'},
                          markrs: dict[str:str] = {'g1pre' : 'o', 
                                                   'g1post': '*', 
                                                   'g2pre' : 'v', 
                                                   'g2post': 'd'}):
    
    minmin = min(MIdata[X_VAR].min(), MIdata[Y_VAR].min())
    maxmax = max(MIdata[X_VAR].max(), MIdata[Y_VAR].max())
    diagx = (minmin - .05, maxmax + .05)
    diagy = diagx
    if not square:
        YLIM = (MIdata[Y_VAR].min() - .05, MIdata[Y_VAR].max() + .05)
        XLIM = (MIdata[X_VAR].min() - .05, MIdata[X_VAR].max() + .05)
    else:
        if 'RCI' in NAME:
            YLIM = (-1,1)
            XLIM = (-1,1)
            diagy = YLIM
            diagx = diagy
        else:
            YLIM = diagy
            XLIM = diagx

    g = sns.JointGrid(data=MIdata, x=X_VAR, y=Y_VAR, hue = HUE_VAR, palette= colmap,
                      height=8, ratio=8, space = 0, xlim=XLIM, ylim=YLIM)
    g.ax_joint.plot(diagx, diagy, linestyle = '--', color = 'dimgray')
    g.plot_joint(sns.scatterplot, data = MIdata, alpha = 0.7, style = HUE_VAR, 
                 markers = markrs, s = 18)
    if kde:
        g.plot_joint(sns.kdeplot, levels = 1)
    
    # sns.rugplot(data=MIdata, x = X_VAR, hue=HUE_VAR, ax=g.ax_marg_x, 
    #             height=-.08, clip_on=False, lw = 1, alpha = .2, palette=colmap, legend=False)
    sns.histplot(data=MIdata, x = X_VAR, hue=HUE_VAR, ax=g.ax_marg_x, kde=True, palette=colmap, legend=False)
    
    # sns.rugplot(data=MIdata, x = Y_VAR, hue=HUE_VAR, ax=g.ax_marg_y, 
    #             height=-.08, clip_on=False, lw = 1, alpha = .2, palette=colmap, legend=False)
    sns.histplot(data=MIdata, y = Y_VAR, hue=HUE_VAR, ax=g.ax_marg_y, kde=True, palette=colmap, legend=False)

    if reg:
        for group,gr in MIdata.groupby(HUE_VAR):
            sns.regplot(x=X_VAR, y=Y_VAR, data=gr, scatter=False, ax=g.ax_joint, truncate=False, color = colmap[group])
    
    sns.move_legend(obj = g.ax_joint, loc = 2)
    
    # Quantifications using Mann-Whitney-U test
    pval, means, sems, group_names = Quantification(MIdata, X_VAR, Y_VAR, HUE_VAR, kind=statsmethod)
    # Inset plot
    sig_label_map = {0: 'n.s.', 1: '*', 2: '**', 3: '***'}
    inset_ax = g.figure.add_axes([0.675, 0.675, 0.2, 0.2])
    indices = np.arange(len(group_names))
    # Plot each bar with error bars; use the same colors as in colmap.
    for i, group in enumerate(group_names):
        group = [k for k in colmap if k in group][0]
        if i % 2 == 0:
            sig_text = get_sig_label(pval[i//2], sig_label_map)
            # Connect the means with a dashed line.
            inset_ax.plot(indices[i:i+2], means[i:i+2], linestyle='--', color='k', marker='o', alpha = 0.8)
            # Place the significance label above the highest error bar in the inset.
            x_mid = np.mean(indices[i:i+2])
            # Calculate y_top from the highest of (mean+sem) values.
            y_top = np.mean([np.max(means[i:i+2]), np.mean(means[i:i+2])])
            # Determine a vertical offset relative to the inset's y-range:
            inset_ax.text(x_mid, y_top, sig_text, ha='center', va='bottom', fontsize=12)
        
        # Use colmap to get the bar color; default to gray if not found.
        color = colmap.get(group, 'gray')
        inset_ax.bar(indices[i], means[i], yerr=sems[i],
                     color=color, width=0.6, capsize=4, alpha=0.8)

    # Set x-ticks with group names and adjust font size/rotation.
    inset_ax.set_xticks(indices)
    inset_ax.set_xticklabels(group_names, rotation=60, fontsize=6)
    inset_ax.set_ylabel(f'Mean F response', fontsize=8)
    # breakpoint()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'{NAME}.png'), dpi = 300)
    else:
        plt.savefig(f'{NAME}.png', dpi = 300)
    plt.close('all')

def RCI_dist_plot(RCIs:np.ndarray,
                  ax: artist = plt,
                  xlab: bool = False):
    RCIsorted = np.sort(RCIs)
    pos = np.where(RCIsorted > 0)[0]
    neg = np.where(RCIsorted < 0)[0]
    sign = np.empty_like(RCIsorted, dtype=str)
    sign[pos] = '+'
    sign[neg] = '-'
    perc_pos = round(100 * (len(pos) / len(RCIs)), 1)
    perc_neg = round(100 * (len(neg) / len(RCIs)), 1)
    colors = {'+':'green', '-':'red'}
    df = pd.DataFrame({'Response change index':RCIsorted, 'sign':sign, '_':np.linspace(0,1,RCIsorted.size)})
    bp = sns.barplot(df, x = '_', y = 'Response change index', hue = 'sign', 
                     width = 1, palette=colors, legend=False, ax = ax)
    bp.vlines(RCIsorted.size // 2, -1, 1, linestyles='--', colors = 'dimgray')
    sns.despine(bottom=True)
    bp.set(xticks = [], xticklabels = [])
    if xlab:
        bp.set(xlabel = 'Neurons sorted by\nresponse change index')
    else:
        bp.set(xlabel = None)
    bp.set_ylim((-1,1))
    # Create custom legend handles for the hues 
    legend_handles = [Patch(facecolor=colors['+'], label=f'{perc_pos}% > 0'),
                      Patch(facecolor=colors['-'], label=f'{perc_neg}% < 0')]
    
    ax.legend(handles=legend_handles, loc=2, fontsize = 9)
    plt.tight_layout()
    return df


# TODO: Add proportion histograms & significance testing
def RCI_dist_plots_all(MIdata: pd.DataFrame, 
                       pref:str = 'pref',
                       area: str = 'all',
                       savedir: str | None = None,
                       quantify: bool = True,
                       includedMOD:tuple[str] = ('VIS', 'AUD')):
    ngroups = MIdata['Group'].nunique()
    rci_DFlist = []
    for mod in includedMOD:
        rcifig, rciaxs = plt.subplots(ngroups, 2, sharey='row')
        for ig, group in enumerate(MIdata['Group'].unique()):
            for icond, cond in enumerate(('congruent', 'incongruent')):
                rcis = MIdata[f'RCI ({mod}{pref} {cond})'].loc[MIdata['Group'] == group].copy()
                print(f'{group}: Area {area} using {rcis.shape[0]} neurons for analysis of RCI {mod}{pref} {cond}')
                rciDF = RCI_dist_plot(rcis, rciaxs[ig, icond], xlab=True if ig == ngroups-1 else False)
                name = f'{mod}_{group}_{cond}'
                rciDF.rename(columns={'Response change index': f'RCI'}, inplace=True)
                rciDF['modality'] = [mod] * len(rciDF)
                rciDF['group'] = [group] * len(rciDF)
                rciDF['congruency'] = [cond] * len(rciDF)
                rciDF.drop(columns='_', inplace=True)
                rci_DFlist.append(rciDF)

                rciaxs[ig, icond].set_title(f'{group}-{mod}{pref}-{cond}')
        # save
        rcifig.tight_layout()

        # save into directory
        if savedir is not None:
            rcifig.savefig(os.path.join(savedir, f'RCIs_{area}_{mod}{pref}.png'), dpi = 300)
        else:
            rcifig.savefig(f'RCIs_{area}_{mod}{pref}.png', dpi = 300)
        plt.close('all')
    
    # Quantification
    RCIQuantDF = pd.concat(rci_DFlist)
    RCI_proportions(RCIQuantDF, area = area, savedir=savedir)

def RCI_proportions(RCIs:pd.DataFrame, area:str, savedir:str):
    colors = {'+':'green', '-':'red'}
    # separate plots & stats for each modality
    for mod, modDF in RCIs.groupby('modality'):
        within_statTABs = {}
        between_statTABs = {}
        plotTABs = {}
        # separate contingency tables for each group
        for g, gDF in modDF.groupby('group'):
            within_statTABs[g] = pd.crosstab(index= gDF['congruency'], columns= gDF['sign'])
            plotTABs[g] = pd.crosstab(index= gDF['congruency'], columns= gDF['sign'], normalize='index')
        
        for c, cDF in modDF.groupby('congruency'):
            between_statTABs[c] = pd.crosstab(index= cDF['group'], columns= cDF['sign'])
        
        # Do Stats
        cong_comparison = spstats.fisher_exact(between_statTABs['congruent']).pvalue
        incong_comparison = spstats.fisher_exact(between_statTABs['incongruent']).pvalue
        
        ### Do the plots
        # ── tidy table → long format ------------------------------------
        long = (pd.concat(plotTABs, names=['group', 'congruency'])   # (group, congruency)
                .stack()                                           # column 'sign' → rows
                .rename('value')                                   # <- single value column
                .reset_index())                                    # cols = [group, congruency, sign, value]

        groups       = long['group'].unique()         # ['g1pre', 'g2pre']
        congruencies = long['congruency'].unique()    # ['congruent', 'incongruent']
        signs        = long['sign'].unique()          # ['+', '–', …]

        # ── x-positions --------------------------------------------------
        bar_w    = 0.35                               # width of each stacked bar
        inner_gap = 0.05                              # tiny space between g1 & g2
        outer_gap = 0.60                              # big space between blocks

        # centres of the *congruency* blocks
        block_cx = np.arange(len(congruencies)) * (2*bar_w + outer_gap)
        # exact x for every group within each block
        x_pos = {g: block_cx + (-bar_w/2 - inner_gap/2 if i == 0 else
                                +bar_w/2 + inner_gap/2)
                for i, g in enumerate(groups)}

        # ── draw ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))

        for g in groups:
            bottoms = np.zeros(len(congruencies))
            for s in signs:
                vals = (long
                        .query('group == @g and sign == @s')
                        .sort_values('congruency')['value']
                        .to_numpy())
                ax.bar(x_pos[g], vals, bar_w,
                    bottom=bottoms,
                    label=s if g == groups[0] else "",   # only label once in legend
                    edgecolor='black',
                    color=colors[s])                  # your colour palette
                bottoms += vals

        # ── dual-row x-labels -------------------------------------------
        # bottom row = groups (minor ticks)
        ax.set_xticks(np.concatenate(list(x_pos.values())), minor=True)
        ax.set_xticklabels(np.repeat(groups, len(congruencies)),
                        minor=True, rotation=0, ha='center')

        # upper row = congruency, centred across the pair of bars
        ax.set_xticks(block_cx)
        ax.set_xticklabels(congruencies,
                        rotation=0, ha='center', fontweight='bold')
        
        ax.tick_params(axis='x', which='major', pad=16)  # congruency (bottom row)

        ax.set_ylabel('Proportion enhanced / suppressed')
        ax.tick_params(axis='x', which='both', length=0)
        ax.legend(title='Sign', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # ❷  Convert your two p-values to star labels
        cong_label   = get_sig_label(cong_comparison*2)
        incong_label = get_sig_label(incong_comparison*2)

        # ❸  Work out where to place the bars
        bars_x     = np.sort(np.array(list(x_pos.values())).flatten())
        heights    = [1 for _ in range(4)]
        cong_x1, cong_x2 = bars_x[0], bars_x[1]
        incong_x1, incong_x2 = bars_x[2], bars_x[3]

        # y position = a tad above the tallest bar in the pair
        y_offset = 0.03
        cong_y   = max(heights[0], heights[1]) + y_offset
        incong_y = max(heights[2], heights[3]) + y_offset

        # keep room for the annotations
        ax.set_ylim(top = max(cong_y, incong_y) + 0.06)

        # ❹  Actually draw the two annotations
        add_sig(ax, cong_x1,   cong_x2,   cong_y,   cong_label,
                color='k', linewidth=1.3)
        add_sig(ax, incong_x1, incong_x2, incong_y, incong_label,
                color='k', linewidth=1.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f'RCI_proportions_{area}_{mod}'))
