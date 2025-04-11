# @matushalak
# contains calculations to quantify multisensory enhancement / suppression, as well as, direction selectivity
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import artist


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
    FLUORO_RESP = FLUORO_RESP[:,:2,:]
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


# TODO: if we want to say that they are direction TUNED (significantly) [and only look at those neurons]
def DSI_threshold(trial_labels : np.ndarray | list, all_signals) -> float:
    '''
    Average 99th percentile of trial-label shuffled DSI distribution as discussed in Meijer et al. (2017)
    '''
    pass


# --------- prepare plotting dataframe ---------
def getMIdata(FLUORO_RESP: np.ndarray, group_cond_name: str,
              MIdata_dict: dict[str: list]) -> dict[str: list]:
    '''
    Returns lists which can be iterated over and appended to from different conditions
    '''
    #1) get direction selectivity info
    pref_stats, orth_stats, congruent_stats, incongruent_stats = np.split(direction_selectivity(FLUORO_RESP), 
                                                                            indices_or_sections=4, 
                                                                            axis = 2)
    #2) get direction selectivity index for visual and auditory separately
    DSI_vis, DSI_aud = np.split(DSI(pref_stats, orth_stats),
                                indices_or_sections=2,
                                axis = 2)
    
    MIdata_dict['Group'] += [group_cond_name] * DSI_vis.size
    MIdata_dict['DSI (VIS)'] += [*DSI_vis.squeeze()]
    MIdata_dict['DSI (AUD)'] += [*DSI_aud.squeeze()]
    #3) Get response change index for visual and auditory separately
    # for congruent MST
    RCI_vis_congruent, RCI_aud_congruent = np.split(RCI(congruent_stats, pref_stats),
                                                    indices_or_sections=2,
                                                    axis = 2)
    MIdata_dict['RCI (VIS congruent)'] += [*RCI_vis_congruent.squeeze()]
    MIdata_dict['RCI (AUD congruent)'] += [*RCI_aud_congruent.squeeze()]
    # for incongruent MST
    RCI_vis_incongruent, RCI_aud_incongruent = np.split(RCI(incongruent_stats, pref_stats),
                                                        indices_or_sections=2,
                                                        axis = 2)
    MIdata_dict['RCI (VIS incongruent)'] += [*RCI_vis_incongruent.squeeze()]
    MIdata_dict['RCI (AUD incongruent)'] += [*RCI_aud_incongruent.squeeze()]

    return MIdata_dict


def prepare_long_format_Areas(out_size : int, 
                              Area_indices:dict[str: np.ndarray]) -> np.ndarray:
    regions = np.empty(shape = out_size, dtype='U7')
    for region_name, where_region in Area_indices.items():
        regions[where_region] = str(region_name)
    return regions

#------- plotting --------
def scatter_hist_reg_join(MIdata: pd.DataFrame,
                          NAME: str,
                          X_VAR: str, Y_VAR:str, HUE_VAR :str, 
                          kde:bool = False, reg:bool = False, square: bool = False,
                          savedir: str | None = None,
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
        else:
            YLIM = diagy
            XLIM = diagx

    g = sns.JointGrid(data=MIdata, x=X_VAR, y=Y_VAR, hue = HUE_VAR, palette= colmap,
                      height=8, ratio=8, space = 0, xlim=XLIM, ylim=YLIM)
    g.ax_joint.plot(diagx, diagy, linestyle = '--', color = 'dimgray')
    g.plot_joint(sns.scatterplot, data = MIdata, alpha = 0.35, style = HUE_VAR, 
                 markers = markrs, s = 12)
    if kde:
        g.plot_joint(sns.kdeplot, levels = 5)
    
    # sns.rugplot(data=MIdata, x = X_VAR, hue=HUE_VAR, ax=g.ax_marg_x, 
    #             height=-.08, clip_on=False, lw = 1, alpha = .2, palette=colmap, legend=False)
    sns.histplot(data=MIdata, x = X_VAR, hue=HUE_VAR, ax=g.ax_marg_x, kde=True, palette=colmap, legend=False)
    
    # sns.rugplot(data=MIdata, x = Y_VAR, hue=HUE_VAR, ax=g.ax_marg_y, 
    #             height=-.08, clip_on=False, lw = 1, alpha = .2, palette=colmap, legend=False)
    sns.histplot(data=MIdata, y = Y_VAR, hue=HUE_VAR, ax=g.ax_marg_y, kde=True, palette=colmap, legend=False)

    if reg:
        for group,gr in MIdata.groupby(HUE_VAR):
            sns.regplot(x=X_VAR, y=Y_VAR, data=gr, scatter=False, ax=g.ax_joint, truncate=False, color = colmap[group])
    
    sns.move_legend(obj = g.ax_joint, loc = 1 if 'DSI' in NAME else 2)
    # g.figure.tight_layout()
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f'{NAME}.png'), dpi = 300)
    else:
        plt.savefig(f'{NAME}.png', dpi = 300)
    plt.close()

def RCI_dist_plot(RCIs:np.ndarray,
                  ax: artist = plt,
                  xlab: bool = False):
    RCIsorted = np.sort(RCIs)
    pos = np.where(RCIsorted > 0)[0]
    neg = np.where(RCIsorted < 0)[0]
    sign = np.empty_like(RCIsorted, dtype=str)
    sign[pos] = '+'
    sign[neg] = '-'
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
    plt.tight_layout()


def RCI_dist_plots_all(MIdata: pd.DataFrame, 
                       area: str = 'all',
                       savedir: str | None = None):
    ngroups = MIdata['Group'].unique().size
    for mod in ('VIS', 'AUD'):
        rcifig, rciaxs = plt.subplots(ngroups, 2, sharey='row')
        for ig, group in enumerate(MIdata['Group'].unique()):
            for icond, cond in enumerate(('congruent', 'incongruent')):
                rcis = MIdata[f'RCI ({mod} {cond})'].loc[MIdata['Group'] == group]
                RCI_dist_plot(rcis, rciaxs[ig, icond], xlab=True if ig == ngroups-1 else False)
                rciaxs[ig, icond].set_title(f'{group}-{mod}-{cond}')
        # save
        rcifig.tight_layout()

        # save into directory
        if savedir is not None:
            rcifig.savefig(os.path.join(savedir, f'RCIs_{area}_{mod}.png'), dpi = 300)
        else:
            rcifig.savefig(f'RCIs_{area}_{mod}.png', dpi = 300)
        plt.close()
