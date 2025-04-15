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
from utils import get_sig_label

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
    results = np.full((FLUORO_RESP.shape[0], 2, 4, 2), np.nan)
    
    # Get preferred trial type
    for imodality, modality_tts in enumerate([vis, aud]):
        # only the trials for xthat modality
        sense_tt = FLUORO_RESP[:,:,modality_tts] 
        # absolute value of the MEAN fluorescence response for each neuron
        abs_sense_tt = sense_tt[:,0,:].__abs__()
        # Create a boolean mask identifying rows that are NOT all NaN
        non_nan_mask = ~np.all(np.isnan(abs_sense_tt), axis=1)
        # Prepare sense_pref as an integer index array of length nneurons.
        sense_pref = np.zeros(abs_sense_tt.shape[0], dtype=int)
        # find preference within that modality (take absolute value, if stronger inhibitory response)
        sense_pref[non_nan_mask] = np.nanargmax(abs_sense_tt[non_nan_mask], axis = 1) # 0 is LEFT, 1 is RIGHT
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
    
    # assert not np.isnan(results).any(), 'Output array should not contain NaNs!'
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
    regions = np.full(shape = out_size, fill_value='', dtype='U7')
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
            diagy = YLIM
            diagx = diagy
        else:
            YLIM = diagy
            XLIM = diagx

    g = sns.JointGrid(data=MIdata, x=X_VAR, y=Y_VAR, hue = HUE_VAR, palette= colmap,
                      height=8, ratio=8, space = 0, xlim=XLIM, ylim=YLIM)
    g.ax_joint.plot(diagx, diagy, linestyle = '--', color = 'dimgray')
    g.plot_joint(sns.scatterplot, data = MIdata, alpha = 0.35, style = HUE_VAR, 
                 markers = markrs, s = 12)
    if kde:
        g.plot_joint(sns.kdeplot, levels = 3)
    
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
    pval, means, sems, group_names = distanceQuantification(MIdata, X_VAR, Y_VAR, HUE_VAR)
    
    # Inset plot
    sig_label_map = {0: 'n.s.', 1: '*', 2: '**', 3: '***'}
    sig_text = get_sig_label(pval, sig_label_map)
    inset_ax = g.figure.add_axes([0.675, 0.675, 0.2, 0.2])
    indices = np.arange(len(group_names))
    # Plot each bar with error bars; use the same colors as in colmap.
    for i, group in enumerate(group_names):
        # Use colmap to get the bar color; default to gray if not found.
        color = colmap.get(group, 'gray')
        inset_ax.bar(indices[i], means[i], yerr=sems[i],
                     color=color, width=0.6, capsize=4, alpha=0.8)
    # Connect the means with a dashed line.
    inset_ax.plot(indices, means, linestyle='--', color='k', marker='o', alpha = 0.8)
    # Place the significance label above the highest error bar in the inset.
    x_mid = np.mean(indices)
    # Calculate y_top from the highest of (mean+sem) values.
    y_top = np.mean([np.max(means), np.mean(means)])
    # Determine a vertical offset relative to the inset's y-range:
    inset_ax.text(x_mid, y_top, sig_text, ha='center', va='bottom', fontsize=12)

    # Set x-ticks with group names and adjust font size/rotation.
    inset_ax.set_xticks(indices)
    inset_ax.set_xticklabels(group_names, rotation=45, fontsize=8)
    inset_ax.set_ylabel(f'Mean {NAME[:3]}', fontsize=8)

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


def RCI_dist_plots_all(MIdata: pd.DataFrame, 
                       area: str = 'all',
                       savedir: str | None = None):
    ngroups = MIdata['Group'].nunique()
    for mod in ('VIS', 'AUD'):
        rcifig, rciaxs = plt.subplots(ngroups, 2, sharey='row')
        for ig, group in enumerate(MIdata['Group'].unique()):
            for icond, cond in enumerate(('congruent', 'incongruent')):
                rcis = MIdata[f'RCI ({mod} {cond})'].loc[MIdata['Group'] == group].copy()
                print(f'{group}: Area {area} using {rcis.shape[0]} neurons for analysis of RCI {mod} {cond}')
                RCI_dist_plot(rcis, rciaxs[ig, icond], xlab=True if ig == ngroups-1 else False)
                rciaxs[ig, icond].set_title(f'{group}-{mod}-{cond}')
        # save
        rcifig.tight_layout()

        # save into directory
        if savedir is not None:
            rcifig.savefig(os.path.join(savedir, f'RCIs_{area}_{mod}.png'), dpi = 300)
        else:
            rcifig.savefig(f'RCIs_{area}_{mod}.png', dpi = 300)
        plt.close('all')


def distances(vectors:np.ndarray, 
              distance_from:tuple[float,...] = (0,0)
              )->list[float]:
    '''
    Computes Euclidian distance(s) of vectors from a given group
    '''
    assert len(vectors.shape) == 2, 'Input to distances() must be a (n_neurons, n_dims) numpy array'
    assert len(distance_from) == vectors.shape[1], 'The dimensions of distance_from and the vector for each neuron dont match'
    distance_from = np.array(distance_from)
    vectors = vectors - distance_from
    # handles nans
    eucl_dists = np.linalg.norm(vectors, axis = 1)
    return eucl_dists


def distanceQuantification(MIdata:pd.DataFrame, 
                           X_VAR:str, Y_VAR:str, HUE_VAR:str):
    ngroups = MIdata[HUE_VAR].nunique()
    distances_list = []
    groups_list = []
    for g in MIdata[HUE_VAR].unique():
        vecs = MIdata.loc[MIdata[HUE_VAR] == g][[X_VAR, Y_VAR]].to_numpy()
        distances_list.append(distances(vecs))
        groups_list.append(g)
    
    assert len(distances_list) in (2,4, 2*4, 4*4), 'Current version compares between 2 or 4 groups (not according to region) or 8 or 16 groups (per region)'
    match distances_list:
        case g1ds, g2ds:
            MWU = spstats.mannwhitneyu(g1ds, g2ds, alternative='two-sided', nan_policy='omit')
            pval = MWU.pvalue
            g1d_mean, g2d_mean = np.nanmean(g1ds), np.nanmean(g2ds)
            g1d_SEM, g2d_SEM = spstats.sem(g1ds, nan_policy='omit'), spstats.sem(g2ds, nan_policy='omit')
            return pval, [g1d_mean, g2d_mean], [g1d_SEM, g2d_SEM], groups_list
        
        # TODO: brain regions, all 4 groups, etc
        case _:
            raise NotImplementedError
        