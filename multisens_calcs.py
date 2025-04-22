# @matushalak
# contains calculations to quantify multisensory enhancement / suppression, as well as, direction selectivity
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import os
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import artist
from matplotlib.patches import Patch
from utils import get_sig_label
from typing import Literal

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
    results = np.full((FLUORO_RESP.shape[0], 2, 4, 2), np.nan)
    index_results = {'VIS':dict(),
                     'AUD':dict()}
    modnames = ['VIS', 'AUD']
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

        # get indices for preferred stimuli and corresponding congruent / incongruent trials
        preferences = sense_pref.copy()
        left_pref = (preferences == 0)
        right_pref = (preferences == 1)
        # 0 = 6 for visual and 0 = 0 for auditory
        preferences[left_pref] = mod_trials[imodality][0]
        # 1 = 7 for visual and 1 = 3 for auditory
        preferences[right_pref] = mod_trials[imodality][1]
        assert all(np.unique(preferences) == mod_trials[imodality]), 'Incorrect indices for preferred trial-type'
        # add index results
        modname = modnames[imodality]
        index_results[modname]['preferred'] = preferences
        index_results[modname]['congruent'] = congruent_tts
        index_results[modname]['incongruent'] = incongruent_tts

    # assert not np.isnan(results).any(), 'Output array should not contain NaNs!'
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

def Quantification(MIdata:pd.DataFrame, 
                   X_VAR:str, Y_VAR:str, HUE_VAR:str,
                   kind: Literal['distance', 'means'] = 'distance'):
    ngroups = MIdata[HUE_VAR].nunique()
    comparison_values = []
    groups_list = []
    for g in MIdata[HUE_VAR].unique():
        vecs = MIdata.loc[MIdata[HUE_VAR] == g][[X_VAR, Y_VAR]].to_numpy()
        if kind == 'distance':
            comparison_values.append(distances(vecs))
            groups_list.append(g)
        else:
            # values for (x, y)
            comparison_values.append((vecs[:,0], vecs[:,1]))
            groups_list.append(g)
    
    assert len(comparison_values) in (2,4, 2*4, 4*4), 'Current version compares between 2 or 4 groups (not according to region) or 8 or 16 groups (per region)'
    match comparison_values:
        # means comparison
        # TODO: generalize code
        case (g1pref, g1MST), (g2pref, g2MST):
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

        case g1ds, g2ds:
            MWU = spstats.mannwhitneyu(g1ds, g2ds, alternative='two-sided', nan_policy='omit')
            pval = MWU.pvalue
            g1d_mean, g2d_mean = np.nanmean(g1ds), np.nanmean(g2ds)
            g1d_SEM, g2d_SEM = spstats.sem(g1ds, nan_policy='omit'), spstats.sem(g2ds, nan_policy='omit')
            return pval, [g1d_mean, g2d_mean], [g1d_SEM, g2d_SEM], groups_list
        

        # TODO: brain regions, all 4 groups, etc
        case _:
            raise NotImplementedError


# --------- prepare plotting dataframe ---------
def initializeMIdata()-> dict:
    MIdata_dict = {'DSI (VIS)' : [], 'DSI (AUD)': [], # collect DSIs
                   'Group':[],# collect group_names,
                   'BrainRegion':[],
                   'RCI (VIS congruent)': [], 'RCI (VIS incongruent)' : [], # collect RCIs
                   'RCI (AUD congruent)': [], 'RCI (AUD incongruent)' : [],
                   'pref_VIS':[], 'VIS_MST+':[], 'VIS_MST-':[],
                   'pref_AUD':[], 'AUD_MST+':[], 'AUD_MST-':[],
                   'pref_VIS_idx':[], 'VIS_MST+_idx':[], 'VIS_MST-_idx':[],
                   'pref_AUD_idx':[], 'AUD_MST+_idx':[], 'AUD_MST-_idx':[],}
    return MIdata_dict

def getMIdata(FLUORO_RESP: np.ndarray, group_cond_name: str,
              MIdata_dict: dict[str: list]) -> dict[str: list]:
    '''
    Returns lists which can be iterated over and appended to from different conditions
    '''
    #1) get direction selectivity info
    FRs, index_infos = direction_selectivity(FLUORO_RESP)
    pref_stats, orth_stats, congruent_stats, incongruent_stats = np.split(FRs, 
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
    # 4) Save preferred direction and corresponding MST indices
    MIdata_dict['pref_VIS_idx'] += [*index_infos['VIS']['preferred']]
    MIdata_dict['pref_AUD_idx'] += [*index_infos['AUD']['preferred']]
    MIdata_dict['VIS_MST+_idx'] += [*index_infos['VIS']['congruent']]
    MIdata_dict['AUD_MST+_idx'] += [*index_infos['AUD']['congruent']]
    MIdata_dict['VIS_MST-_idx'] += [*index_infos['VIS']['incongruent']]
    MIdata_dict['AUD_MST-_idx'] += [*index_infos['AUD']['incongruent']]
    # 4) Save preferred direction and corresponding MST Fluorescence responses
    MIdata_dict['pref_VIS'] += [*pref_stats.squeeze()[:,0,0]]
    MIdata_dict['pref_AUD'] += [*pref_stats.squeeze()[:,0,1]]
    MIdata_dict['VIS_MST+'] += [*congruent_stats.squeeze()[:,0,0]]
    MIdata_dict['AUD_MST+'] += [*congruent_stats.squeeze()[:,0,1]]
    MIdata_dict['VIS_MST-'] += [*incongruent_stats.squeeze()[:,0,0]]
    MIdata_dict['AUD_MST-'] += [*incongruent_stats.squeeze()[:,0,1]]

    return MIdata_dict


def prepare_long_format_Areas(out_size : int, 
                              Area_indices:dict[str: np.ndarray]) -> np.ndarray:
    regions = np.full(shape = out_size, fill_value='', dtype='U7')
    for region_name, where_region in Area_indices.items():
        regions[where_region] = str(region_name)
    return regions

#------- plotting --------
# TODO: fix with annotator
def scatter_hist_reg_join(MIdata: pd.DataFrame,
                          NAME: str,
                          X_VAR: str, Y_VAR:str, HUE_VAR :str, 
                          kde:bool = False, reg:bool = False, square: bool = False,
                          savedir: str | None = None, statsmethod: Literal['distance', 'means'] = 'distance',
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
        if statsmethod == 'means':
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
        
    if statsmethod == 'distance':
        sig_text = get_sig_label(pval, sig_label_map)
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
    # else:
    #     if len(pval) == 3:
    #         diff_sig_text = get_sig_label(pval[2], sig_label_map)
    #         # x‑positions: the centres of the two existing dashed lines
    #         x_pair1 = np.mean(indices[0:2])
    #         x_pair2 = np.mean(indices[2:4])
    #         # y‑positions: put the new line a bit above the taller of the two pairwise lines
    #         # Grab the highest bar‑top (mean+sem) in each pair …
    #         y_pair1 = np.max(means[0:2] + sems[0:2])
    #         y_pair2 = np.max(means[2:4] + sems[2:4])
    #         # … then lift them by a small offset so the new connector sits clear of the bars.
    #         offset = 0.05 * (np.max(means + sems) - np.min(means - sems))  # 5 % of the data range
    #         y_pair1 += offset
    #         y_pair2 += offset
    #         # Draw the connector
    #         inset_ax.plot([x_pair1, x_pair2], [y_pair1, y_pair2],
    #                     linestyle='--', color='k', alpha=0.5)
    #         # Drop the significance label halfway along
    #         inset_ax.text(np.mean([x_pair1, x_pair2]),
    #                     np.mean([y_pair1, y_pair2]) + offset,
    #                     diff_sig_text, ha='center', va='bottom', fontsize=12)
    #         inset_ax.set_ylim((0, 1.3* np.max([y_pair1, y_pair2])))

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


