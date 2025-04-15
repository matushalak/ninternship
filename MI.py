# @matushalak
import numpy as np
import pandas as pd
import os
import multisens_calcs as MScalc
from Analyze import Analyze
from AUDVIS import AUDVIS, Behavior, load_in_data
from VisualAreas import Areas
from analysis_utils import calc_avrg_trace, build_snake_grid, snake_plot, plot_avrg_trace
from typing import Literal
#TODO: check how many neurons are being used if they are really being correctly indexed
# ---------------------- MULTISENSORY ENHANCEMENT ANALYSIS -----------------------------
def MI(pre_post: Literal['pre', 'post', 'both'] = 'pre',
       byAreas: bool = False,
       GROUP_type:Literal['modulated','modality_specific', 'all', None] = None):
    # Load in all data and perform necessary initial calculations
    AVS : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]

    MIdata_dict = {'DSI (VIS)' : [], 'DSI (AUD)': [], # collect DSIs
                   'Group':[],# collect group_names,
                   'BrainRegion':[],
                   'RCI (VIS congruent)': [], 'RCI (VIS incongruent)' : [], # collect RCIs
                   'RCI (AUD congruent)': [], 'RCI (AUD incongruent)' : []}
    
    if GROUP_type is not None:
        if GROUP_type == 'modulated':
            GROUPS = ('VIS', 'AUD')
        else:
            GROUPS = ('VIS', 'AUD', 'MST')
        # this creates index subsets across groups investigated
        subsets = {f'{g}_{GROUP_type}' if GROUP_type != 'all' else f'{g}' : [] 
                   for g in GROUPS}

    last_n_neur = 0
    for i, (Av, Analys) in enumerate(zip(AVS, ANS)):
        # selecting neuron groups / subsets
        if GROUP_type is not None:
            for gt in subsets.keys():
                sub_set = Analys.NEURON_groups[gt] + last_n_neur # from the other analysis
                subsets[gt] += [*sub_set]

        # update MIdata_dict
        MIdata_dict = MScalc.getMIdata(Analys.FLUORO_RESP, Av.NAME, MIdata_dict)
        n_neurons = Analys.FLUORO_RESP.shape[0]
        last_n_neur += n_neurons
        if byAreas:
            # get indices for each
            Ar_indices = Areas.separate_areas(Av)
            # get long_format indices
            MIdata_dict['BrainRegion'] += [*MScalc.prepare_long_format_Areas(n_neurons, 
                                                                             Ar_indices)]

        else:
            # fill with nans if brain regions not being analyzed
            MIdata_dict['BrainRegion'] += [np.nan] * n_neurons

    # turn into long-format dataframe for seaborn plotting!
    MIdataFULL : pd.DataFrame = pd.DataFrame(MIdata_dict)
    print('MIdata dataframe completed!')
    # MScalc.distanceQuantification(MIdataFULL, 'DSI (VIS)', 'DSI (AUD)', 'Group')
    # make plots!
    if not os.path.exists(saveDir := 'MIplots'):
        os.makedirs(saveDir)

    if GROUP_type is None:
        subsets = {"allneurons":np.arange(MIdata.shape[0])}
    
    # after making the dataframe for ALL neurons, 
    # do analysis for Areas & Neuron groups of interest
    for sub, sub_indices in subsets.items():
        MIdata = MIdataFULL.iloc[sub_indices,:].copy()
        if GROUP_type is None:
            assert MIdata.shape == MIdataFULL.shape, 'If not splitting by groups, should use the FULL dataframe'
        else:
            saveDir_subset = os.path.join(saveDir, sub)
            if not os.path.exists(saveDir_subset):
                os.makedirs(saveDir_subset)
        
        if byAreas: # do analysis for (Responsive) Neuron groups of interest separately for each area
            # exclude neurons outside areas of interest
            MIdata_bA = MIdata[~(MIdata['BrainRegion'] == '')]
            for region in MIdata_bA['BrainRegion'].unique():
                print('Starting MI analysis for {} region'.format(region))
                MIdata_region = MIdata_bA.loc[MIdata_bA['BrainRegion'] == region].copy()
                plot_MI_data(MIdata_region, name = region.replace('/', '|'), kde=True, savedir=saveDir_subset)

        else: # just do analysis for (Responsive) Neuron groups of interest across all areas
            plot_MI_data(MIdata, kde=True, savedir=saveDir_subset)
    
    return MIdataFULL
    

def plot_MI_data(MIdata:pd.DataFrame, savedir: str, name:str = 'all', kde: bool = False):
    # Main plot!
    MScalc.RCI_dist_plots_all(MIdata, area = name, savedir = savedir)
    print(f'RCI {name} distribution plots done')

    # Direction Selectivity Index Plot
    MScalc.scatter_hist_reg_join(MIdata, NAME=f'DSI_{name}_plot', X_VAR='DSI (VIS)', Y_VAR='DSI (AUD)', HUE_VAR='Group',
                                kde = kde, reg = False, savedir=savedir)
    print(f'DSI {name} plot done!')

    # Response change index plots
    # VIS
    MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_vis', 
                                X_VAR='RCI (VIS congruent)', Y_VAR='RCI (VIS incongruent)', HUE_VAR='Group',
                                square=True, reg= False, kde=kde, savedir=savedir)
    print(f'VIS RCI {name} plot done!')
    # AUD
    MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_aud', 
                                X_VAR='RCI (AUD congruent)', Y_VAR='RCI (AUD incongruent)', HUE_VAR='Group',
                                square=True, reg= False, kde=kde, savedir=savedir)
    print(f'AUD RCI {name} plot done!')


### ---------- Main block that runs the file as a script
if __name__ == '__main__':
    MIdata = MI(byAreas=True, GROUP_type='all')
