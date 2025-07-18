# @matushalak
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations import Annotator
import src.multisens_calcs as MScalc
from src.Analyze import Analyze
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.VisualAreas import Areas
from src.analysis_utils import calc_avrg_trace, build_snake_grid, snake_plot, plot_avrg_trace
from typing import Literal
from src import MIPLOTSDIR, PYDATA, PLOTSDIR
NEWMIPLOTS = os.path.join(PLOTSDIR, 'MInewplots')
# TODO: traces of enhanced neurons vs inhibited neurons / area, with RCI > threshold
# ---------------------- MULTISENSORY ENHANCEMENT ANALYSIS -----------------------------
def MI(pre_post: Literal['pre', 'post', 'both'] = 'pre',
       byAreas: bool = False,
       GROUP_type:Literal['modulated','modality_specific', 'all', None] = None,
       all_RESP:bool = False,
       LOADdf:bool = False):
    DFPATH = os.path.join(PYDATA, 'MIdataframe.csv')

    if LOADdf:
        if os.path.exists(DFPATH):
            return pd.read_csv(DFPATH, index_col=0)

    # Load in all data and perform necessary initial calculations
    AVS : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANS : list[Analyze] = [Analyze(av) for av in AVS]

    MIdata_dict:dict = MScalc.initializeMIdata()
    
    if GROUP_type is not None:
        if GROUP_type == 'modulated':
            GROUPS = ('VIS', 'AUD')
        else:
            GROUPS = ('VIS', 'AUD', 'MST')
            if all_RESP:
                GROUPS = ('TOTAL') # all neurons significantly responding to "something"
        # this creates index subsets across groups investigated
        if all_RESP:
            subsets = {'TOTAL':[]}
        else:
            subsets = {f'{g}_{GROUP_type}' if GROUP_type != 'all' else f'{g}' : [] 
                    for g in GROUPS}

    last_n_neur = 0
    for i, (Av, Analys) in enumerate(zip(AVS, ANS)):
        # selecting neuron groups / subsets
        if GROUP_type is not None:
            for gt in subsets.keys():
                # need to do this to index into the big dataframe with all different groups of mice
                sub_set = Analys.NEURON_groups[gt] + last_n_neur # from the other analysis
                subsets[gt] += [*sub_set]

        # update MIdata_dict
        MIdata_dict = MScalc.getMIdata(Analys.FLUORO_RESP, Av.NAME, MIdata_dict)
        n_neurons = Analys.FLUORO_RESP.shape[0]
        MIdata_dict['NeuronID'] += np.arange(n_neurons).tolist()
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
    MIdataFULL: pd.DataFrame = pd.DataFrame(MIdata_dict)
    for sub, sub_indices in subsets.items():
        if all_RESP:
            if 'TOTAL' not in sub:
                continue
        MIdataFULL.loc[sub_indices, 'NeuronType'] = sub[0]
    
    MIdataFULL.to_csv(DFPATH)
    print('MIdata dataframe completed and saved!')
    
    # make plots!
    if not os.path.exists(saveDir := MIPLOTSDIR):
        os.makedirs(saveDir)

    if GROUP_type is None:
        subsets = {"allneurons":np.arange(MIdataFULL.shape[0])}
    
    # after making the dataframe for ALL neurons, 
    # do analysis for Areas & Neuron groups of interest
    for sub, sub_indices in subsets.items():
        if all_RESP:
            if 'TOTAL' not in sub:
                continue
        MIdata = MIdataFULL.iloc[sub_indices,:].copy()
        if GROUP_type is None:
            assert MIdata.shape == MIdataFULL.shape, 'If not splitting by groups, should use the FULL dataframe'
        else:
            assert MIdata.isna().sum().all() == 0
            saveDir_subset = os.path.join(saveDir, sub)
            if not os.path.exists(saveDir_subset):
                os.makedirs(saveDir_subset)
        
        includeDict = {'VIS':(True, False),
                       'AUD':(False, True),
                       'MST':(True, True)}
        
        includeVIS, includeAUD = includeDict[sub]

        if byAreas: # do analysis for (Responsive) Neuron groups of interest separately for each area
            # exclude neurons outside areas of interest
            MIdata_bA = MIdata[~(MIdata['BrainRegion'] == '')]
            for region in MIdata_bA['BrainRegion'].unique():
                print('Starting MI analysis for {} region'.format(region))
                MIdata_region = MIdata_bA.loc[MIdata_bA['BrainRegion'] == region].copy()
                plot_MI_data(MIdata_region, name = region.replace('/', '|'), kde=False, savedir=saveDir_subset, 
                             includeVIS=includeVIS, includeAUD=includeAUD)

        else: # just do analysis for (Responsive) Neuron groups of interest across all areas
            plot_MI_data(MIdata, kde=False, savedir=saveDir_subset)
    
    return MIdataFULL
    

def plot_MI_data(MIdata:pd.DataFrame, savedir: str, includeVIS:bool, includeAUD:bool, 
                 name:str = 'all', kde: bool = False):
    # WITHIN Group RCI
    if includeVIS:
        # VIS
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_visPREF_within', 
                                    X_VAR='RCI (VISpref congruent)', Y_VAR='RCI (VISpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='within')
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_visNONPREF_within', 
                                    X_VAR='RCI (VISnonpref congruent)', Y_VAR='RCI (VISnonpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='within')
        print(f'VIS RCI {name} plots done!')
    
    if includeAUD:
        # AUD
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_audPREF_within', 
                                    X_VAR='RCI (AUDpref congruent)', Y_VAR='RCI (AUDpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='within')
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_audNONPREF_within', 
                                    X_VAR='RCI (AUDnonpref congruent)', Y_VAR='RCI (AUDnonpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='within')
        print(f'AUD RCI {name} plots done!')
    
    # 1) Proportion plot
    includedMOD = []
    if includeVIS:
        includedMOD.append('VIS')
    if includeAUD:
        includedMOD.append('AUD')
    
    MScalc.RCI_dist_plots_all(MIdata, area = name, savedir = savedir, pref='pref', includedMOD=includedMOD)
    MScalc.RCI_dist_plots_all(MIdata, area = name, savedir = savedir, pref='nonpref', includedMOD=includedMOD)
    print(f'RCI {name} distribution plots done')

    # 2) Unimodal fluorescence response against MST
    # skip this
    # if includeVIS:
    #     # VIS
    #     # i] Pref VIS against MST congruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'visPref_MS+_{name}_plot', X_VAR='pref_VIS', Y_VAR='prefVIS_MST+', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # ii] Pref VIS against MST incongruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'visPref_MS-_{name}_plot', X_VAR='pref_VIS', Y_VAR='prefVIS_MST-', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # i] NonPref VIS against MST congruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'visNONPref_MS+_{name}_plot', X_VAR='nonpref_VIS', Y_VAR='nonprefVIS_MST+', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # ii] NonPref VIS against MST incongruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'visNONPref_MS-_{name}_plot', X_VAR='nonpref_VIS', Y_VAR='nonprefVIS_MST-', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    # if includeAUD:
    #     # AUD
    #     # iii] Pref AUD against MST congruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'audPref_MS+_{name}_plot', X_VAR='pref_AUD', Y_VAR='prefAUD_MST+', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # iv] Pref AUD against MST incongruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'audPref_MS-_{name}_plot', X_VAR='pref_AUD', Y_VAR='prefAUD_MST-', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # iii] NonPref AUD against MST congruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'audNONPref_MS+_{name}_plot', X_VAR='nonpref_AUD', Y_VAR='nonprefAUD_MST+', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')
    #     # iv] NonPref AUD against MST incongruent
    #     MScalc.scatter_hist_reg_join(MIdata, NAME=f'audNONPref_MS-_{name}_plot', X_VAR='nonpref_AUD', Y_VAR='nonprefAUD_MST-', HUE_VAR='Group',
    #                                 kde = False, savedir=savedir, statsmethod='within')

    # 3) Direction Selectivity Index Plot
    MScalc.scatter_hist_reg_join(MIdata, NAME=f'DSI_{name}_plot', X_VAR='DSI (VIS)', Y_VAR='DSI (AUD)', HUE_VAR='Group',
                                kde = kde, reg = False, savedir=savedir, statsmethod='between')
    print(f'DSI {name} plot done!')

    # 4) Response change index plots (between group comparisons)
    if includeVIS:
        # VIS
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_visPREF_betweem', 
                                    X_VAR='RCI (VISpref congruent)', Y_VAR='RCI (VISpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='between')
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_visNONPREF_between', 
                                    X_VAR='RCI (VISnonpref congruent)', Y_VAR='RCI (VISnonpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='between')
        print(f'VIS RCI {name} plots done!')
    
    if includeAUD:
        # AUD
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_audPREF_between', 
                                    X_VAR='RCI (AUDpref congruent)', Y_VAR='RCI (AUDpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='between')
        MScalc.scatter_hist_reg_join(MIdata, NAME=f'RCI_{name}_audNONPREF_between', 
                                    X_VAR='RCI (AUDnonpref congruent)', Y_VAR='RCI (AUDnonpref incongruent)', HUE_VAR='Group',
                                    square=True, reg= False, kde=kde, savedir=savedir, statsmethod='between')
        
        print(f'AUD RCI {name} plot done!')


# New cleaner analysis
def MIanalysis(MIDF:pd.DataFrame):
    groupmapper = {'g1':'DR', 'g2':'NR'}
    palette = {('V', 'DRpre'):'dodgerblue',
                ('A', 'DRpre'):'red',
                ('M', 'DRpre'):'goldenrod',
                ('V', 'NRpre'):'lightskyblue',
                ('A', 'NRpre'):'lightsalmon',
                ('M', 'NRpre'):'palegoldenrod'}
    hue_order = (('V', 'DRpre'), ('A', 'DRpre'), ('M', 'DRpre'), 
                ('V', 'NRpre'),('A', 'NRpre'),('M', 'NRpre'))
    if not os.path.exists(NEWMIPLOTS):
        os.makedirs(NEWMIPLOTS)

    # preprocess into long DF
    longMIDF:pd.DataFrame = MScalc.processFULLDFintoLONG(MIDF)
    longMIDF.loc[:, 'Group'] = longMIDF.loc[:, 'Group'].transform(lambda x: groupmapper[x[:2]]+x[2:])

    # direction selectivity for different neuron types
    DSIall(longMIDF=longMIDF, hue_order=hue_order, palette=palette)

    # to look at RCI: filter on RCIdf (where RCI is not NaN)
    RCIall(longMIDF=longMIDF, hue_order=hue_order, palette=palette)

    # to look at FR: filter on FRdf (where FR is not NaN)

def DSIall(longMIDF:pd.DataFrame, hue_order:list, palette:dict,
           show:bool = False):
    # to look at DSI: filter on DSIdf (one where DSI is not None / NaN)
    DSIdf = longMIDF.loc[~longMIDF['DSI'].isna()].iloc[:,[0,1,2,3,4,7]].copy()
    hue = DSIdf[['NeuronType', 'Group']].apply(tuple, axis = 1)
    dsi = sns.catplot(data = DSIdf, y = 'DSI', 
                      x = 'BrainRegion', order=['V1', 'AM/PM', 'A/RL/AL', 'LM'],
                      hue = hue, hue_order=hue_order, palette=palette, 
                      row = 'Modality', row_order=['VIS', 'AUD'],
                      col = 'NeuronType', col_order=['V', 'A', 'M'],
                      kind = 'point', dodge=True, legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(NEWMIPLOTS, 'DSIareasNeuronTypes.svg'))
    if show:
        plt.show()
    plt.close()

def RCIall(longMIDF:pd.DataFrame, hue_order:list, palette:dict):
    ''''
    Does RCI proportions as well as RCI comparisons between congruent & incongruent etc.
    '''
    # to look at RCI: filter on RCIdf (one where RCI is not NaN)
    RCIdf = longMIDF.loc[~longMIDF['RCI'].isna()].iloc[:,[0,1,2,3,4,5, 6, 8]].copy()
    pos = np.where(RCIdf['RCI'] > 0)[0]
    neg = np.where(RCIdf['RCI'] < 0)[0]
    sign = np.empty_like(RCIdf['RCI'], dtype=str)
    sign[pos] = '+'
    sign[neg] = '-'
    RCIdf['Sign'] = sign

    # TODO: RCI +- proportions (maybe treat just proportion enhanced - automatically the rest is suppressed)
    # XXX (compare also between V & A neurons; V 50/50 enhanced A more supressed)
    # XXX (also compare between preferred and non-preferred WITHIN group)
    # multisens_proportions(RCIdf=RCIdf)
    # RCI magnitude MAIN
    RCImag(RCIdf, hue_order, palette)

def RCImag(RCIdf:pd.DataFrame, hue_order:list, palette:dict):
    for TYPE in ['V', 'A', 'M']:
        RCI_within_Type(Type=TYPE, RCIdf=RCIdf, hue_order=hue_order, palette=palette)

def RCI_within_Type(Type:str, RCIdf:pd.DataFrame, hue_order:list, palette:dict,
                    show:bool = False):
    '''
    Not really different if we split by sign of enhanced vs suppressed
    '''
    typeToMod = {'V':['VIS'], 'A':['AUD'], 'M':['VIS', 'AUD']}
    for mod in typeToMod[Type]:
        TypeDF = RCIdf.loc[(RCIdf['NeuronType'] == Type) & (RCIdf['Modality'] == mod)].copy()
        TypeDF.loc[:, 'RCI'] = TypeDF.loc[:, 'RCI'].abs()
        hue = TypeDF[['NeuronType', 'Group']].apply(tuple, axis = 1)
        rcitype = sns.catplot(data = TypeDF, y = 'RCI',
                            x = 'BrainRegion', order=['V1', 'AM/PM', 'A/RL/AL', 'LM'],
                            col = 'Congruency', col_order=['congruent', 'incongruent'],
                            row = 'Preference', row_order=['pref', 'nonpref'],
                            hue = hue, hue_order=hue_order, palette=palette,
                            kind = 'point', dodge=True, legend=False)
        plt.tight_layout()
        if Type == 'M':
            plt.savefig(os.path.join(NEWMIPLOTS, f'RCIareas{Type}_{mod}.svg'))
        else:
            plt.savefig(os.path.join(NEWMIPLOTS, f'RCIareas{Type}.svg'))
        if show:
            plt.show()
        plt.close()

def multisens_proportions(RCIdf:pd.DataFrame):
    # get proportion DF
    RCIdf.groupby(['Group', 'BrainRegion', 'NeuronType', 'Modality', 'Preference', 'Congruency'])
    breakpoint()


### ---------- Main block that runs the file as a script
if __name__ == '__main__':
    MIdata = MI(byAreas=True, GROUP_type='all', 
                LOADdf=True
                )
    MIanalysis(MIdata)

