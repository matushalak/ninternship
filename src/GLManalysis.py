from src.GLM import design_matrix, clean_group_signal
from src.AUDVIS import Behavior, AUDVIS
from src.VisualAreas import Areas
from src.analysis_utils import plot_avrg_trace, fluorescence_response
from src.glm_utils import drives_loader

import pickle
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import os
import sklearn.cluster as skClust

from collections import defaultdict
from typing import Literal

from src import PYDATA, PLOTSDIR

# ------------ Class that analyzes explained variance calculated above -----------
class EvAnalysis:
    def __init__(self,
                 resultsDF: str | pd.DataFrame,
                 ARs:list[Areas],
                 combine_TT_AV:bool = True,
                 savedir:str = PYDATA):
        
        EVshuffles = [f for f in os.listdir(savedir) 
                        if 'EVshuffle_distribution' in f]

        assert (len(EVshuffles) > 0 and 
                isinstance(EVshuffles, list) and 
                isinstance(EVshuffles[0], str)
                ), 'Error in provided list of shuffle paths. EVshuffles expects list[path, ...]'
        # null distributions for each neuron, predictor, trial type,...
        self.EVnull = dict()
        assert os.path.exists(explain_file := os.path.join(savedir, 
                                                           'EV3Dexplanation.pkl')
                            ), 'Explanation file for the shuffled distributions is missing, try rerunning glm_utils.py for a few neurons, '\
                            'it will be recreated immediately'
        with open(explain_file, 'rb') as expl_f:
            self.shuffle_explain = pickle.load(expl_f)

        for shuffle_path in EVshuffles:
            group_name = shuffle_path.split('_')[0]
            self.EVnull[group_name] = np.load(os.path.join(savedir, shuffle_path))

        self.signif_df = self.nullXpercentile(percentile=99)

        if isinstance(resultsDF, str):
            if savedir not in resultsDF:
                resultsDF = os.path.join(savedir, resultsDF)
            resultsDF = pd.read_csv(resultsDF)
        
        # load DF with results
        self.df = resultsDF
        
        # load in areas into a dictionary
        self.Areas =  {ar.NAME:ar for ar in ARs}
        # absolute neuron IDs
        self.n_per_group = self.add_absolute_neuron_ids()
        # transform to long format for easier plotting
        self.df = self.long_format_df()
        # add shuffled distribution percentile into main dataframe
        self.df = self.add_nullXpercentile()

        if combine_TT_AV:
            self.df.loc[(self.df['trial_type'] == 'AV+') | 
                        (self.df['trial_type'] == 'AV-'), 
                        'trial_type'] = 'AV'
            group_cols = [col for col in self.df.columns if 'EV' not in col]
            self.df = self.df.groupby(group_cols, as_index=False)[['EV', f'EV{self.PERCENTILE}']].max()
            self.df.sort_values(by=['group_id', 'session_id', 'neuron_id_overall', 'Predictors'], inplace=True)

        # add boolean significance
        self.df = self.add_signif()
        
        # add brain areas to each neuron
        self.df = self.add_brain_areas()

        self.saveDIR = os.path.join(PLOTSDIR, 'GLManalysis')
        if not os.path.exists(self.saveDIR):
            os.makedirs(self.saveDIR)
        

    def add_signif(self):
        '''
        Assign significance based on held-out performance
        '''
        # significance based on held-out performance
        df = self.df.copy()
        df['Significant'] = ((df['Dataset'] == 'held_out') & 
                            (df['EV'] > df[f'EV{self.PERCENTILE}']))
        key_cols = [
        "neuron_id_overall",
        'session_id',
        "group_id",
        "trial_type",
        "Predictors",
        "Calculation",
        ]

        # Propagate the significant flag to every in-sample row 
        df["Significant"] = df.groupby(key_cols)["Significant"].transform("any")
        
        # Overall model significant for that trial type
        model_sig_cols = [
        "neuron_id_overall",
        'session_id',
        "group_id",
        "trial_type",
        "Calculation",
        ]
        df['ModelSignificant'] = (df['Predictors'] == 'Model') & (df['Significant'] == True)
        df['ModelSignificant'] = df.groupby(model_sig_cols)['ModelSignificant'].transform('any')
        
        return df

    def add_nullXpercentile(self):
        """
        Merge `sig_df` (held-out significance results) into `base_df`
        and create/column `EV{self.PERCENTILE}`.

        Returns
        -------
        pd.DataFrame
            Same rows as `base_df`, but with an `EV{self.PERCENTILE}` column filled-in.
        """
        key_cols = [
            "neuron_id_overall",
            "group_id",
            "trial_type",
            "Predictors",
            "Calculation",
        ]
        sig_df = self.signif_df.copy()
        sig_series = sig_df.set_index(key_cols)[f"EV{self.PERCENTILE}"]

        base_df = self.df.copy()
        base_df["EV99"] = (
            base_df.set_index(key_cols)                 # build the same index
                    .index.map(sig_series)              # lookup
                    .to_numpy()                         # back to ndarray
        )
        return base_df


    def nullXpercentile(self, percentile:float = 99
                         )->pd.DataFrame:
        self.PERCENTILE = percentile
        self.raw_percentiles = {g: np.percentile(gEVarr, q = percentile, 
                                             axis = 2) # gEVarr is a 4D array
                            for g, gEVarr in self.EVnull.items()}
        
        n_per_group = [evarr.shape[-1] for evarr in self.raw_percentiles.values()]
        # first build up a wide dataframe
        sigDF = {'neuron_id_overall':np.concatenate(
            [np.repeat(np.arange(npg), repeats=len(self.shuffle_explain['rows']))
             for npg in n_per_group]),
             'group_id':np.concatenate(
            [[gname] * len(self.shuffle_explain['rows']) * npg
             for npg, gname in zip(n_per_group, self.raw_percentiles.keys())]
             )}
        
        sigDF['trial_type'] = self.shuffle_explain['rows'] * (sigDF['group_id'].size // 
                                                              len(self.shuffle_explain['rows']))

        dfPercentiles = [np.vstack(np.unstack(EVperc, axis = 2)) # here EVperc is a 3D array
                         for g, EVperc in self.raw_percentiles.items()]
        dfPercentiles = np.concatenate(dfPercentiles, axis=0)
        
        for icol, colname in enumerate(self.shuffle_explain['columns']):
            sigDF[colname] = dfPercentiles[:,icol]

        wide_significanceDF = pd.DataFrame(sigDF)
        long_significanceDF = self.long_format_df(df = wide_significanceDF,
                                                  valname=f'EV{self.PERCENTILE}',
                                                  include_in_sample=False)
        
        return long_significanceDF


    def add_absolute_neuron_ids(self):
        n_per_group = self.df.groupby("group_id").size().values // self.df["trial_type"].nunique()
        neuron_ids = np.concatenate([np.repeat(np.arange(npg), repeats=self.df["trial_type"].nunique())
                                    for npg in n_per_group])
        self.df["neuron_id_overall"] = neuron_ids
        return n_per_group
    

    def long_format_df(self, 
                       df:pd.DataFrame | None = None,
                       valname:str = 'EV',
                       include_in_sample:bool = True):
        if df is None:
            df = self.df 
        else:
            assert isinstance(df, pd.DataFrame), f'Supplied df is not a dataframe but {type(df)}!'
        # 1) melt into long form
        df_long = df.melt(
            id_vars=[c for c in df.columns if 'EV' not in c],  # whatever other columns you have
            value_vars=[c for c in df.columns if 'EV' in c],
            var_name='metric',
            value_name=valname
        )

        # 1) split that 'metric' into the three pieces
        #    the pattern is EV_<Predictors>_<Calculation>_<Dataset>
        if include_in_sample:
            split_cols = ['drop','Predictors','Calculation','Dataset']
        else:
            split_cols = ['drop','Predictors','Calculation']
        
        df_long[split_cols] = (
            df_long['metric']
            .str.split('_', expand=True)
        )

        # 2) drop the no-longer needed columns
        df_long = df_long.drop(columns=['metric','drop'])

        # now df_long has:
        #   — all your original id_vars
        #   — Predictors  in {'AV','V','A','Motor','Model'}
        #   — Calculation in {'a','t'}     (trial-averaged vs trial-trial)
        #   — Dataset     in {'full','eval'}
        #   — EV          the explained‐variance value

        # 3) More descriptive names:
        df_long['Calculation']    = df_long['Calculation'].map({'a':'averaged','t':'trial'})
        if include_in_sample:
            df_long['Dataset'] = df_long['Dataset'].map({'full':'in_sample', 'eval':'held_out'})
        
        # 4) Sort in the right order
        if 'session_id' in df_long.columns:
            df_long.sort_values(by=['group_id', 'session_id', 'neuron_id_overall'], inplace=True)
            df_long['EV'] = df_long['EV'].clip(lower=0)
        else:
            df_long.sort_values(by=['group_id', 'neuron_id_overall'], inplace=True)

        return df_long
    

    def wide_format_df(self, 
                       tt:Literal['A', 'V', 'AV', 'all'] = 'AV',
                       calc:Literal['averaged', 'trial'] = 'averaged', 
                       dataset:Literal['in_sample', 'held_out'] = 'held_out',
                       sig:bool = True
                       )->pd.DataFrame:
        selection = ((self.df['Dataset']==dataset) & 
                     (self.df['Calculation']==calc) & 
                     (self.df['Predictors'].isin(['V', 'A', 'Motor'])) 
                     )
        if tt != 'all':
            selection = ((selection) & (self.df['trial_type'] == tt))
        
        if sig:
            selection = selection & self.df['ModelSignificant'] == True

        wide_slice = self.df.loc[selection]
        wide = wide_slice.pivot_table(index=['group_id', 'Region', 'neuron_id_overall'],
                                      columns='Predictors',
                                      values=['EV', 'Significant'])
        return wide
        

    def add_brain_areas(self):
        self.df['Region'] = np.full(shape = self.df.shape[0], fill_value='', dtype='U7')
        for g, gAreas in self.Areas.items():
            for region_name, where_region in gAreas.area_indices.items():
                mask = (
                (self.df['group_id'] == g) &
                (np.isin(self.df['neuron_id_overall'], where_region))
                )
                # 3) Assign *into* the 'Region' column (note the *string* 'Region', not self.df['Region'])
                self.df.loc[mask, 'Region'] = region_name
        
        return self.df.loc[self.df['Region']!='']


    def preditor_comparison(self,
                            tt:Literal['A', 'V', 'AV'] = 'AV',
                            calc:Literal['averaged', 'trial'] = 'averaged', 
                            dataset:Literal['in_sample', 'held_out'] = 'held_out'):
        data = self.df.loc[((self.df['Calculation'] == calc) & 
                            (self.df['Dataset'] == dataset) &
                            (self.df['trial_type'] == tt) & 
                            (self.df['Predictors'] != 'Model') &
                            (self.df['EV'] > 0) &
                            (self.df['ModelSignificant'] == True) # where overall model significant
                            )].copy()
        
        bp = sns.catplot(data=data, x = 'Region', y = 'EV', 
                            hue = 'group_id', 
                            estimator='median',
                            kind='box', 
                            col = 'Predictors', col_order=['V', 'A', 'Motor'],  
                            # edgecolor = 'k',
                            # palette=color_scheme,
                            # alpha = 0.7,
                            showfliers = False
                            )
            
        bp.legend.set_visible(False)
        bp.figure.legend()
        bp.figure.suptitle(f'EV comparison [EV on {dataset} {calc} data for {tt} trials]')
        plt.tight_layout()
        plt.savefig(f'EV_region_group_comparison_{calc}_{dataset}_TT({tt}).svg', 
                    # dpi = 300
                    )
        plt.show()
    

    def order_neurons(self,
                      tt:Literal['A', 'V', 'AV'] = 'AV',
                      calc:Literal['averaged', 'trial'] = 'averaged', 
                      dataset:Literal['in_sample', 'held_out'] = 'held_out'):
        wide = self.wide_format_df(calc=calc, dataset=dataset, 
                                   tt = tt, sig=True)
        group_ids = wide.index.get_level_values('group_id').unique()
        regions = wide.index.get_level_values('Region').unique()
        # mapping predictor → row
        row = {'V':0, 'A':1, 'Motor':2}
        # colours for exclusive neurons
        col_map = {'V':'dodgerblue', 'A':'red', 'Motor':'green'}

        # Label which neuron is which category (based on shuffle)
        wide['Label'] = wide.apply(lambda r: r['Significant'].idxmax() if 
                                   (r['Significant'].max() != 0 and r['Significant'].sum() != 2)
                                   else r['EV'].idxmax(), 
                                   axis = 1)
        # Proportion plot
        wide['STATLabel'] = wide.apply(lambda r: r['EV'].idxmax(), 
                                       axis = 1)
        # 1) count
        # wideprop = wide.loc[wide['STATLabel'] != 'Nonsignificant']
        counts = ( wide
                .groupby(["Region","group_id","STATLabel"])
                .size()
                .reset_index(name="n") )

        # 2) fraction within each Region × group_id
        counts["prop"] = ( counts
                        .groupby(["Region","group_id"])["n"]
                        .transform(lambda x: x / x.sum()) )

        # 3) plot
        out = (
            so.Plot(counts, x="group_id", y="prop", color="STATLabel", 
                    )
            .facet("Region")
            .add(so.Bars(), so.Stack())
            .scale(color = {'V':'dodgerblue', 'A':'red', 'Motor':'green'})
            )
        plot = out.plot(pyplot=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveDIR, 'proportion_neurons.svg'))
        plt.show()
        plt.close()
        
        for region in regions:
            for group in group_ids:
                mask = (wide.index.get_level_values('Region') == region) & \
                   (wide.index.get_level_values('group_id') == group)
                sub_wide = wide.loc[mask]
                # Order for plot
                v_minus_m = sub_wide['EV']['V'] - sub_wide['EV']['Motor']
                order = sub_wide.assign(v_m = v_minus_m).sort_values('v_m', ascending=False)

                fig, axes = plt.subplots(3, 1, figsize=(14, 4), sharex=True)
                for n, row_data in order.iterrows():
                    x = np.where(order.index==n)[0][0]                # bar position
                    for pred, ev_value in row_data['EV'][['V','A','Motor']].items():
                        ax = axes[row[pred]]
                        color = (col_map[pred] if (row_data['Label']==pred).bool() else 'gray')
                        ax.bar(x, ev_value, color=color, width=1.0)

                for ax in axes:
                    ax.set_ylabel('EV')
                    ax.spines[['right','top']].set_visible(False)
                    ax.set_xticks([])

                fig.suptitle(f'Region: {region}, Group: {group}')
                plt.tight_layout()
                plt.show()

    def cluster_neurons(self):
        skClust.KMeans()

    
    def well_modelled(self, 
                      calc:Literal['averaged', 'trial'] = 'averaged'):
        sub_df = self.df.loc[(self.df['Calculation'] == calc) & (self.df['Dataset'] == 'held_out')]
        anygood = sub_df.groupby(['group_id', 'neuron_id_overall']
                                  )['ModelSignificant'].any()
        anygood = pd.DataFrame(anygood.reset_index())
        anygood = anygood.loc[anygood['ModelSignificant'] == True, 
                              ['group_id', 'neuron_id_overall']]
        good_dict = {g:s.to_numpy() for g, s in anygood.groupby('group_id')['neuron_id_overall']}
        return good_dict
    


def average_clean_plot(AVs:list[AUDVIS], 
                       well_modelled_neurons:dict[str:np.ndarray],
                       pre_post: Literal['pre', 'post', 'both'] = 'pre', 
                       savedir:str = PLOTSDIR,
                       show:bool = True):
    component_colors = {'Vdrive':'dodgerblue', 'Adrive':'red', 'AVdrive':'goldenrod', 
                        'Motordrive':'green',
                        'Modeldrive':'magenta', 
                        'Cleaned dF/F':'grey',
                        'Raw dF/F':'black'
                        }
    # prepare time for plotting
    pre = round(-1/AVs[0].SF * AVs[0].trial_frames[0])
    post = (1/AVs[0].SF * AVs[0].trial_frames[1])
    time = np.arange(pre, post, 1/AVs[0].SF)
    trial_groups:list[tuple[int]] = [(6,7), (0,3), (1,5,2,4)]
    trial_group_labels:list[str] = ['V', 'A', 'AV']

    # quant comparisons
    quantFRdrives = {'Group':[],
                      'Region':[],
                      'Predictor':[],
                      'FR':[]
                      }

    for AV in AVs:
        # collect drives for plotting
        drives: dict[str:np.ndarray] = drives_loader(group_name=AV.NAME)
        drives['Cleaned dF/F'] = clean_group_signal(group_name=AV.NAME, pre_post=pre_post)
        drives['Raw dF/F'] = AV.baseline_correct_signal(AV.zsig, AV.TRIAL[0])
        # well modelled neurons (significantly at least for one trial type)
        selection = well_modelled_neurons[AV.NAME]

        # get brain area indices
        AR = Areas(AV, get_indices=True)

        # set-up figure
        f, ax  = plt.subplots(nrows=4, ncols=3, sharey='all', sharex='col',
                              figsize = (6, 8))

        for iarea, (area_name, area_indices) in enumerate(AR.area_indices.items()):
            for iname, (name, sig) in enumerate(drives.items()):
                # split into trials
                sig_dict = AV.separate_signal_by_trial_types(signal=sig)
                # get well-modeled neurons for each trial-type group
                sig2 = defaultdict(list)
                for tt, sigtt in sig_dict.items():
                    ttname = None
                    for itg, tg in enumerate(trial_groups):
                        if tt in tg:
                            ttname = trial_group_labels[itg]
                    sig2[ttname].append(sigtt[:,:,np.intersect1d(selection, area_indices)])
                
                del sig_dict # free up memory
                
                for itg, tn in enumerate(trial_group_labels):
                    if tn in ('V', 'A') and 'AV' in name:
                        continue
                    grouped_trial_all_neurons = np.vstack(sig2[tn])
                    if name in ('Adrive', 'Motordrive') and tn == 'A':
                        driveFR = fluorescence_response(signal=grouped_trial_all_neurons, window=AV.TRIAL, retMEAN_only=True)
                        outsize = driveFR.size
                        # save to dict
                        quantFRdrives['Group'] += [AV.NAME]*outsize
                        quantFRdrives['Region'] += [area_name]*outsize
                        quantFRdrives['Predictor'] += [name]*outsize
                        quantFRdrives['FR'] += driveFR.tolist()

                    sig2[tn] = np.mean(grouped_trial_all_neurons, axis = 0)
                    plot_avrg_trace(time=time, avrg=np.mean(sig2[tn], axis=1), #SEM = stats.sem(sig2[tn], axis = 1),
                                    Axis=ax[iarea, itg], label=name, title=f'{tn}trials_{area_name}', 
                                    col=component_colors[name],
                                    tt = itg)
                    if iarea == len(AR.area_indices)-1:
                        ax[iarea, itg].set_xlabel('Time (s)')
                    if itg == 0:
                        ax[iarea, itg].set_ylabel('z(dF/F0)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f'glmCleanAverage_{AV.NAME}.png'), dpi = 300)
        if show:
            plt.show()
    
    # Quantification Dataframe
    quantDF = pd.DataFrame(quantFRdrives)
    # export for easy stat analysis in JASP / R
    quantDF.to_csv(os.path.join(PYDATA, 'GLMdrives_quantification.csv'))


def driveQuantPlot(savedir:str):
    assert os.path.exists(path := os.path.join(PYDATA, 'GLMdrives_quantification.csv'))

    df = pd.read_csv(path, index_col=False)
    plot = sns.catplot(data = df, x = 'Predictor', y = 'FR', 
                       hue = 'Group', row = 'Region', 
                       kind = 'point', dodge = True)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'DriveQuants.svg'))


if __name__ == '__main__':
    gXY, AVs = design_matrix(pre_post='pre', group='both', returnAVs=True, 
                            #  show=True
                             )
    
    # Initialize class
    EVa = EvAnalysis(resultsDF=os.path.join(PYDATA, 
                                            'GLM_ev_results_neuron.csv'), 
                     ARs=[Areas(av, get_indices=True) for av in AVs])
    
    # Analysis 1) Compare V, A, motor predictors in neurons significantly explained by model
    # for AV trials (that's where all 3 components can come through)
    # TODO: separately for running / whisker / pupil
    # main
    # EVa.preditor_comparison(tt='AV', calc='averaged', dataset='held_out')
    # supplementary
    # EVa.preditor_comparison(tt='V', calc='averaged', dataset='held_out')
    # EVa.preditor_comparison(tt='A', calc='averaged', dataset='held_out')
    # EVa.preditor_comparison(tt = 'AV', calc='trial', dataset='held_out')

    # Analysis 2) Distribution of neurons based on explained variance
    # EVa.order_neurons(tt = 'AV', calc='averaged', dataset='held_out')
    # EVa.order_neurons(tt = 'AV', calc='trial', dataset='held_out')
    
    # Analysis 3-4) Average drives plot over well-modelled neurons + examples of well-modelled neurons
    # average_clean_plot(AVs=AVs, well_modelled_neurons=EVa.well_modelled(calc='averaged'), savedir=EVa.saveDIR)

    # Plot
    driveQuantPlot(savedir=EVa.saveDIR)
    

    
    