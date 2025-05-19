from GLM import design_matrix, quantify_encoding_models
from AUDVIS import Behavior, AUDVIS
from VisualAreas import Areas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn.cluster as skClust

from typing import Literal

# ------------ Class that analyzes explained variance calculated above -----------
# TODO: add brain area column
class EvAnalysis:
    def __init__(self,
                 resultsDF: str | pd.DataFrame,
                 ARs:list[Areas],
                 filterOUT_AV:bool = True,
                 combine_TT_AV:bool = True):
        if isinstance(resultsDF, str):
            resultsDF = pd.read_csv(resultsDF)
        
        # load DF with results
        self.df = resultsDF
        
        if filterOUT_AV:
            self.df = self.df.loc[:,[c for c in self.df.columns if 'AV' not in c]]

        # load in areas into a dictionary
        self.Areas =  {ar.NAME:ar for ar in ARs}
        # absolute neuron IDs
        self.n_per_group = self.add_absolute_neuron_ids()
        # transform to long format for easier plotting
        self.df = self.long_format_df()

        if combine_TT_AV:
            self.df.loc[(self.df['trial_type'] == 'AV+') | 
                        (self.df['trial_type'] == 'AV-'), 
                        'trial_type'] = 'AV'

        # add brain areas to each neuron
        self.df = self.add_brain_areas()

    
    def add_absolute_neuron_ids(self):
        n_per_group = self.df.groupby("group_id").size().values // self.df["trial_type"].nunique()
        neuron_ids = np.concatenate([np.repeat(np.arange(npg), repeats=self.df["trial_type"].nunique())
                                    for npg in n_per_group])
        self.df["neuron_id_overall"] = neuron_ids
        return n_per_group
    

    def long_format_df(self):
        # 1) melt into long form
        df_long = self.df.melt(
            id_vars=[c for c in self.df.columns if 'EV' not in c],  # whatever other columns you have
            value_vars=[c for c in self.df.columns if 'EV' in c],
            var_name='metric',
            value_name='EV'
        )

        # 1) split that 'metric' into the three pieces
        #    the pattern is EV_<Predictors>_<Calculation>_<Dataset>
        df_long[['drop','Predictors','Calculation','Dataset']] = (
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
        df_long['Dataset'] = df_long['Dataset'].map({'full':'in_sample', 'eval':'held_out'})
        
        # 4) Sort in the right order
        df_long.sort_values(by=['group_id', 'session_id', 'neuron_id_overall'], inplace=True)
        df_long['EV'] = df_long['EV'].clip(lower=0)

        return df_long
    

    def wide_format_df(self, 
                       calc:Literal['averaged', 'trial'] = 'averaged', 
                       dataset:Literal['in_sample', 'held_out'] = 'in_sample'
                       )->pd.DataFrame:
        selection = ((self.df['Dataset']==dataset) & 
                     (self.df['Calculation']==calc) & 
                     (self.df['Predictors'].isin(['V', 'A', 'Motor'])))
        wide_slice = self.df.loc[selection]

        wide = wide_slice.pivot_table(index=['group_id', 'Region', 'neuron_id_overall'],
                                      columns='Predictors',
                                      values='EV')
        return wide
        

    def add_brain_areas(self):
        self.df['Region'] = np.full(shape = self.df.shape[0], fill_value='', dtype='U7')
        for g, gAreas in self.Areas.items():
            for region_name, where_region in gAreas.area_indices.items():
                mask = (
                (self.df['group_id']     == g) &
                (np.isin(self.df['neuron_id_overall'], where_region))
                )
                # 3) Assign *into* the 'Region' column (note the *string* 'Region', not self.df['Region'])
                self.df.loc[mask, 'Region'] = region_name
        
        return self.df.loc[self.df['Region']!='']


    # TODO: Should quantify EV only during stimulus presentation (and shortly after)
    def comparison(self, calc:Literal['averaged', 'trial'] = 'averaged', 
                   dataset:Literal['in_sample', 'held_out'] = 'in_sample',
                   hist_only:bool = False,
                   show:bool = False):
        
        color_scheme = {'g1pre':'grey', 'g2pre':'darkorange'}

        if hist_only:
            hist_data = self.df.loc[((self.df['Calculation'] == calc) & 
                                    (self.df['Dataset'] == dataset) &
                                    (self.df['Predictors'] == 'Model') &
                                    (self.df['trial_type'] == 'AV'))].copy()
            dp= sns.displot(data = hist_data, x = 'EV', hue = 'group_id', col = 'Region',
                            palette=color_scheme,
                            kind='hist')
            plt.legend(loc = 3)
            plt.tight_layout()
            plt.show()
            plt.close()
            return 'Done'
        
        for reg in self.df['Region'].unique():
            data = self.df.loc[((self.df['Calculation'] == calc) & 
                                (self.df['Dataset'] == dataset) &
                                (self.df['Region'] == reg)
                                & (self.df['EV'] > 0)
                                )].copy()
            
            # Barplot layer
            bp = sns.catplot(data=data, x = 'Predictors', y = 'EV', 
                            order = ['V', 'A', 'Motor', 'Model'],
                            hue = 'group_id', 
                            col = 'trial_type',
                            estimator='mean', # median maybe more faithful but in LM then no AUD EV
                            kind='box', 
                            # edgecolor = 'k',
                            col_order=['V', 'A', 'AV'], 
                            palette=color_scheme,
                            # alpha = 0.7
                            )
            
            bp.legend.set_visible(False)
            plt.suptitle(reg)
            
            if calc != 'trial':
                # strip plot layer (data points)
                for ax, this_tt in zip(bp.axes.flat, bp.col_names):
                    sns.stripplot(data=data.query("trial_type == @this_tt"),
                                x='Predictors', y='EV',
                                hue='group_id', dodge=True,          # align with bars
                                # linewidth=0.1, edgecolor='k',
                                palette=color_scheme,
                                jitter=0.3, size=4,
                                alpha = 0.2, 
                                ax=ax,              # draw on the same subplot
                                legend=False)       # avoid duplicate legends
            
            bp.figure.legend()
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(f'{reg.replace('/', '|')}_EV_{calc}_{dataset}.png', dpi = 300)
            plt.close()

    
    # NOTE: implement LABEL selection based on shuffle test!!!
    def order_neurons(self,
                      calc:Literal['averaged', 'trial'] = 'averaged', 
                      dataset:Literal['in_sample', 'held_out'] = 'in_sample'):
        wide = self.wide_format_df(calc=calc, dataset=dataset)
        group_ids = wide.index.get_level_values('group_id').unique()
        regions = wide.index.get_level_values('Region').unique()
        # mapping predictor → row
        row = {'V':0, 'A':1, 'Motor':2}
        # colours for exclusive neurons
        col_map = {'V':'dodgerblue', 'A':'red', 'Motor':'green'}

        # TODO: which neuron is which category (based on shuffle)
        # now naive MAX
        wide['Label'] = wide.apply(lambda r: r.idxmax(), axis = 1)
        
        for region in regions:
            for group in group_ids:
                mask = (wide.index.get_level_values('Region') == region) & \
                   (wide.index.get_level_values('group_id') == group)
                sub_wide = wide.loc[mask]
                # Order for plot
                v_minus_m = sub_wide['V'] - sub_wide['Motor']
                order = sub_wide.assign(v_m = v_minus_m).sort_values('v_m', ascending=False)

                fig, axes = plt.subplots(3, 1, figsize=(14, 4), sharex=True)
                for n, row_data in order.iterrows():
                    x = np.where(order.index==n)[0][0]                # bar position
                    for pred, ev_value in row_data[['V','A','Motor']].items():
                        ax = axes[row[pred]]
                        color = (col_map[pred] if row_data['Label']==pred else 'gray')
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
    
if __name__ == '__main__':
    gXY, AVs = design_matrix(pre_post='pre', group='both', returnAVs=True, 
                            #  show=True
                             )
    # EV_res = quantify_encoding_models(
    #     gXY=gXY, yTYPE='neuron', 
    #     plot=False, EV=True,
    #     rerun=False
    #     )
    
    # Initialize class
    EVa = EvAnalysis(resultsDF=os.path.join('pydata', 'GLM_ev_results_neuron.csv'), 
                     ARs=[Areas(av, get_indices=True) for av in AVs])

    # Analysis 1) Compare across :
    #   brain regions (separate subfigures, col = ), 
    #   the different Predictor sets (x)
    #   the different groups (hue)

    #   the different trial_types (x =),
    # averaged vs trial - 2 figures
    # in sample vs held out, 2 figures
    # EVa.comparison(calc='averaged')
    # EVa.comparison(calc='averaged', dataset='held_out', 
    #             #    hist_only=True,
    #             # show=True
    #                )
    # EVa.comparison(calc='trial', dataset='held_out',
    #             #    hist_only=True,
    #             # show = True
    #                )
    # EVa.comparison(calc='trial', dataset='held_out')

    # Analysis 2) Distribution of neurons based on explained variance
    EVa.order_neurons(calc='averaged', dataset='held_out')
    
    
    