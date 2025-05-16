from GLM import design_matrix, quantify_encoding_models
from AUDVIS import Behavior, AUDVIS
from VisualAreas import Areas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Literal

# ------------ Class that analyzes explained variance calculated above -----------
# TODO: add brain area column
class EvAnalysis:
    def __init__(self,
                 resultsDF: str | pd.DataFrame,
                 ARs:list[Areas]):
        if isinstance(resultsDF, str):
            resultsDF = pd.read_csv(resultsDF)
        
        # load DF with results
        self.df = resultsDF
        # load in areas into a dictionary
        self.Areas =  {ar.NAME:ar for ar in ARs}
        # absolute neuron IDs
        self.n_per_group = self.add_absolute_neuron_ids()
        # transform to long format for easier plotting
        self.df = self.long_format_df()
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

    def comparison(self, calc:Literal['averaged', 'trial'] = 'averaged', 
                   dataset:Literal['in_sample', 'held_out'] = 'in_sample'):
        
        for reg in self.df['Region'].unique():
            data = self.df.loc[((self.df['Calculation'] == calc) & 
                                (self.df['Dataset'] == dataset) &
                                (self.df['Region'] == reg))].copy()
            bp = sns.catplot(data=data, x = 'Predictors', y = 'EV', 
                            hue = 'group_id', col = 'trial_type',
                            kind='bar')
            
            plt.suptitle(reg)
            plt.show()
            plt.close()
    
if __name__ == '__main__':
    gXY, AVs = design_matrix(pre_post='pre', group='both', returnAVs=True)
    EV_res = quantify_encoding_models(
        gXY=gXY, yTYPE='neuron', 
        plot=False, EV=True,
        rerun=False
        )
    
    # Initialize class
    EVa = EvAnalysis(EV_res, 
                     ARs=[Areas(av, get_indices=True) for av in AVs])

    # Analysis 1) Compare across :
    #   brain regions (separate subfigures, col = ), 
    #   the different Predictor sets (x)
    #   the different groups (hue)

    #   the different trial_types (x =),
    # averaged vs trial - 2 figures
    # in sample vs held out, 2 figures
    EVa.comparison(calc='averaged')
    EVa.comparison(calc='averaged', dataset='held_out')
    # EVa.comparison(calc='trial')
    # EVa.comparison(calc='trial', dataset='held_out')
    
    
    