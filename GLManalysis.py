# ------------ Class that analyzes explained variance calculated above -----------
class EvAnalysis:
    def __init__(self,
                 resultsDF: str | pd.DataFrame):
        if isinstance(resultsDF, str):
            resultsDF = pd.read_csv(resultsDF)
        
        self.df = resultsDF
        self.n_per_group = self.add_absolute_neuron_ids()
    
    def add_absolute_neuron_ids(self):
        n_per_group = self.df.groupby("group_id").size().values // self.df["trial_type"].nunique()
        neuron_ids = np.concatenate([np.repeat(np.arange(npg), repeats=self.df["trial_type"].nunique())
                                    for npg in n_per_group])
        self.df["neuron_id_overall"] = neuron_ids
        return n_per_group