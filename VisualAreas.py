# @matushalak
# group_condition.rois.pkl contains a DATAFRAME with the XYlocation and area name for each neuron
# AllenBrainAtlasOverlay.mat contains the areaNames (MATLAB INDEXING!!! replace 0 with NaN) 
# and mask onto which XY location is mapped!
from matplotlib_venn import venn3
from AUDVIS import AUDVIS, Behavior, load_in_data
from Analyze import Analyze, calc_avrg_trace, plot_avrg_trace, neuron_typesVENN_analysis
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as ski
import numpy as np

class Areas:
    def __init__(self, 
                 AV : AUDVIS):
        self.area_names = AV.ABA_regions
        self.region_indices = self.separate_areas(AV)

    @staticmethod
    def separate_areas(AV:AUDVIS) -> dict[str : np.ndarray]:
        '''
        gives back indices of neurons for each area
        '''
        regions_matlab = np.array([*AV.rois['ABAregion']], dtype = int)
        nans = (regions_matlab == 0)
        if np.sum(nans) != 0:
            regions_matlab[nans] = 10000
        # correct for 0 index meaning NaN
        regions_python = regions_matlab[~nans] - 1

        # AM/PM (1, 0) medial dorsal stream; A/RL/AL (2, 3, 4) anterio-lateral dorsal; 
        # LM (5) ventral; V1 (6) primary VIS
        # in numpy logical indexing, must enclose each condition in parentheses
        return {'AM/PM' : np.where((regions_python == 0) | (regions_python == 1)),
                'A/RL/AL' : np.where((regions_python == 2 )
                                    | (regions_python == 3) 
                                    | (regions_python == 4)),
                'LM' : np.where(regions_python == 5),
                'V1' : np.where(regions_python == 6)}


def by_areas_VENN():
    AVs : tuple[AUDVIS] = load_in_data()
    ANs : tuple[Analyze] = [Analyze(av) for av in AVs]
    
    for i, (AV, AN) in enumerate(zip(AVs, ANs)):
        responsive_all_areas = AN.TT_RES[-1]
        AR : Areas = Areas(AV)
         # venn diagram figure
        VennRegfig, VennRegaxs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8))
        for iarea, (area_name, area_ax) in enumerate(zip(AR.region_indices, VennRegaxs.flatten())):
            responsive_this_area = [np.intersect1d(area_indices := AR.region_indices[area_name], resp_all) 
                                    for resp_all in responsive_all_areas]
            # now ready to group neurons based on responsiveness
            neuron_groups_this_area = AN.neuron_groups(responsive_this_area)
            
            # prep for venn diagram
            plot_args = neuron_groups_this_area['diagram_setup']
            total_responsive_this_area = len(neuron_groups_this_area['TOTAL'])
            area_ax.set_title(f'{area_name} ({total_responsive_this_area}/{area_indices[0].size})')
            venn3(*plot_args, ax = area_ax,
                  # percentage of responsive neurons
                  subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total_responsive_this_area):1.0%}" + ")")
        
        VennRegfig.suptitle(f'{AV.NAME}')
        VennRegfig.tight_layout()
        VennRegfig.savefig(f'{AV.NAME}_VennDiagramAREAS.png', dpi = 200)
        plt.close()
    print('Done with venn diagrams!')



# TODO: 90% reused code from Analyze plotting functions; GENERALIZE!!!
#   NOTE: specify number of figures, subplots and lines in each sublot and prepare the figure in such a way 
#   that once arguments passed in, automatically executes the plot
def by_areas_TSPLOT(GROUP_type:Literal['modulated', 
                                        'modality_specific', 
                                        'all'] = 'modulated'):
    # Load in all data and perform necessary calculations
    AVs : list[AUDVIS] = load_in_data() # -> av1, av2, av3, av4
    ANs : list[Analyze] = [Analyze(av) for av in AVs]

    # prepare everything for plotting
    if GROUP_type == 'modulated':
        tsnrows = 2
        NEURON_types = ('VIS', 'AUD')
    else:
        tsnrows = 3
        NEURON_types = ('VIS', 'AUD', 'MST')

    linestyles = ('-', '--', '-', '--') # pre -, post :
    colors = {'VIS':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
              'AUD':('darkred', 'darkred', 'coral', 'coral'),
              'MST':('darkmagenta','darkmagenta', 'orchid', 'orchid')}
    
    tsncols = 4 # 4 combined trial types
    # prepare timeseries figure and axes for four separate figures
    Artists = [plt.subplots(nrows=tsnrows, ncols=tsncols, 
                            sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * 3)) 
                            for _ in range(4)]

    trials = ['VT', 'AT', 'MS+', 'MS-']
    # NEURON_types correspond to rows of each figure
    for ig, group in enumerate(NEURON_types):
        # Different conditions = different types of recording sessions
        for icond, (AV, AN) in enumerate(zip(AVs, ANs)):
            responsive_all_areas = AN.TT_RES[-1]
            AR : Areas = Areas(AV)
            # different AREAS for each of which we want a separate figure
            for iarea, area_name in enumerate(AR.region_indices):
                area_indices = AR.region_indices[area_name]
                print(AV.NAME, group, area_name)
                TT_info, group_size = AN.tt_BY_neuron_group(group, GROUP_type, 
                                                            BrainRegionIndices = area_indices) 
                
                # area determines which Artists we use
                ts_fig, ts_ax = Artists[iarea]
                # different Trial Types for each of which we want a separate subplot in each figure
                for itt, (avr, sem) in enumerate(TT_info):
                    plot_avrg_trace(time = AN.time, avrg=avr, SEM = sem, Axis=ts_ax[ig, itt],
                                    title = trials[itt] if ig == 0 else False,
                                    label = f'{AV.NAME} ({group_size})' if itt == len(list(TT_info)) - 1 else None, 
                                    col = colors[group][icond], lnstl=linestyles[icond])

                    if icond == len(AVs) - 1:
                        if ig == len(NEURON_types) - 1:
                            ts_ax[ig, itt].set_xlabel('Time (s)')

                        if itt == 0:
                            ts_ax[ig, itt].set_ylabel('z(∆F/F)')

                        if itt == len(list(TT_info)) - 1:
                            twin = ts_ax[ig, itt].twinx()
                            twin.set_ylabel(NEURON_types[ig], rotation = 270, 
                                                        va = 'bottom', 
                                                        color = colors[group][0],
                                                        fontsize = 20)
                            twin.set_yticks([])

                if ig == len(NEURON_types) - 1 and icond == len(ANs) - 1:
                    ts_fig.suptitle(f'{area_name} neurons')
                    ts_fig.tight_layout(rect = [0,0,0.85,1])
                    ts_fig.legend(loc = 'outside center right')
                    ts_fig.savefig(f'[{area_name.replace('/', '|')}]Neuron_type({GROUP_type})_average_res.png', dpi = 200)

def by_areas_SNAKE():
    pass

if __name__ == '__main__':
    # Venn diagram of neuron classes in in the 4 different regions
    by_areas_VENN()

    # Timeseries plots for neurons from different regions
    by_areas_TSPLOT(GROUP_type = 'modulated')
    by_areas_TSPLOT(GROUP_type = 'modality_specific')
    by_areas_TSPLOT(GROUP_type = 'all')