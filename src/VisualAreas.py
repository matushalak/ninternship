# @matushalak
# group_condition.rois.pkl contains a DATAFRAME with the XYlocation and area name for each neuron
# AllenBrainAtlasOverlay.mat contains the areaNames (MATLAB INDEXING!!! replace 0 with NaN) 
# and mask onto which XY location is mapped!
from matplotlib_venn import venn3
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.Analyze import Analyze, neuron_typesVENN_analysis
from src.analysis_utils import calc_avrg_trace, plot_avrg_trace, snake_plot
from typing import Literal
from scipy.io.matlab import loadmat
from matplotlib import artist, colors
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import skimage as ski
import numpy as np
import pandas as pd
from scipy import ndimage
import os
from src import PYDATA, PLOTSDIR

class Areas:
    '''
    Areas.dfROI contains the following columns:
    -------------------------------------------
    'ABAborder', : distance from nearest border in ABA coordinates
    'ABAcontour', : full 2D contour for each neuron in ABA coordinates
    'ABAregion', : numerical representation of region (0-7)
    'ABAx', 'ABAy', : each neuron's center vector <x, y> in ABA coordinates
    'Area', : area of ROI (how big each ROI is)
    'msdelay', : delay of neuron based on in which order it was scanned
    'px', 'py', : each neuron's center vector <x, y> in FOV coordinates
    'red', : boolean indicating whether neuron was tagged with red indicator
    'x', 'y', : x and y coordinates of full 2D contour in FOV coordinates
    'Region' : (optional if get_indices=True): name of region where neuron is located (V1, etc.) 
    '''
    def __init__(self, 
                 AV : AUDVIS,
                 get_indices:bool = True):
        self.NAME = AV.NAME
        self.area_names = AV.ABA_regions
        self.region_indices = self.separate_areas(AV)
        self.overlay = self.getMask()
        # Adjust to array coordinates
        self.dfROI: pd.DataFrame = self.adjustROIdf(dfROI=AV.rois, 
                                                    overlayDIMS=self.overlay.shape)
        # TODO: fix and integrate into main init
        if get_indices: # for compatibility 
            self.area_indices = self.region_indices
            regions = np.full(shape = self.dfROI.shape[0], fill_value='', dtype='U7')
            for region_name, where_region in self.area_indices.items():
                regions[where_region] = str(region_name)
            self.dfROI['Region'] = regions
        
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
        regions_python = regions_matlab.copy()
        regions_python[~nans] -= 1
        # AM/PM (1, 0) medial dorsal stream; A/RL/AL (2, 3, 4) anterio-lateral dorsal; 
        # LM (5) ventral; V1 (6) primary VIS
        # in numpy logical indexing, must enclose each condition in parentheses
        return {'V1' : np.where(regions_python == 6)[0],
                'AM/PM' : np.where((regions_python == 0) | (regions_python == 1))[0],
                'A/RL/AL' : np.where((regions_python == 2 )
                                    | (regions_python == 3) 
                                    | (regions_python == 4))[0],
                'LM' : np.where(regions_python == 5)[0]
                }
    
    def getMask(self, name:str = 'AllenBrainAtlasOverlay'):
        if os.path.exists(overlay_file := os.path.join(PYDATA, f'{name}.npy')):
            overlay = np.load(overlay_file)
            return overlay
        else:
            overlay_dict = loadmat(overlay_file.replace('npy', 'mat'))
            overlay_mask:np.ndarray = overlay_dict['mask']
            overlay_mask = overlay_mask.astype(float)
            overlay_mask -= 1
            # overlay_mask[overlay_mask == -1] = 255
            return overlay_mask

    def show_neurons(self, 
                     indices: list[np.ndarray] | None = None, 
                     ind_colors: list[str] = ['navy'],
                     ind_labels: list[str | None] = [None],
                     ONLY_indices:bool = False,
                     ZOOM:bool = False, 
                     CONTOURS:bool = False,
                     title: str | None = None,
                     svg:bool = False,
                     show:bool = False):
        if indices is not None:
            if not ONLY_indices:
                if len(indices) == 1:
                    indices = list(self.region_indices.values()) + indices
                    ind_colors = ['mediumseagreen', 'coral', 'rosybrown', 'orchid'] + ind_colors
                    ind_labels = list(self.region_indices.keys()) + ind_labels 
                elif len(indices) == 2:
                    nonresp, resp = indices
                    ind_labels = ind_labels + list(self.region_indices.keys())
                    brainresp = [np.intersect1d(resp, self.region_indices[k]) for k in ind_labels[1:]]
                    indices = [nonresp] + brainresp
                    ind_colors = ind_colors + ['mediumseagreen', 'coral', 'rosybrown', 'orchid']
        else:
            indices = list(self.region_indices.values())
            ind_colors: list[str] = ['mediumseagreen', 'coral', 'rosybrown', 'orchid']
            ind_labels = list(self.region_indices.keys())
        
        alpha = 0.3
        outlines = ski.segmentation.find_boundaries(self.overlay.astype(int), connectivity=1, mode='thick')
        # dilate outline mask (thicker)
        outlines = ndimage.binary_dilation(outlines, iterations=7)
        self.overlay[outlines] = 255
        # Add black outlines
        areaColors = {6:colors.to_rgba('mediumseagreen', alpha=alpha),#V1
                      0:colors.to_rgba('coral', alpha=alpha), # PM
                      1:colors.to_rgba('coral', alpha=alpha), # AM
                      2:colors.to_rgba('rosybrown', alpha=alpha), # A
                      3:colors.to_rgba('rosybrown', alpha=alpha), # RL
                      4:colors.to_rgba('rosybrown', alpha=alpha), # AL
                      5:colors.to_rgba('orchid', alpha=alpha), # LM
                      255:colors.to_rgba('black', alpha=1), # outlines
                      -1:colors.to_rgba('white', alpha=1)} # background
        
        # Raster image, doesn't work with SVG
        # overlayRGBA = np.dstack([self.overlay for _ in range(4)])
        # for area, RGBAcolor in areaColors.items():
        #     overlayRGBA[(self.overlay==area),:] = RGBAcolor
        
        fig, ax = plt.subplots()
        # im = ax.imshow(overlayRGBA) # raster image
        labels  = sorted(areaColors)                         # [-1, 0, 1, …, 6, 255]
        levels  = np.array(labels, dtype=float) - 0.5
        levels  = np.append(levels, labels[-1] + 0.5)

        # colour list must match the labels order
        cmap_list = [areaColors[v] for v in labels]

        # draw filled contours  ->  each region becomes one closed SVG path
        im = ax.contourf(self.overlay,
                        levels=levels,
                        colors=cmap_list,
                        antialiased=False)
        
        if ZOOM:
            maxX, minX = [], []
            maxY, minY = [], []
        for ii, (inds, indcol, indlab) in enumerate(zip(indices, ind_colors, ind_labels)):
            if CONTOURS:
                for i in inds:
                    x, y = self.dfROI.ABAcontour.iloc[i]['x'], self.dfROI.ABAcontour.iloc[i]['y']
                    # Check that on ABA scale
                    # print('Centers:', self.dfROI.ABAx.iloc[i], self.dfROI.ABAy.iloc[i])
                    # print('Contours:', x, y) 
                    coords = np.column_stack((x, y))
                    patch = Polygon(coords,
                                    closed=True,
                                    facecolor=indcol,
                                    edgecolor='none',
                                    alpha= 1 if indlab != 'Unresponsive' else 0.6,
                                    zorder = 10)
                    ax.add_patch(patch)
            else:
                sc = ax.scatter(self.dfROI.ABAx.iloc[inds],
                                self.dfROI.ABAy.iloc[inds],
                                s = 7 if indlab != 'Unresponsive' else 5.5,
                                alpha= 1 if indlab != 'Unresponsive' else 0.6,
                                color = indcol,
                                label = indlab if indlab != 'Unresponsive' else None) 
            if ZOOM:
                maxX.append(self.dfROI.ABAx.iloc[inds].max())
                minX.append(self.dfROI.ABAx.iloc[inds].min())
                maxY.append(self.dfROI.ABAy.iloc[inds].max())
                minY.append(self.dfROI.ABAy.iloc[inds].min())
        
        if ZOOM:
            ax.set_xlim(min(minX)-50, max(maxX)+50)
            ax.set_ylim(min(minY)-50, max(maxY)+50)

        if ind_labels is not None:
            ax.legend(loc = 1)
        
        if title is not None:
            fig.suptitle(title)
        
        ax.set_axis_off()

        # Necessary to flip because in cartesian coordinates origin bottom left
        # image processing libraries assume origin is top left
        ax.invert_yaxis()
        fig.tight_layout()
        if show:
            plt.show()
            plt.close()
            return None
        
        if svg:
            plt.savefig(f'{title}_ALL_NEURONS_ABA.svg')
        else:
            plt.savefig(f'{title}_ALL_NEURONS_ABA.png', dpi = 500)
        plt.close()

    def adjustROIdf(self, dfROI:pd.DataFrame, overlayDIMS:tuple[int,int]
                    ) -> pd.DataFrame:
        yd, xd = overlayDIMS
        dfROI['ABAx'] *= xd
        dfROI['ABAy'] *= yd
        # adjust contours
        for i in range(dfROI.shape[0]):
            dfROI['ABAcontour'].iloc[i]['x'] *= xd
            dfROI['ABAcontour'].iloc[i]['y'] *= yd
        return dfROI
    
    def adjacencyMATRIX(self, 
                        indices:np.ndarray | None = None,
                        signal:np.ndarray | None = None,
                        ) -> np.ndarray:
        '''
        By default returns (neurons x neurons) 2D adjacency matrix with all pairwise distances of neurons
        based on their (x,y) coordinate in the ABA coordinate system. This is to be able to compare distances
        more easily across sessions and brain regions. Because the coordinate system is based on the 
        ABA regions map the distance is arbitrary

        NOTE: it should be possible to find a mapping between ABAcoord units and metric units

        if signal is provided, returns (neurons x neurons x 2) 3D adjacency matrix, with 
            first depth corresponding to the pairwise neuron distances, and the 
            second corresponding to pairwise signal comparisons according to np.corrcoef
        '''
        # Get df with positions of each neuron
        if indices is not None:
            posDF = self.dfROI.iloc[indices][['ABAx', 'ABAy']].copy()
        else:
            posDF = self.dfROI[['ABAx', 'ABAy']].copy()
        
        # position vectors (neurons x coords)
        positions:np.ndarray = posDF.to_numpy()
        positions = np.array(positions, dtype=float)
        # Need newaxis to have notion of column vs row vector
        # both xs and ys are now (neurons x 1) column vectors
        xs, ys = positions[:, 0, np.newaxis], positions[:, 1, np.newaxis]
        
        # get pairwise differences
        dx = xs - xs.T
        dy = ys - ys.T
        # difference vectors 
        # vdiff is (neurons x neurons x 2) array
        # each i x j x 2 slice (for i, j in neurons) is a (x, y) 
        # difference vector between coordinates of neuron i and neuron j
        vdiff = np.dstack((dx, dy))
        # get norm of the difference vector to obtain distances in a (neuron x neuron) distance matrix
        # Vector norm: ||(x,y)|| = sqrt(x^2 + y^2)
        adjM = np.sqrt(np.sum(np.square(vdiff), axis = 2))
        
        # Correlation between signals
        # TODO / NOTE: could also consider cross-correlation function (signal processing sense) to identify 
        # the optimal LAG between each neuron cross-correlation function would peak at that lag
        # for that could use np.correlate (1D) or sp.correlate2D
        if signal is not None:
            assert len(signal.shape) in (2,3), 'signal must be either 2D (timestamps x neurons) or 3D (trials x timestamps x neurons)'
            if len(signal.shape) == 3:
                # flatten trials X rows
                signal = signal.reshape(-1, signal.shape[2])
            # matrix of neuron-neuron signal correlations
            corrM = np.corrcoef(signal, rowvar=False)
            # add as second layer to adjacency matrix
            adjM = np.dstack((adjM, corrM))

        return adjM    

#%% Analyses by areas
def by_areas_VENN(svg:bool=False,
                  pre_post: Literal['pre', 'post', 'both'] = 'both'):
    AVs : tuple[AUDVIS] = load_in_data(pre_post=pre_post)
    ANs : tuple[Analyze] = [Analyze(av) for av in AVs]
    
    for i, (AV, AN) in enumerate(zip(AVs, ANs)):
        responsive_all_areas = AN.TT_RES[-1]
        AR : Areas = Areas(AV)
         # venn diagram figure
        VennRegfig, VennRegaxs = plt.subplots(nrows = 1, ncols = 4, figsize = (16, 4))
        for iarea, (area_name, area_ax) in enumerate(zip(AR.region_indices, VennRegaxs.flatten())):
            responsive_this_area = [np.intersect1d(area_indices := AR.region_indices[area_name], resp_all) 
                                    for resp_all in responsive_all_areas]
            # now ready to group neurons based on responsiveness
            neuron_groups_this_area = AN.neuron_groups(responsive_this_area)
            
            # prep for venn diagram
            plot_args = neuron_groups_this_area['diagram_setup']
            total_responsive_this_area = len(neuron_groups_this_area['TOTAL'])
            area_ax.set_title(f'{area_name} ({total_responsive_this_area}/{area_indices.size})')
            venn3(*plot_args, ax = area_ax,
                  # percentage of responsive neurons
                  subset_label_formatter=lambda x: str(x) + "\n(" + f"{(x/total_responsive_this_area):1.0%}" + ")")
        
        VennRegfig.suptitle(f'{AV.NAME}')
        VennRegfig.tight_layout()
        if not svg:
            VennRegfig.savefig(f'{AV.NAME}_VennDiagramAREAS.png', dpi = 300)
        else:
            VennRegfig.savefig(f'{AV.NAME}_VennDiagramAREAS.svg')
        plt.close()
    print('Done with venn diagrams!')


def by_areas_TSPLOT(GROUP_type:Literal['modulated', 
                                        'modality_specific', 
                                        'all',
                                        'TOTAL'] = 'modulated',
                    add_CASCADE: bool = False, 
                    pre_post: Literal['pre', 'post', 'both'] = 'pre',
                    svg:bool = False):
    # Load in all data and perform necessary calculations
    AVs : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANs : list[Analyze] = [Analyze(av) for av in AVs]

    # prepare everything for plotting
    if GROUP_type == 'modulated':
        tsnrows = 2
        NEURON_types = ('VIS', 'AUD')
    elif GROUP_type == 'TOTAL':
        tsnrows = 4
        NEURON_types = ('TOTAL',)
    else:
        tsnrows = 3
        NEURON_types = ('VIS', 'AUD', 'MST')

    match pre_post:
        case 'both':
            linestyles = ('-', '--', '-', '--') # pre -, post --
            if GROUP_type != 'TOTAL':
                colors = {'VIS':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
                        'AUD':('darkred', 'darkred', 'coral', 'coral'),
                        'MST':('darkmagenta','darkmagenta', 'orchid', 'orchid')}
            else:
                colors = {'V1':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
                        'AM/PM':('darkred', 'darkred', 'coral', 'coral'),
                        'A/RL/AL':('saddlebrown', 'saddlebrown', 'rosybrown', 'rosybrown'),
                        'LM':('darkmagenta','darkmagenta', 'orchid', 'orchid')}
        case 'pre':
            linestyles = ('-', '-') # pre -, 
            if GROUP_type != 'TOTAL':
                colors = {'VIS':('darkgreen','mediumseagreen'),
                        'AUD':('darkred', 'coral'),
                        'MST':('darkmagenta', 'orchid')}
            else:
                colors = {'V1':('darkgreen', 'mediumseagreen'),
                        'AM/PM':('darkred', 'coral'),
                        'A/RL/AL':('saddlebrown', 'rosybrown'),
                        'LM':('darkmagenta', 'orchid')}

        case 'post':
            linestyles = ('--', '--') # pre -, 
            if GROUP_type != 'TOTAL':
                colors = {'VIS':('darkgreen','mediumseagreen'),
                        'AUD':('darkred', 'coral'),
                        'MST':('darkmagenta', 'orchid')}
            else:
                colors = {'V1':('darkgreen', 'mediumseagreen'),
                        'AM/PM':('darkred', 'coral'),
                        'A/RL/AL':('saddlebrown', 'rosybrown'),
                        'LM':('darkmagenta', 'orchid')}
    
    tsncols = 4 # 4 combined trial types
    # prepare timeseries figure and axes for four separate figures
    if GROUP_type != 'TOTAL':
        Artists = [plt.subplots(nrows=tsnrows, ncols=tsncols, 
                                sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * 3)) 
                                for _ in range(4)]
        
        if GROUP_type == 'all':
            SnakeArtists = [plt.subplots(nrows=tsnrows*len(AVs), ncols=tsncols, 
                                    sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * len(AVs) * 3)) 
                                    for _ in range(4)]
    else:
        Artists = plt.subplots(nrows=tsnrows, ncols=tsncols, 
                                sharex='col', sharey='all', figsize = ((tsncols * 3) + .8, tsnrows * 3)) 
        
        SnakeArtists = plt.subplots(nrows=tsnrows*len(AVs), ncols=tsncols, 
                        sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * len(AVs) * 3))
                       
    trials = ['VT', 'AT', 'MS+', 'MS-']
    # Collect data for Quantification
    if GROUP_type == 'TOTAL':
        QuantDict = {'Group':[], 'BRAIN_AREA': [], 'neuronID': [],
                    'F_VIS':[], 'F_AUD':[],'F_MST+':[],'F_MST-':[]}
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
                singleNeuronRes = AN.tt_BY_neuron_group(group, GROUP_type, 
                                                        BrainRegionIndices = area_indices,
                                                        return_single_neuron_data=True)
                
                TT_info, group_size, ind_neuron_traces, ind_neuron_pvals, Fresps = singleNeuronRes
                
                # Add quantification data
                if GROUP_type == 'TOTAL':
                    QuantDict['Group'] += [AV.NAME] * group_size
                    QuantDict['BRAIN_AREA'] += [area_name] * group_size
                    QuantDict['F_VIS'] += [*Fresps[0]]
                    QuantDict['F_AUD'] += [*Fresps[1]]
                    QuantDict['F_MST+'] += [*Fresps[2]]
                    QuantDict['F_MST-'] += [*Fresps[3]]
                    QuantDict['neuronID'] += [*np.intersect1d(AN.NEURON_groups['TOTAL'], 
                                                              area_indices)]

                # also plot cascade traces
                if add_CASCADE:
                    CASCADE_TT_info, _ = AN.tt_BY_neuron_group(group, GROUP_type, 
                                                               SIGNALS_TO_USE= AN.CASCADE,
                                                               BrainRegionIndices=area_indices) 
                
                # area determines which Artists we use
                if GROUP_type != 'TOTAL':
                    ts_fig, ts_ax = Artists[iarea]
                    if GROUP_type == 'all':
                        snake_fig, snake_ax = SnakeArtists[iarea]
                else:
                    ts_fig, ts_ax = Artists
                    snake_fig, snake_ax = SnakeArtists
                # different Trial Types for each of which we want a separate subplot in each figure
                COL = colors[group][icond] if GROUP_type != 'TOTAL' else colors[area_name][icond]
                for itt, (avr, sem) in enumerate(TT_info):
                    plot_avrg_trace(time = AN.time, avrg=avr, SEM = sem, 
                                    Axis=ts_ax[ig, itt] if GROUP_type != 'TOTAL' else ts_ax[iarea, itt],
                                    title = trials[itt] if (
                                        ig == 0 if GROUP_type != 'TOTAL' else iarea == 0) else False,
                                    label = f'{AV.NAME} ({group_size})' if itt == 0 else None, 
                                    col = COL, lnstl=linestyles[icond],
                                    vspan=True if icond == 0 else False,
                                    tt=itt)
                    if GROUP_type == 'TOTAL':
                        ts_ax[iarea,itt].spines['top'].set_visible(False)
                        ts_ax[iarea,itt].spines['right'].set_visible(False)
                        # suppress the x-axis everywhere EXCEPT the last row
                        if iarea != len(list(AR.region_indices)) - 1:
                            ts_ax[iarea, itt].spines['bottom'].set_visible(False)
                            ts_ax[iarea, itt].tick_params(axis='x', bottom=False, labelbottom=False)

                        # suppress the y-axis everywhere EXCEPT the first column
                        if itt != 0:
                            ts_ax[iarea, itt].spines['left'].set_visible(False)
                            ts_ax[iarea, itt].tick_params(axis='y', left=False, labelleft=False, right=False)

                    if add_CASCADE:
                        plot_avrg_trace(time = AN.time, avrg=CASCADE_TT_info[itt][0], SEM = None, 
                                        Axis=ts_ax[ig, itt] if GROUP_type != 'TOTAL' else ts_ax[iarea, itt],
                                        label = 'Est. FR' if itt == len(list(TT_info)) - 1 else None, 
                                        col = COL, lnstl=linestyles[icond], alph=.5, vspan=False)
                        
                    # Add snake plot - of already the significant neurons in the region
                    if GROUP_type == 'all' or GROUP_type == 'TOTAL':
                        snake_plot(ind_neuron_traces[itt], 
                                stats = ind_neuron_pvals[itt],
                                trial_window_frames=AV.TRIAL, 
                                time=AN.time,
                                Axis=snake_ax[(len(ANs)*ig)+icond,itt
                                              ] if GROUP_type != 'TOTAL' else snake_ax[(len(ANs)*iarea)+icond, itt],
                                MODE = 'onset',
                                cmap=COL)

                    if icond == len(AVs) - 1:
                        if GROUP_type != 'TOTAL':
                            if ig == len(NEURON_types) - 1:
                                ts_ax[ig, itt].set_xlabel('Time (s)')

                            if itt == 0:
                                ts_ax[ig, itt].set_ylabel('z(∆F/F)')
                        # GROUP_TYPE == TOTAL
                        else:
                            if iarea == len(list(AR.region_indices)) - 1:
                                ts_ax[iarea, itt].set_xlabel('Time (s)')
                                ts_ax[iarea, itt].spines['bottom'].set_visible(True)
                                ts_ax[iarea, itt].tick_params(axis='x', bottom=True, labelbottom=True)

                            if itt == 0:
                                ts_ax[iarea, itt].set_ylabel('z(∆F/F)')
                                ts_ax[iarea, itt].spines['left'].set_visible(True)
                                ts_ax[iarea, itt].tick_params(axis='y', left=True, labelleft=True)                                
                                # ts_ax[iarea, itt].legend(loc = 4, fontsize = 8)

                        if itt == len(list(TT_info)) - 1:
                            twin = ts_ax[ig, itt].twinx() if GROUP_type != 'TOTAL' else ts_ax[iarea, itt].twinx()
                            twin.spines['top'].set_visible(False)
                            twin.spines['right'].set_visible(False)
                            twin.spines['left'].set_visible(False)
                            twin.spines['bottom'].set_visible(False)
                            
                            rowLab = NEURON_types[ig] if GROUP_type != 'TOTAL' else area_name
                            twin.set_ylabel(rowLab, rotation = 270, 
                                            va = 'bottom', 
                                            color = COL,
                                            fontsize = 20)
                            twin.set_yticks([])

                if GROUP_type != 'TOTAL':
                    if ig == len(NEURON_types) - 1 and icond == len(ANs) - 1:
                        if GROUP_type == 'all':
                            # snake_fig.suptitle(f'{area_name} neurons')
                            snake_fig.tight_layout()
                            # snake_fig.legend(loc = 'outside center right')
                            snake_fig.savefig(f'[{area_name.replace('/', '|')}]Neuron_type({GROUP_type})SNAKE({pre_post}).png', dpi = 500)

                        ts_fig.suptitle(f'{area_name} neurons')
                        ts_fig.tight_layout(rect = [0,0,0.85,1])
                        ts_fig.legend(loc = 'outside center right')
                        ts_fig.savefig(f'[{area_name.replace('/', '|')}]Neuron_type({GROUP_type})_average_res({pre_post}).png', dpi = 300)
    
    if GROUP_type == 'TOTAL':
        snake_fig.tight_layout()
        ts_fig.tight_layout()
        if not svg:
            snake_fig.savefig(f'Neuron_type({GROUP_type})SNAKE({pre_post}).png', dpi = 1000)
            ts_fig.savefig(f'Neuron_type({GROUP_type})_average_res({pre_post}).png', dpi = 300)
        else:
            snake_fig.savefig(f'Neuron_type({GROUP_type})SNAKE({pre_post}).svg')
            ts_fig.savefig(f'Neuron_type({GROUP_type})_average_res({pre_post}).svg')
        QuantDF: pd.DataFrame = pd.DataFrame(QuantDict)
        return QuantDF

# TODO: Two-way ANOVA / Kruskal-Wallis
# Main effect of GROUP & Main effect of TT + Post-hoc (Tukey)
def Quantification(df: pd.DataFrame,
                   pre_post: Literal['pre', 'post', 'both'] = 'pre',
                   svg:bool = False):
    AVs : tuple[AUDVIS] = load_in_data(pre_post)
    AreasN_neurons = dict()
    for AV in AVs:
        AR = Areas(AV)
        AreasN_neurons[AV.NAME] = {k: v[0].size for k,v in AR.region_indices.items()}
    # Melt from wide to long
    df_long = df.melt(
            id_vars=['Group', 'BRAIN_AREA', 'neuronID'],      # columns to keep
            value_vars=['F_VIS','F_AUD','F_MST+','F_MST-'],    # columns to unpivot
            var_name='TT',                                     # new column for variable names
            value_name='F'                                     # new column for the values
        )
    df_long['TT'] = df_long['TT'].str.replace(r'^F_', '', regex=True)
    colors = {'V1':{'g1pre':'darkgreen', 'g2pre':'mediumseagreen'},
            'AM/PM':{'g1pre':'darkred', 'g2pre':'coral'},
            'A/RL/AL':{'g1pre':'saddlebrown', 'g2pre':'rosybrown'},
            'LM':{'g1pre':'darkmagenta', 'g2pre':'orchid'}}
    
    # TODO:
    import statsmodels.formula.api as smf
    # smf.ols()
    
    for ar, arDF in df_long.groupby('BRAIN_AREA'):
        bp = sns.catplot(arDF, x = 'TT', y='F', hue = 'Group', 
                    kind = 'violin', split = True,
                    inner = 'quart', palette=colors[ar], legend=False)
        ax = bp.ax
        fg = bp.figure
        hue_order = arDF['Group'].unique().tolist()       # e.g. ['g1pre','g2pre']
        new_labels = [f"{hue_order[0]}   {hue_order[1]}\n{tt}" 
                      for tt in arDF['TT'].unique()]
        ax.set_xticklabels(new_labels)
        ax.set_xlabel('')
        # annotations and stats
        # split your pairs
        between_pairs = [
            (("VIS", "g1pre"), ("VIS", "g2pre")),
            (("AUD", "g1pre"), ("AUD", "g2pre")),
            (("MST+", "g1pre"), ("MST+", "g2pre")),
            (("MST-", "g1pre"), ("MST-", "g2pre")),
        ]
        within_pairs = [
            (("MST-", "g1pre"), ("MST+", "g1pre")),
            (("MST-", "g2pre"), ("MST+", "g2pre")),
        ]

        # first: Mann‑Whitney for all between‑group comparisons
        annot_bw = Annotator(
            ax,
            between_pairs,
            data=arDF,
            x='TT',
            y='F',
            hue='Group'
        )
        annot_bw.configure(
            test='t-test_ind',#'Mann-Whitney',              # between‑group unpaired
            comparisons_correction='Bonferroni',
            text_format='star',
            loc='outside',
            hide_non_significant = False,
            correction_format="replace"
        )
        annot_bw.apply_and_annotate()

        # then: Wilcoxon signed‑rank (paired) for within‑group
        annot_wi = Annotator(
            ax, within_pairs,
            data=arDF, x='TT', y='F', hue='Group'
        )
        annot_wi.configure(
            test='t-test_paired',#'Wilcoxon',
            comparisons_correction='Bonferroni',
            text_format='star',
            loc='outside',
            hide_non_significant = False,
            correction_format="replace"
        )
        annot_wi.apply_and_annotate()
        fg.tight_layout()
        if not svg:
            fg.savefig(f'{ar.replace('/', '|')}_traces_QUANT.png', dpi = 300)
        else:
            fg.savefig(f'{ar.replace('/', '|')}_traces_QUANT.svg')
        plt.close()

def recordedNeurons(pre_post: Literal['pre', 'post', 'both'] = 'pre',
                    svg:bool = False):
    AVs : list[AUDVIS] = load_in_data(pre_post=pre_post)
    # ANs : list[Analyze] = [Analyze(av) for av in AVs]
    ARs: list[Areas] = [Areas(av) for av in AVs]

    for av, ar in zip(AVs, ARs):
        AN = Analyze(av)
        # Responsive
        Total_RESP = AN.NEURON_groups['TOTAL']
        Total_RESP_color = 'gold'
        # not responsive
        Total_NOT_RESP = np.where(~np.isin(np.arange(AN.sess_neur[-1][-1]), Total_RESP))[0]
        Total_NOT_RESP_color = 'dimgray'
        
        ar.show_neurons(title=av.NAME, 
                        indices=[Total_NOT_RESP, Total_RESP], 
                        ind_colors=[Total_NOT_RESP_color], ind_labels=['Unresponsive'], svg=svg,
                        CONTOURS=True)

if __name__ == '__main__':
    ### Venn diagram of neuron classes in in the 4 different regions
    by_areas_VENN(svg=False, pre_post='pre')

    ### Timeseries plots for neurons from different regions
    by_areas_TSPLOT(GROUP_type = 'modulated', add_CASCADE=False)
    # by_areas_TSPLOT(GROUP_type = 'modality_specific', add_CASCADE=False)
    by_areas_TSPLOT(GROUP_type = 'all', add_CASCADE=False)

    ### TOTAL timeseries plot and quantification
    QuantDF = by_areas_TSPLOT(GROUP_type = 'TOTAL', add_CASCADE=False, svg=False)
    Quantification(QuantDF, svg=False)

    # Recorded neurons plot
    recordedNeurons(svg=True)