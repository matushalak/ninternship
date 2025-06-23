# @matushalak
# group_condition.rois.pkl contains a DATAFRAME with the XYlocation and area name for each neuron
# AllenBrainAtlasOverlay.mat contains the areaNames (MATLAB INDEXING!!! replace 0 with NaN) 
# and mask onto which XY location is mapped!
from matplotlib_venn import venn3
from src.AUDVIS import AUDVIS, Behavior, load_in_data
from src.Analyze import Analyze, neuron_typesVENN_analysis
import src.analysis_utils as anut
from typing import Literal
from scipy.io.matlab import loadmat
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import artist, colors
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from matplotlib.cm     import ScalarMappable
from matplotlib.patches import Polygon
import matplotlib.cm as mplcm
from joblib import delayed, Parallel, cpu_count

from statannotations.Annotator import Annotator
import seaborn as sns
import seaborn.objects as so
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
        self.sessionNEURONS = AV.session_neurons
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
                     show:bool = False,
                     values: np.ndarray | None = None, # continuous values, one per ROI
                     cmap: str | plt.Colormap = 'viridis', # any matplotlib colormap
                     colorbar: str | None = None, # colorbar label,
                     areasalpha:float = 0.3,
                     suppressAreaColors:bool = False,
                     interpolate_vals:bool = False # only works with values
                     ):
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
        
        alpha = areasalpha if not suppressAreaColors else 0
        outlines = ski.segmentation.find_boundaries(self.overlay.astype(int), connectivity=1, mode='thick')
        # dilate outline mask (thicker)
        outlines = ndimage.binary_dilation(outlines, iterations=7)
        overlay = self.overlay.copy()
        overlay[outlines] = 255
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
        im = ax.contourf(overlay,
                        levels=levels,
                        colors=cmap_list,
                        antialiased=False)
        
        if ZOOM:
            maxX, minX = [], []
            maxY, minY = [], []
        
        if values is not None:
            # determine min/max
            _vmin = -0.2#float(np.nanmin(values))
            _vmax = 0.2#float(np.nanmax(values))
            norm = Normalize(vmin=_vmin, vmax=_vmax)
            sm   = ScalarMappable(norm=norm, cmap=cmap)

        for ii, (inds, indcol, indlab) in enumerate(zip(indices, ind_colors, ind_labels)):
            if interpolate_vals:
                x = self.dfROI.ABAx.iloc[inds].values
                y = self.dfROI.ABAy.iloc[inds].values
                values = np.clip(values, -0.2, 0.2)
                z = values[inds]
                triang = tri.Triangulation(x, y)
                # 3) compute the maximum edge‐length for each triangle
                pts = np.column_stack([x, y])
                max_edge = np.zeros(len(triang.triangles))
                for i, (i0, i1, i2) in enumerate(triang.triangles):
                    a = np.linalg.norm(pts[i0] - pts[i1])
                    b = np.linalg.norm(pts[i1] - pts[i2])
                    c = np.linalg.norm(pts[i2] - pts[i0])
                    max_edge[i] = max(a, b, c)

                # 4) pick a threshold: e.g. everything longer than the 90th percentile
                thr = np.percentile(max_edge, 99.5)
                mask = max_edge > thr
                triang.set_mask(mask)
                cf = ax.tricontourf(triang, z, levels = 4, cmap = cmap)

            if CONTOURS:
                for i in inds:
                    x, y = self.dfROI.ABAcontour.iloc[i]['x'], self.dfROI.ABAcontour.iloc[i]['y']
                    # Check that on ABA scale
                    # print('Centers:', self.dfROI.ABAx.iloc[i], self.dfROI.ABAy.iloc[i])
                    # print('Contours:', x, y) 
                    coords = np.column_stack((x, y))
                    if values is not None:
                        try:
                            rgba = sm.to_rgba(values[i])
                        except IndexError:
                            print(f'Index {i} out of range!')
                            continue

                        alpha_here = 1.0
                    else:
                        rgba = indcol
                        alpha_here = (1.0 if indlab!='Unresponsive' else 0.6)

                    patch = Polygon(coords,
                                    closed=True,
                                    facecolor=rgba,
                                    edgecolor='none',
                                    alpha=alpha_here,
                                    zorder=10)
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
        
        if values is not None and isinstance(colorbar, str):
            fig.colorbar(sm, ax=ax, label=colorbar)
        
        if ZOOM:
            ax.set_xlim(min(minX)-50, max(maxX)+50)
            ax.set_ylim(min(minY)-50, max(maxY)+50)

        if ind_labels is not None and values is None:
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
            fig.savefig(os.path.join(PLOTSDIR, f'{title}_ALL_NEURONS_ABA.svg'))
            plt.close()
        else:
            fig.savefig(os.path.join(PLOTSDIR,f'{title}_ALL_NEURONS_ABA.png'), dpi = 500)
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

#%%--------- Analyses by areas -------------
def by_areas_VENN(svg:bool=False,
                  pre_post: Literal['pre', 'post', 'both'] = 'both'
                  )->pd.DataFrame:
    AVs : tuple[AUDVIS] = load_in_data(pre_post=pre_post)
    ANs : tuple[Analyze] = [Analyze(av) for av in AVs]
    
    proportion_df = {'Group':[], 'Area':[], 'Type':[], 
                     'x':[], 'y':[], 'NeuronID':[]}
    areasDict:dict[str:Areas] = {}
    sessionsDict:dict[str:list[tuple[int,int]]] = {}
    for i, (AV, AN) in enumerate(zip(AVs, ANs)):
        g = 'DR' if 'g1' in AV.NAME else 'NR'
        gname = AV.NAME.replace(AV.NAME[:2], g)
        responsive_all_areas = AN.TT_RES[-1]
        AR : Areas = Areas(AV)
        areasDict[gname] = AR
        sessionsDict[gname] = AV.session_neurons
         # venn diagram figure
        VennRegfig, VennRegaxs = plt.subplots(nrows = 1, ncols = 4, figsize = (16, 4))
        for iarea, (area_name, area_ax) in enumerate(zip(AR.region_indices, VennRegaxs.flatten())):
            responsive_this_area = [np.intersect1d(area_indices := AR.region_indices[area_name], resp_all) 
                                    for resp_all in responsive_all_areas]
            # now ready to group neurons based on responsiveness
            neuron_groups_this_area = AN.neuron_groups(responsive_this_area)
            
            for nt in ['VIS', 'AUD', 'MST']:
                neurtype = nt[0]
                # neuron type indices in this area
                ntindices = neuron_groups_this_area[nt]
                nneur = len(ntindices)

                # neuron specific
                # x axis is medio-lateral 0 is medial, max is lateral
                proportion_df['x'] += AR.dfROI.ABAx.iloc[ntindices].tolist()
                # y axis is rostro-causal 0 is caudal, max is rostral
                proportion_df['y'] += AR.dfROI.ABAy.iloc[ntindices].tolist()

                proportion_df['Group'] += [gname] * nneur
                proportion_df['Area'] += [area_name] * nneur
                proportion_df['Type'] += [neurtype] * nneur

                proportion_df['NeuronID'] += ntindices.tolist()

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
            VennRegfig.savefig(os.path.join(PLOTSDIR, f'{AV.NAME}_VennDiagramAREAS.png'), 
                               dpi = 300)
        else:
            VennRegfig.savefig(os.path.join(PLOTSDIR, f'{AV.NAME}_VennDiagramAREAS.svg'))
        plt.close()
    print('Done with venn diagrams!')

    propDF = pd.DataFrame(proportion_df)
    counts = (propDF
              .groupby(['Group', 'Area', 'Type'])
              .size()
              .reset_index(name="n"))
    
    counts["Proportion"] = ( counts
                    .groupby(["Group","Area"])["n"]
                    .transform(lambda x: x / x.sum()) )
    
    prop = (
        so.Plot(counts, x = 'Group', y = 'Proportion', color = 'Type')
        .facet('Area', order = ['V1', 'AM/PM', 'A/RL/AL', 'LM'])
        .add(so.Bars(), so.Stack())
        .scale(color = {'V':'dodgerblue', 'A':'red', 'M':'goldenrod'})
    )
    plot = prop.plot(pyplot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTSDIR, 'neuron_groupsAREAS.svg'))
    plt.close()
    print('Done with neuron groups proportion plots diagrams!')

    # control counts
    return propDF, areasDict, sessionsDict


def by_areas_TSPLOT(GROUP_type:Literal['modulated', 
                                        'modality_specific', 
                                        'all',
                                        'TOTAL'] = 'modulated',
                    add_CASCADE: bool = False, 
                    pre_post: Literal['pre', 'post', 'both'] = 'pre',
                    combineMST:bool = True,
                    svg:bool = False):
    ''''
    Generates a separate plot (4 areas [rows] x 4 TT [cols]) for each Neuron Type group
    '''
    # Load in all data and perform necessary calculations
    AVs : list[AUDVIS] = load_in_data(pre_post=pre_post) # -> av1, av2, av3, av4
    ANs : list[Analyze] = [Analyze(av) for av in AVs]

    if not os.path.exists(SAVEDIR := os.path.join(PLOTSDIR, 'Amplitude', pre_post, GROUP_type)):
        os.makedirs(SAVEDIR)

    SUFFIX = '.png' if not svg else '.svg'

    # prepare everything for plotting
    if GROUP_type == 'modulated':
        NEURON_types = ('VIS', 'AUD')
    elif GROUP_type == 'TOTAL':
        NEURON_types = ('TOTAL',)
    else:
        NEURON_types = ('VIS', 'AUD', 'MST')

    match pre_post:
        case 'both':
            linestyles = ('-', '--', '-', '--') # pre -, post --
            colors = {'V1':('darkgreen', 'darkgreen' ,'mediumseagreen', 'mediumseagreen'),
                    'AM/PM':('darkred', 'darkred', 'coral', 'coral'),
                    'A/RL/AL':('saddlebrown', 'saddlebrown', 'rosybrown', 'rosybrown'),
                    'LM':('darkmagenta','darkmagenta', 'orchid', 'orchid')}
        case 'pre':
            linestyles = ('-', '-') # pre -, 
            colors = {'V1':('darkgreen', 'mediumseagreen'),
                    'AM/PM':('darkred', 'coral'),
                    'A/RL/AL':('saddlebrown', 'rosybrown'),
                    'LM':('darkmagenta', 'orchid')}

        case 'post':
            linestyles = ('--', '--') # pre -, 
            colors = {'V1':('darkgreen', 'mediumseagreen'),
                    'AM/PM':('darkred', 'coral'),
                    'A/RL/AL':('saddlebrown', 'rosybrown'),
                    'LM':('darkmagenta', 'orchid')}
    
    nfigs = len(NEURON_types)
    tsnrows = 4
    tsncols = 4 - int(combineMST) # 4 combined trial types if MS+ MS- separate, otherwise 3

    # prepare timeseries figure and axes for nfigs separate figures
    Artists = [plt.subplots(nrows=tsnrows, ncols=tsncols, 
                            sharex='col', sharey='all', figsize = ((tsncols * 3) + .8, tsnrows * 3)) 
                            for _ in range(nfigs)]
    
    SnakeArtists = [plt.subplots(nrows=tsnrows*len(AVs), ncols=tsncols, 
                            sharex='col', sharey='row', figsize = ((tsncols * 3) + .8, tsnrows * len(AVs) * 3)) 
                            for _ in range(nfigs)]
                       
    trials = ['VT', 'AT', 'MS+', 'MS-']
    # Collect data for Quantification
    QuantDict = {'Group':[], 'BRAIN_AREA': [], 'NeuronType':[], 'neuronID': [],
                'F_VIS':[], 'F_AUD':[],
                # 'F_MST+':[],'F_MST-':[],
                'F_MST':[]}
    # NEURON_types correspond to rows of each figure
    for itype, neuronTYPE in enumerate(NEURON_types):
        # Different conditions = different types of recording sessions
        for icond, (AV, AN) in enumerate(zip(AVs, ANs)):
            AR : Areas = Areas(AV)
            # different AREAS for each of which we want a separate figure
            for iarea, area_name in enumerate(AR.region_indices):
                area_indices = AR.region_indices[area_name]
                print(AV.NAME, neuronTYPE, area_name)
                singleNeuronRes = AN.tt_BY_neuron_group(neuronTYPE, GROUP_type, 
                                                        BrainRegionIndices = area_indices,
                                                        return_single_neuron_data=True,
                                                        combineMST=combineMST)
                
                TT_info, group_size, ind_neuron_traces, ind_neuron_pvals, Fresps = singleNeuronRes
                
                # Add quantification data
                QuantDict['Group'] += [AV.NAME] * group_size
                QuantDict['BRAIN_AREA'] += [area_name] * group_size
                QuantDict['NeuronType'] += [neuronTYPE] * group_size
                # responses to different trial types
                QuantDict['F_VIS'] += [*Fresps[0]]
                QuantDict['F_AUD'] += [*Fresps[1]]
                QuantDict['F_MST'] += [*Fresps[2]]
                # QuantDict['F_MST+'] += [*Fresps[2]]
                # QuantDict['F_MST-'] += [*Fresps[3]]
                QuantDict['neuronID'] += [*np.intersect1d(AN.NEURON_groups[neuronTYPE], 
                                                              area_indices)]

                # also plot cascade traces
                if add_CASCADE:
                    CASCADE_TT_info, _ = AN.tt_BY_neuron_group(neuronTYPE, GROUP_type, 
                                                               SIGNALS_TO_USE= AN.CASCADE,
                                                               BrainRegionIndices=area_indices,
                                                               combineMST=combineMST) 
                
                # neuron type determines which Artists we use
                ts_fig, ts_ax = Artists[itype]
                snake_fig, snake_ax = SnakeArtists[itype]

                # different Trial Types for each of which we want a separate subplot in each figure
                COL = colors[area_name][icond]
                for itt, (avr, sem) in enumerate(TT_info):
                    anut.plot_avrg_trace(time = AN.time, avrg=avr, SEM = sem, 
                                    Axis=ts_ax[iarea, itt],
                                    title = trials[itt] if iarea == 0 else False,
                                    label = f'{AV.NAME} ({group_size})' if itt == 0 else None, 
                                    col = COL, lnstl=linestyles[icond],
                                    vspan=True if icond == 0 else False,
                                    tt=itt)
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
                        anut.plot_avrg_trace(time = AN.time, avrg=CASCADE_TT_info[itt][0], SEM = None, 
                                        Axis=ts_ax[iarea, itt],
                                        label = 'Est. FR' if itt == len(list(TT_info)) - 1 else None, 
                                        col = COL, lnstl=linestyles[icond], alph=.5, vspan=False)
                        
                    # Add snake plot - of already the significant neurons in the region
                    if GROUP_type == 'all' or GROUP_type == 'TOTAL':
                        anut.snake_plot(ind_neuron_traces[itt], 
                                stats = ind_neuron_pvals[itt],
                                trial_window_frames=AV.TRIAL, 
                                time=AN.time,
                                Axis=snake_ax[(len(ANs)*iarea)+icond, itt],
                                MODE = 'onset',
                                cmap=COL)

                    if icond == len(AVs) - 1:
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
                            twin = ts_ax[iarea, itt].twinx()
                            twin.spines['top'].set_visible(False)
                            twin.spines['right'].set_visible(False)
                            twin.spines['left'].set_visible(False)
                            twin.spines['bottom'].set_visible(False)
                            
                            rowLab = area_name
                            twin.set_ylabel(rowLab, rotation = 270, 
                                            va = 'bottom', 
                                            color = COL,
                                            fontsize = 20)
                            twin.set_yticks([])

        # after both groups of animals for each neuron type
        if GROUP_type == 'all':
            # snake_fig.suptitle(f'{area_name} neurons')
            snake_fig.tight_layout()
            # snake_fig.legend(loc = 'outside center right')
            snake_fig.savefig(os.path.join(SAVEDIR,f'{neuronTYPE}_Neuron_type({GROUP_type})SNAKE({pre_post}).png'), dpi = 500)

            ts_fig.suptitle(f'{neuronTYPE} neurons')
            ts_fig.tight_layout(rect = [0,0,0.85,1])
            ts_fig.legend(loc = 'outside center right')
            ts_fig.savefig(os.path.join(SAVEDIR, f'{neuronTYPE}_Neuron_type({GROUP_type})_average_res({pre_post}){SUFFIX}'), dpi = 300)
    
    if GROUP_type == 'TOTAL':
        snake_fig.tight_layout()
        ts_fig.tight_layout()
        snake_fig.savefig(os.path.join(SAVEDIR, f'Neuron_type({GROUP_type})SNAKE({pre_post}).png', dpi = 500))
        ts_fig.savefig(os.path.join(SAVEDIR, f'Neuron_type({GROUP_type})_average_res({pre_post}){SUFFIX}', dpi = 300))
    
    QuantDF: pd.DataFrame = pd.DataFrame(QuantDict)
    # Transform to Long format
    QuantDF = QuantDF.melt(
        # columns to keep
        id_vars=['Group', 'BRAIN_AREA', 'NeuronType', 'neuronID'],
        # columns to unpivot
        value_vars=['F_VIS','F_AUD',
                    # 'F_MST+','F_MST-',
                    'F_MST'],
        # new column for variable names
        var_name='TT',
        # new column for the values
        value_name='F'
    )
    plt.close()
    return QuantDF

# TODO: Two-way ANOVA / Kruskal-Wallis
# Main effect of GROUP & Main effect of TT + Post-hoc (Tukey)
def Quantification(df_long: pd.DataFrame,
                   pre_post: Literal['pre', 'post', 'both'] = 'pre',
                   svg:bool = False,
                   combineMST:bool = True):
    AVs : tuple[AUDVIS] = load_in_data(pre_post)
    AreasN_neurons = dict()
    for AV in AVs:
        AR = Areas(AV)
        AreasN_neurons[AV.NAME] = {k: v[0].size for k,v in AR.region_indices.items()}
    
    SAVEDIR = os.path.join(PLOTSDIR, 'Amplitude')
    SUFFIX = '.png' if not svg else '.svg'

    df_long['TT'] = df_long['TT'].str.replace(r'^F_', '', regex=True)
    colors = {'V1':{'g1pre':'darkgreen', 'g2pre':'mediumseagreen'},
            'AM/PM':{'g1pre':'darkred', 'g2pre':'coral'},
            'A/RL/AL':{'g1pre':'saddlebrown', 'g2pre':'rosybrown'},
            'LM':{'g1pre':'darkmagenta', 'g2pre':'orchid'}}
    
    for (ar, ntype), arDF in df_long.groupby(['BRAIN_AREA', 'NeuronType']):
        bp = sns.catplot(arDF, x = 'TT', y='F', hue = 'Group', 
                    kind = 'point', palette=colors[ar], dodge = 0.3, 
                    legend=False)
        
        ax = bp.ax
        fg = bp.figure
        hue_order = arDF['Group'].unique().tolist()       # e.g. ['g1pre','g2pre']
        new_labels = [f"{hue_order[0]}   {hue_order[1]}\n{tt}" 
                      for tt in arDF['TT'].unique()]
        ax.set_xticklabels(new_labels)
        ax.set_xlabel('')
        # annotations and stats
        # split your pairs
        if combineMST:
            between_pairs = [
                (("VIS", "g1pre"), ("VIS", "g2pre")),
                (("AUD", "g1pre"), ("AUD", "g2pre")),
                (("MST", "g1pre"), ("MST", "g2pre")),
            ]
            # compare combined mst to unimodal
            within_pairs = [
                (("MST", "g1pre"), ("VIS", "g1pre")),
                (("MST", "g1pre"), ("AUD", "g1pre")),
                (("MST", "g2pre"), ("VIS", "g2pre")),
                (("MST", "g2pre"), ("AUD", "g2pre")),
            ]
        else:
            between_pairs = [
                (("VIS", "g1pre"), ("VIS", "g2pre")),
                (("AUD", "g1pre"), ("AUD", "g2pre")),
                (("MST+", "g1pre"), ("MST+", "g2pre")),
                (("MST-", "g1pre"), ("MST-", "g2pre")),
            ]
            # compare ntype MST+ vs ntype MST -
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
            # between‑group unpaired ['t-test_ind','Mann-Whitney']
            test='Mann-Whitney',
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
            # within-group paired ['t-test_paired','Wilcoxon']
            test='Wilcoxon', 
            comparisons_correction='Bonferroni',
            text_format='star',
            loc='outside',
            hide_non_significant = False,
            correction_format="replace"
        )
        annot_wi.apply_and_annotate()
        fg.tight_layout()
        fg.savefig(os.path.join(SAVEDIR,f'{ar.replace('/', '|'), ntype}_traces_QUANT{SUFFIX}'), dpi = 300)
        plt.close()

def recordedNeurons(pre_post: Literal['pre', 'post', 'both'] = 'pre',
                    svg:bool = False):
    AVs : list[AUDVIS] = load_in_data(pre_post=pre_post)
    # ANs : list[Analyze] = [Analyze(av) for av in AVs]
    ARs: list[Areas] = [Areas(av) for av in AVs]

    for av, ar in zip(AVs, ARs):
        AN = Analyze(av)
        # not responsive
        Total_NOT_RESP = np.where(~np.isin(np.arange(AN.sess_neur[-1][-1]), AN.NEURON_groups['TOTAL']))[0]
        Total_NOT_RESP_color = 'dimgray'

        sessneurs = [np.arange(s0, sE) for s0, sE in av.session_neurons]
        print(sessneurs[-1])
        
        # session neurons
        ar.show_neurons(title=av.NAME + '_sessions', 
                        indices=sessneurs, 
                        ind_colors=mplcm.tab20(range(len(sessneurs))),
                        ind_labels=[f'{i}' for i in range(len(sessneurs))], 
                        svg=svg,
                        areasalpha=0.025,
                        ZOOM=False,
                        CONTOURS=True)
        
        # significant neurons of different types
        ar.show_neurons(title=av.NAME + '_neuronTYPES', 
                        indices=[AN.NEURON_groups['VIS'], AN.NEURON_groups['AUD'], AN.NEURON_groups['MST'], Total_NOT_RESP], 
                        ind_colors=['dodgerblue', 'red', 'goldenrod', Total_NOT_RESP_color], 
                        ind_labels=['V', 'A', 'M', 'Unresponsive'], 
                        svg=svg,
                        areasalpha=0.1,
                        ZOOM=False,
                        CONTOURS=True)

class Architecture:
    def __init__(self,
                 df:pd.DataFrame,
                 ARdict:dict[str:Areas],
                 SESSdict:dict[str:list[tuple[int,int]]]):
        self.df = df
        self.areas = ARdict
        self.session_neurons = SESSdict

        # normalize within each area to 0-1
        self.df = self.normalizeXY()

        if not os.path.exists(SAVEDIR:=os.path.join(PLOTSDIR, 'ARCHITECTURE')):
            os.makedirs(SAVEDIR)
        self.SAVEDIR = SAVEDIR
    
    def normalizeXY(self)->pd.DataFrame:
        '''
        Normalizes x, y across areas and groups
        '''
        # for x traditional min-max is fine (0: medial, 1:lateral)
        self.df['Medio-Lateral'] = (self.df['x'] - self.df['x'].min()
                                    ) / (self.df['x'].max() - self.df['x'].min())
        
        # for y, need to flip the 1 - (min-max); max - x / max - min to get (0: caudal, 1: rostral)
        self.df['Caudo-Rostral'] = (self.df['y'].max() - self.df['y']
                                    ) / (self.df['y'].max() - self.df['y'].min())
        return self.df
    
    def spatial_distribution(self):
        '''
        Makes sense to compare WITHIN group and INTERACTION effects Group x Type
        NOT absolute positions between groups
        '''
        ml = sns.catplot(data = self.df, y = 'Type', x = 'Medio-Lateral', hue = 'Group',
                         kind = 'box', showfliers = False)
        plt.savefig(os.path.join(self.SAVEDIR, 'mediolateral.svg'))
        plt.close()

        # Control - the overall ML
        controlml = sns.catplot(data = self.df, y = 'Group', x = 'Medio-Lateral',
                                kind = 'box', showfliers = False)
        plt.savefig(os.path.join(self.SAVEDIR, 'CONTROLmediolateral.svg'))
        plt.close()

        rc = sns.catplot(data = self.df, x = 'Type', y = 'Caudo-Rostral', hue = 'Group',
                         kind = 'box', showfliers = False)
        plt.savefig(os.path.join(self.SAVEDIR, 'rostrocaudal.svg'))
        plt.close()

        # Control - the overall RC
        controlrc = sns.catplot(data = self.df, x = 'Group', y = 'Caudo-Rostral',
                                kind = 'box', showfliers = False)
        plt.savefig(os.path.join(self.SAVEDIR, 'CONTROLrostrocaudal.svg'))
        plt.close()
    
    def neighbors(self):
        neighborDFs = []
        nullAreas = []
        nullOverall = []

        for g in self.df.Group.unique():
            groupDF = self.df.loc[self.df['Group'] == g,:].copy()
            groupAR:Areas = self.areas[g]   
            adjacencyMatrix:np.ndarray = groupAR.adjacencyMATRIX() # pairwise distances of all recorded neurons
            # neighbor identity and distance analysis
            nbDF = neighbor_analysis(groupDF, adjacencyMatrix, self.session_neurons[g])
            NULLdistAreas = null_distributions(groupDF.copy(), adjacencyMatrix, session_neurons=self.session_neurons[g],
                                               perArea=True, niter=10000)
            NULLdistOverall = null_distributions(groupDF.copy(), adjacencyMatrix, session_neurons=self.session_neurons[g], 
                                                 perArea=False, niter=10000)
            neighborDFs.append(nbDF)
            nullAreas.append(NULLdistAreas)
            nullOverall.append(NULLdistOverall)
        
        # for probability of nearest neighbor analyses
        DF = pd.concat(neighborDFs,ignore_index=True)
        # for distance to nearest neighbor of given type analysis
        DISTDF = pd.melt(DF, id_vars=['Group', 'Area', 'Type', 'NeuronID'], 
                        value_vars=['Vneighbor','Aneighbor','Mneighbor'],
                        var_name='NNtype', value_name='Distance')
        
        NULLareas = pd.concat(nullAreas, ignore_index=True)
        NULLoverall = pd.concat(nullOverall, ignore_index=True)
        
        ## PROBABILITY of being Nearest Neighbor
        # everything per-area
        anut.catplot_proportions(DF = DF,
                                 NULLDF=NULLareas,
                                countgroupby=['Group', 'Type', 'Area', 'NeighborTYPE'],
                                propgroupby=['Group', 'Type', 'Area'],
                                row = 'Type', col = 'Area',
                                x = 'NeighborTYPE',
                                hue = 'Group',
                                col_order=['V1', 'AM/PM', 'A/RL/AL', 'LM'],
                                row_order=['V','A','M'],
                                order=['V','A','M'],
                                show=False,
                                plotname='byAreaNeighbors',
                                savedir=self.SAVEDIR)

        # overall across areas together
        anut.catplot_proportions(DF = DF,
                                 NULLDF=NULLoverall,
                                countgroupby=['Group', 'Type', 'NeighborTYPE'],
                                propgroupby=['Group', 'Type'],
                                row = 'Type', 
                                x = 'NeighborTYPE',
                                hue = 'Group',
                                row_order=['V','A','M'],
                                order=['V','A','M'],
                                show=False,
                                plotname='overallNeighbors',
                                savedir=self.SAVEDIR)
        
        ## DISTANCE to closest neighbor of different TYPES
        # per area
        anut.catplot_distanc(DF = DISTDF,
                             NULLDF=NULLareas,
                             x =  'NeighborTYPE', y = 'Distance',
                             hue = 'Group', row = 'Type', col = 'Area',
                             row_order=['V','A','M'], order=['V','A','M'],
                             col_order=['V1', 'AM/PM', 'A/RL/AL', 'LM'],
                             plotname='byAreaDistances',
                             savedir=self.SAVEDIR)

        # overall across visual cortex
        anut.catplot_distanc(DF = DISTDF,
                             NULLDF=NULLoverall,
                             x =  'NeighborTYPE', y = 'Distance',
                             hue = 'Group', row = 'Type',
                             row_order=['V','A','M'], order=['V','A','M'],
                             plotname='overallDistances',
                             savedir=self.SAVEDIR)

        ## XXX: K-NN (also interesting)

def neighbor_analysis(groupDF:pd.DataFrame, adjM:np.ndarray, 
                      session_neurons:list[tuple[int, int]]):
    '''
    Nearest neighbor analysis and distance to nearest neighbor analysis
    '''
    signeurons:list = groupDF.NeuronID.tolist()
    sigV = groupDF.loc[groupDF['Type'] == 'V', 'NeuronID'].to_list()
    sigA = groupDF.loc[groupDF['Type'] == 'A', 'NeuronID'].to_list()
    sigM = groupDF.loc[groupDF['Type'] == 'M', 'NeuronID'].to_list()
    
    # adjacency matrix with pairwise distances
    # set main diagonal (distance to oneself is 0) to high number
    adjM += np.eye(*adjM.shape) * 2 * adjM.max()
    
    # set all nonsignificant neurons to high number (we ignore nonsignificant neighbors)
    allneuronIDs = np.arange(adjM.shape[0])
    nonsigneurons = np.isin(allneuronIDs, signeurons , invert=True)
    adjM[:, nonsigneurons] = np.full((adjM.shape[0], nonsigneurons.sum()), 2 * adjM.max())
    # 'n' is nonsignificant
    signeurontypes = np.full(adjM.shape[0], 'n', dtype=str)
    signeurontypes[signeurons] = groupDF.Type.to_numpy(dtype=str)
    # prepare output (first full matrix size, will index to only signeurons later)
    nntypes = np.full(adjM.shape[0], 'n', dtype=str)
    nndists = np.full(adjM.shape[0], 2 * adjM.max(), dtype=float)
    vdists = np.full(adjM.shape[0], 2 * adjM.max(), dtype=float)
    adists = np.full(adjM.shape[0], 2 * adjM.max(), dtype=float)
    mdists = np.full(adjM.shape[0], 2 * adjM.max(), dtype=float)
    
    # go over sessions
    for s0, sE in session_neurons:
        sessneurons = np.arange(s0, sE)

        sesssigneurons = np.intersect1d(sessneurons, signeurons) - s0
        sessV = np.intersect1d(sesssigneurons, sigV)
        sessA = np.intersect1d(sesssigneurons, sigA)
        sessM = np.intersect1d(sesssigneurons, sigM)

        adj = adjM[s0:sE, s0:sE, ...]
        # TODO: fix this calculation for per-session (calculate per-session NN) and then broadcast that to the big DF
        # get the nearest neighbor index (in session indices) + s0 to get back to overall neuronID
        nearest_neighbor = np.argmin(adj, axis = 1) + s0
        
        # get nearest neighbor (within session) distances
        nearest_neighbor_dist = np.min(adj, axis = 1)
        nearest_V_neighbor_dist = np.min(adj[:, sessV], axis = 1)
        nearest_A_neighbor_dist = np.min(adj[:, sessA], axis = 1)
        nearest_M_neighbor_dist = np.min(adj[:, sessM], axis = 1)

        # get indices for nearest neighbors for significantly responding neurons
        nnforsigneurons = nearest_neighbor[sesssigneurons]
        
        # nearest neighbor types
        nntypes[sesssigneurons + s0] = signeurontypes[nnforsigneurons]
        # distances
        nndists[sesssigneurons + s0] = nearest_neighbor_dist[sesssigneurons]
        vdists[sesssigneurons + s0] = nearest_V_neighbor_dist[sesssigneurons]
        adists[sesssigneurons + s0] = nearest_A_neighbor_dist[sesssigneurons]
        mdists[sesssigneurons + s0] = nearest_M_neighbor_dist[sesssigneurons]
    
    groupDF.loc[:,'NeighborTYPE'] = nntypes[signeurons]
    groupDF.loc[:, 'NeighborDIST'] = nndists[signeurons]
    groupDF.loc[:, 'Vneighbor'] = vdists[signeurons]
    groupDF.loc[:, 'Aneighbor'] = adists[signeurons]
    groupDF.loc[:, 'Mneighbor'] = mdists[signeurons]
    
    return groupDF

def null_distributions(gDF:pd.DataFrame, adj:np.ndarray, 
                       session_neurons:dict[list[tuple[int,int]]],
                       perArea:bool = True,
                       niter:int = 1000
                       )->pd.DataFrame:
    ''''
    “Null distributions are generated by shuffling cell-type labels within each imaging session, 
    preserving session-specific spatial layout and cell-type count
    '''
    assert len(gDF.Group.unique()) == 1
    if os.path.exists(nulldistpath := os.path.join(PYDATA, 
                                                   f'{gDF.Group.unique()}_neighbornull_area{perArea}.csv')):
        return pd.read_csv(nulldistpath, index_col=0)
    
    cpus = cpu_count()
    # shuffle niter times
    # parallel (fast)
    res = Parallel(n_jobs=cpus, backend='threading')(delayed(null_dist_worker)(gDF.copy(), adj.copy(), perArea, session_neurons.copy()) 
                                                     for _ in tqdm(range(niter)))
    # single-core (debugging)
    # res = [null_dist_worker(gDF.copy(), adj.copy(), perArea, session_neurons.copy()) for _ in range(niter)]
    NULLdist:pd.DataFrame = pd.concat(res, ignore_index=True)
    NULLdist.to_csv(os.path.join(PYDATA, f'{gDF.Group.unique()}_neighbornull_area{perArea}.csv'))

    return NULLdist

def null_dist_worker(gDF:pd.DataFrame, adj:np.ndarray, perArea:bool, 
                     session_neurons:dict[list[tuple[int,int]]]):
    ''''
    shuffle the labels to create null distribution
    '''
    nullgDF = gDF.copy()
    # # shuffles neuron types across all sessions
    # if not perArea:
    #     nullgDF.loc[:,'Type'] = nullgDF['Type'].sample(frac=1).to_numpy()
    # else:
    # shuffle only per-session
    nullgDF.loc[:,'Session'] = np.full(nullgDF.shape[0], -1)
    for indS, (s0, sE) in enumerate(session_neurons):
        sessneurons = np.arange(s0, sE)
        sesssigneurons = np.intersect1d(sessneurons, nullgDF.NeuronID.to_numpy())
        nullgDF.loc[np.isin(nullgDF['NeuronID'], sesssigneurons), 'Session'] = indS
    
    assert -1 not in nullgDF.Session

    nullgDF.loc[:,'Type'] = nullgDF.groupby(['Session'])['Type'].sample(frac = 1).to_numpy()

    DF = neighbor_analysis(nullgDF, adj, session_neurons)
    DISTDF = pd.melt(DF, id_vars=['Group', 'Area', 'Type', 'NeuronID'], 
                        value_vars=['Vneighbor','Aneighbor','Mneighbor'],
                        var_name='NNtype', value_name='Distance')
    if perArea:
        props = anut.get_proportionsDF(DF, 
                                    countgroupby=['Group', 'Type', 'Area', 'NeighborTYPE'],
                                    propgroupby=['Group', 'Type', 'Area'])
        
        dists = anut.get_distancesDF(DISTDF, groupbyVAR=['Group', 'Type', 'Area','NNtype'], 
                                     valueVAR='Distance')
    else:
        props = anut.get_proportionsDF(DF, 
                                    countgroupby=['Group', 'Type', 'NeighborTYPE'],
                                    propgroupby=['Group', 'Type'])
        
        dists = anut.get_distancesDF(DISTDF, groupbyVAR=['Group', 'Type', 'NNtype'], 
                                     valueVAR='Distance')
    dists.loc[:, 'NeighborTYPE'] = dists['NNtype'].transform(lambda x : x[0])
    dists['NeighborTYPE'] = dists['NeighborTYPE'].astype(str)
    dists.drop(columns = ['NNtype'], inplace=True)
    out  = pd.merge(props, dists)
    return out


if __name__ == '__main__':
    ### Venn diagram of neuron classes in in the 4 different regions
    NGDF, ARdict, SESSdict = by_areas_VENN(svg=True, pre_post='pre')

    # Architecture analysis
    # Arch = Architecture(NGDF, ARdict, SESSdict)
    # Arch.spatial_distribution()
    # Arch.neighbors()

    ### Timeseries plots for neurons from different regions
    # by_areas_TSPLOT(GROUP_type = 'modulated', add_CASCADE=False)
    # by_areas_TSPLOT(GROUP_type = 'modality_specific', add_CASCADE=False)
    
    # this is the one that makes sense for V /A /M neuron groups
    QuantDF = by_areas_TSPLOT(GROUP_type = 'all', pre_post='pre',
                              svg=True)
    Quantification(QuantDF, svg=True)
    
    ### TOTAL timeseries plot and quantification
    # QuantDF = by_areas_TSPLOT(GROUP_type = 'TOTAL', add_CASCADE=False, svg=False)

    # # Recorded neurons plot
    # recordedNeurons(svg=True)

    