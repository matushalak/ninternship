# @matushalak
# group_condition.rois.pkl contains a DATAFRAME with the XYlocation and area name for each neuron
# AllenBrainAtlasOverlay.mat contains the areaNames (MATLAB INDEXING!!! replace 0 with NaN) 
# and mask onto which XY location is mapped!
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
import skimage as ski

