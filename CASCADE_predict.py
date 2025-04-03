#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@matushalak modified for SPSIG format for Huub Terra's Dark-rearing project, assumed to be working inside a CASCADE venv
only included here to showcase approach, to use this script, use the same script within an environment setup from
https://github.com/matushalak/Cascade (fork of the CASCADE repository)

Script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""
import os, sys, glob, re, gc

import numpy as np
import scipy.io.matlab as sio_mtl
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
from matplotlib import pyplot as plt
from tkinter import filedialog
from mat73 import loadmat as hdf_loadmat

"""

Main function that runs CASCADE on a file and saves CASCADE output in the correct folders

"""
def main(spsig_file:str,
         PLOT:bool = False):
    """

    Load dF/F traces, define frame rate and plot example traces

    """
    print(f'Estimating spikes from {spsig_file}')

    traces, frame_rate = load_neurons_x_time(spsig_file)
    print('Number of neurons in dataset:', traces.shape[0])
    print('Number of timepoints in dataset:', traces.shape[1])

    """

    Select pretrained model and apply to dF/F data

    """
    # these seem relevant for out data
    # 'GC8m_EXC_15Hz_smoothing100ms_high_noise'
    # 'GC8_EXC_15Hz_smoothing100ms_high_noise'
    model_name = 'GC8m_EXC_15Hz_smoothing100ms_high_noise'
    if model_name not in os.listdir('Pretrained_models'):
        cascade.download_model( model_name,verbose = 1)

    print('Using {} model'.format(model_name))
    # break it up into chunks if more than 100 neurons
    if traces.shape[0] > 100:
        print('Splitting up into batches of 100 neurons to spare computational load')
        spike_prob = np.empty_like(traces)
        hundreds  = traces.shape[0] // 100
        ranges = [(r * 100, r * 100 + 100) for r in range(hundreds)]
        for start, end in ranges:
            spike_prob[start:end,:] = cascade.predict(model_name, traces[start:end, :], padding = 0)
    else:
        print('Processing all at once (<= 100 neurons)')
        spike_prob = cascade.predict( model_name, traces, padding = 0)

    """

    Save predictions to disk

    """
    folder = os.path.dirname(spsig_file)
    save_path = os.path.join(folder, 'full_prediction_'+os.path.basename(spsig_file))

    # save 
    sio_mtl.savemat(save_path+'.mat', {'spike_prob':spike_prob})

    """

    Plot example predictions

    """
    if not PLOT:
        del spike_prob, traces
    else:
        print('Feel free to explore the dataset in the terminal with: \nneuron_indices = np.random.randint(traces.shape[0], size=10)\nplot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)\nplt.show()')
        neuron_indices = np.random.randint(traces.shape[0], size=10)
        plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)
        plt.show()

        breakpoint()


"""

Define function to load dF/F traces from disk, SPSIG specific

"""
def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""
    # traces should be 2d array with shape (neurons, nr_timepoints)
    try:
        # old version < v7.3 mat file
        SPSIG_dict = sio_mtl.loadmat(file_path, simplify_cells = True, variable_names = 'sigCorrected')
        print('Old matlab file')

    except NotImplementedError:
        # Load >= v7.3 .mat hdf5 file
        print('New_matlabfile')
        SPSIG_dict = hdf_loadmat(file_path, use_attrdict=True, only_include='sigCorrected')

    traces : np.ndarray = SPSIG_dict['sigCorrected']
    frame_rate = 15.45703#spsig.freq
    if traces.shape[1] < traces.shape[0]:
       traces = traces.T # to match (neurons, nr_timepoints) expected shape
    return traces, frame_rate

def get_SPSIG_files(root : bool = False, overwrite_existing:bool = False) -> list:
    if not root:
        root = filedialog.askdirectory()
    spsigs = '**/*_SPSIG.mat'
    
    SPSIG_files = []
    
    for spsig_path in glob.glob(os.path.join(root,spsigs), recursive = True):
        # Regex breakdown:
        # .*/                => match any characters ending with a slash
        # (g[12])            => capture group: "g1" or "g2"
        # /                  => literal slash
        # ([^/]+)            => capture "Name" (any characters except slash)
        # /                  => literal slash
        # (\d{8})            => capture "date" in the format YYYYMMDD (8 digits)
        # /                  => literal slash
        # (Bar_Tone_LR(?:2)?) => capture "Bar_Tone_LR" optionally followed by a 2
        # /.*                => followed by a slash and the rest of the path
        group_name_date = r'.*/(g[12])/([^/]+)/(\d{8})/(Bar_Tone_LR(?:2)?)/.*'
        re_match = re.match(group_name_date, spsig_path)
        if re_match is None:
            continue
        
        if any(file.startswith('full_prediction_') for file in os.listdir(os.path.dirname(spsig_path))) and not overwrite_existing:
            continue
        # Add spsig file to extract spikes from
        assert re_match is not None, f'something wrong with: {spsig_path}'
        SPSIG_files.append(spsig_path)
    
    return SPSIG_files


if __name__ == '__main__':
    SPSIG_files = get_SPSIG_files(overwrite_existing=True)#root='/Volumes/my_SSD/NiNdata/data', overwrite_existing=False)
    # align all files
    n_files = len(SPSIG_files)
    if n_files > 0:
        print('Found {} ..._SPSIG.mat files, starting to extract spikes!'.format(n_files))
        for f in SPSIG_files: print(f)

        for i, file in enumerate(SPSIG_files):
            main(file)
            print('Spike estimation from {} completed!'.format(file))
            _ = gc.collect()