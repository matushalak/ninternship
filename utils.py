# @matushalak
# contains loader and random functions
import os
import re
import pickle
import matplotlib.pyplot as plt
from glob import glob
from pandas import read_csv
from numpy import array, ndarray, load, unique
from tkinter import filedialog
from collections import defaultdict
from pandas import DataFrame, read_pickle
from time import time

def time_loops(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        nevals, res = func(*args, **kwargs)
        t2 = time()
        duration = t2-t1
        assert isinstance(nevals, (int,float)), 'This wrapper expects first output to be number of evaluations (loops) within function'
        print(f'Function {func.__name__!r} finished in {(duration)} s ({duration/nevals} per loop)')
        return res
    return wrapper

def get_sig_label(p, 
                  sig_label_map = {0: 'n.s.', 
                                   1: '*', 
                                   2: '**', 
                                   3: '***'}):
        if p > 0.05:
            return sig_label_map[0]
        elif p > 0.01:
            return sig_label_map[1]
        elif p > 0.001:
            return sig_label_map[2]
        else:
            return sig_label_map[3]

def add_sig(ax, x1, x2, y, label,
            bar_height=0.02, text_offset=0.01, **kwargs):
    """
    Draw a significance bar between x1 and x2 at height y on `ax`
    and put `label` centred above it.
    Extra kwargs are passed to ax.plot (e.g. color, linewidth).
    """
    ax.plot([x1, x1, x2, x2],
            [y,  y + bar_height, y + bar_height, y],
            **kwargs)
    ax.text((x1 + x2)/2, y + bar_height + text_offset,
            label, ha='center', va='bottom', fontsize=12)

def show_me(*args:ndarray):
    plt.figure()
    plt.plot(args)
    plt.show()
    plt.close()

def progress_bar(current_iteration: int,
                 total_iterations: int,
                 character: str = '🍎'):
    bar_length = 50
    filled_length = round(bar_length * current_iteration / total_iterations)
    # Build the progress bar
    bar = character * filled_length
    no_bar = ' -' * (bar_length - filled_length)
    progress = round((current_iteration / total_iterations) * 100)
    print(bar + no_bar, f'{progress} %', end='\r')
    

def load_audvis_files(group_condition_name:str)->tuple[
                    tuple[dict, dict, DataFrame, ndarray, ndarray, ndarray], 
                    ndarray | None]:
    # indexing
    with open(f'{group_condition_name}_indexing.pkl', 'rb') as indx_f:
        indexing = pickle.load(indx_f)
    
    neur_i = indexing['neuron_index']
    sess_i = indexing['session_index']

    # roi info
    with open(f'{group_condition_name}_rois.pkl', 'rb') as roi_f:
        ROIs = read_pickle(roi_f)

    # numpy arrays, signal, z-scored signal, trials
    endings = ('_sig.npy', '_zsig.npy', '_trials.npy')
    sig, z, trials_all =  [load(group_condition_name+ending) for ending in endings]

    # check if cascade file exists
    if os.path.exists(cascade_file := (group_condition_name + '_CASCADE.npy')):
        Cascade = load(cascade_file)
    else:
        Cascade = None

    return (neur_i,
            sess_i,
            ROIs,
            sig, 
            z,
            trials_all), Cascade

# trials to int and trials to str
def trialMAPS(trialsSTR:ndarray)-> tuple[dict, dict]:
    trialsSTRtoINT = {s:i for i, s in enumerate(unique(trialsSTR))}
    trialsINTtoSTR = {i:s for i, s in enumerate(unique(trialsSTR))}
    return trialsSTRtoINT, trialsINTtoSTR


# necessary for defining the default dict because lambdas can't be pickled
def default_neuron_index():
    return {'overall_id':0,
            'specific_id':('', 0),
            'region':0,
            'trialIDs':array([]),
            'n_trials':0,
            'facemap_corrected':False}

def default_session_index():
    return {'n_neurons':0,
            'trialIDs':array([]),
            'n_trials':0,
            'behavior':None, #instance of Behavior class
            'session':''}


def make_folders():
    '''
    main folder organization based on Huub's Lab book
    '''
    labbook = read_csv('labbook.csv').dropna()

    lab = labbook.sort_values(by = ['Name', 'Date (YYYYMMDD)'])

    lab.to_csv('better_lab.csv')

    for i, row in enumerate(lab.itertuples()):
        name, date, protocol, zoom, loc = row[1:]
        date = str(round(date))
        path = os.path.join('data', name, date, protocol)
        os.makedirs(path, exist_ok = True)


# TODO: generalize for more groups / different experiment structure etc.
def group_condition_key(root:str = False,
                        raw : bool = False) -> tuple[dict, dict]:
    'Returns 2 dictionaries with'
    if not root:
        root = filedialog.askdirectory()
    spsigs = '**/*_SPSIG.mat'
    
    # for now we only care about SPSIG_Res (trial-locked) files
    g1 = {'pre':[],
          'post':[]}
    g2 = {'pre':[],
          'post':[]}

    animals = defaultdict(lambda: defaultdict(lambda:defaultdict(list)))

    for spsig_path in glob(os.path.join(root,spsigs), recursive = True):
        res_file = spsig_path[:-4] + '_Res.mat' if os.path.exists(spsig_path[:-4] + '_Res.mat') else False

        if not res_file:
            continue # skip this
        
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
        assert re_match is not None, f'something wrong with: {spsig_path}'
        group, name, date, bartone = re_match.groups()

        animals[group][name][int(date)].append(res_file if not raw else spsig_path)# path

    group_dicts = (g1, g2)
    group_keys = ('g1', 'g2')
    # breakpoint()
    for group_key, group_dict in zip(group_keys, group_dicts):
        gr_animals = animals[group_key]
        for an in list(gr_animals):
            pre = min(list(gr_animals[an]))
            post = max(list(gr_animals[an]))

            if pre != post:
                group_dict['pre'].extend(gr_animals[an][pre])
                group_dict['post'].extend(gr_animals[an][post])
            else:
                group_dict['pre'].extend(gr_animals[an][pre])

    return g1, g2

if __name__ == "__main__":
    # make_folders()
    g1, g2 = group_condition_key()