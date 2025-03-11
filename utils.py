# @matushalak
import os
import re
from glob import glob
import pandas as pd
from tkinter import filedialog
from SPSIG import SPSIG
from collections import defaultdict


def make_folders():
    '''
    main folder organization based on Huub's Lab book
    '''
    labbook = pd.read_csv('labbook.csv').dropna()

    lab = labbook.sort_values(by = ['Name', 'Date (YYYYMMDD)'])

    lab.to_csv('better_lab.csv')

    for i, row in enumerate(lab.itertuples()):
        name, date, protocol, zoom, loc = row[1:]
        date = str(round(date))
        path = os.path.join('data', name, date, protocol)
        os.makedirs(path, exist_ok = True)


# TODO: generalize for more groups / different experiment structure etc.
def group_condition_key() -> tuple[dict, dict]:
    'Returns 2 dictionaries with'
    root = filedialog.askdirectory()
    spsigs = '**/*_SPSIG.mat'
    
    # for now we only care about SPSIG_Res files
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
        group, name, date, bartone = re_match.groups()

        animals[group][name][int(date)].append(res_file)# path

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