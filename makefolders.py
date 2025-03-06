import os
import pandas as pd

labbook = pd.read_csv('labbook.csv').dropna()

lab = labbook.sort_values(by = ['Name', 'Date (YYYYMMDD)'])

lab.to_csv('better_lab.csv')

# for i, row in enumerate(lab.itertuples()):
#     name, date, protocol, zoom, loc = row[1:]
#     date = str(round(date))
#     path = os.path.join('data', name, date, protocol)
#     os.makedirs(path, exist_ok = True)
