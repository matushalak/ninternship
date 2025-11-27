from src.AUDVIS import AUDVIS, Behavior, load_in_data

#%% general tester
avs:list[AUDVIS] = load_in_data('both')

def print_sessions(avs):
    for av in avs:
        with open('sessions_overview.txt', 'a') as f:
            f.write(f'{av.NAME}\n')
        
            for s, sdict in av.sessions.items():
                f.write(f'{sdict['session']}\n')

movements = ['whisker', 'running', 'pupil']
for av in avs:
    for m in movements:
        av.nantrials(av.zsig, movement=m, 
                     behWindow = (av.TRIAL[0], av.TRIAL[1]+5), # incl offset
                     plot=True
                     )