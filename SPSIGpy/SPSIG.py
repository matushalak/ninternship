from mat73 import loadmat
from tkinter import filedialog
import os

current_dir = filedialog.askdirectory()
matfiles = [os.path.join(current_dir,file) 
            for file in os.listdir(current_dir) if 'SPSIG' in file and '._' not in file]

f1 = loadmat(matfiles[0])

breakpoint()

class SPSIG ():
    pass
