import os

# Get absolute path to the project root (assuming __init__.py is inside ninternship/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

PYDATA = os.path.join(PROJECT_ROOT, 'pydata')
PLOTSDIR = os.path.join(PROJECT_ROOT, 'plots')
MIPLOTSDIR = os.path.join(PROJECT_ROOT, 'MIplots')