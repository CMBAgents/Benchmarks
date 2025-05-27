import os
from benchmarks.utils import download_file

DATA_DIR = os.path.join(os.path.dirname(__file__),'..', 'data')
N0FILE = os.path.join(DATA_DIR, 'N0.csv')
os.makedirs(DATA_DIR, exist_ok=True)
if  os.path.isfile(N0FILE):
    pass
else:
    download_file('https://github.com/CMBAgents/Benchmarks/releases/download/v1.0.0-alpha/N0.csv', N0FILE)
