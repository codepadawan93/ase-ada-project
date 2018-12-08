import pandas as pd
from . import utils as u
from . import pca
from . import efa

'''
Main wrapper for the library functionality

Encapsulates the logic for reading and writing to files, pandas dataframes, PCA, 
factor analysis, canonical correlation, discriminant analysis, cluster analysis, 
correspondence analysis and data visualisation.

'''

# Tag to identify where errors came from
TAG = "Datalysis"

class Datalysis:

    def __init__(self):
        # Start with a clean slate

        self.data_frame = None
        self.results = None

        self.pca_module = None
        self.efa_module = None
        self.cca_module = None
        self.discr_module = None
        self.cluster_module = None
        self.corresp_module = None

    # Reads a pandas dataframe into memory from a csv file
    def read_file(self, filename, index_col):
        try:
            self.data_frame = pd.read_csv(filename, index_col=index_col)
        except ValueError:
            u.log(TAG, ValueError)
        return self

    # Reads a hardcoded or already-existing pandas dataframe into memory
    def read_data(self, data):
        try:
            self.data_frame = pd.DataFrame(data.values, data.index, data.column_labels)
        except ValueError:
            u.log(TAG, ValueError)
        return self

    # Obtain PCA of data using the PCA class
    def run_pca(self):
        self.pca_module = pca.PCA(self.data_frame)
        return self

    # Perform EFA on data using the EFA class
    def run_efa(self):
        self.efa_module = efa.EFA(self.data_frame)
        return self

    # Writes a SPSS-style report in plain text at the specified location,
    # running all available tests on the provided data
    def put_report(self, filename):
        try:
            pass
        except ValueError:
            pass
        return None
