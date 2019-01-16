import pandas as pd
from . import utils as u
from . import pca
from . import efa
from . import cca
from . import hca
from . import lda

'''
Main wrapper for the library functionality

Encapsulates the logic for reading and writing to files, pandas dataframes, PCA, 
factor analysis, canonical correlation, discriminant analysis, cluster analysis, 
correspondence analysis and data visualisation.

'''

# Tag to identify where errors came from
TAG = "package:Datalysis"

class Datalysis:

    def __init__(self):
        # Start with a clean slate

        self.data_frame = None
        self.results = None

        self.pca_module = None
        self.efa_module = None
        self.cca_module = None
        self.hca_module = None
        self.lda_module = None

    # Reads a pandas dataframe into memory from a csv file
    def read_file(self, filename, index_col=1, na_vals=':'):
        table = pd.read_csv(filename, index_col=index_col, na_values=na_vals)
        self.obs_name = table.index
        self.var_name = table.columns[1:]
        X = table[self.var_name].values
        u.Utils.replace_na(X)
        try:
            self.data_frame = X
        except ValueError:
            u.Utils.log(TAG, ValueError)
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
        self.results = pca.PCA(self.data_frame, self.var_name).get_principal_components()
        return self

    # Perform EFA on data using the EFA class
    def run_efa(self):
        pca_results = pca.PCA(self.data_frame).get_principal_components().results
        self.efa_module = efa.EFA(self.data_frame)
        self.efa_module \
            .explore(pca_results.c, pca_results.alpha, pca_results.R) \
            .bartlett_test() \
            .kmo() \
            .analyse();
        self.results = self.efa_module
        return self

    def run_cca(self):
        self.cca_module = cca.CCA(self.data_frame)
        self.results = self.cca_module
        return self

    def run_hca(self):
        self.hca_module = hca.HCA(self.data_frame)
        self.results = self.hca_module
        return self

    def run_lda(self):
        self.lda_module = lda.LDA(self.data_frame)
        self.results = self.lda_module
        return self

    def visualise(self):
        self.pca_module.visualise()
        self.efa_module.visualise()
        self.cca_module.visualise()
        self.hca_module.visualise()
        self.lda_module.visualise()

    # Writes a SPSS-style report in plain text at the specified location,
    # running all available tests on the provided data
    def put_report(self, filename):
        try:
            pass
        except ValueError:
            pass
        return None
