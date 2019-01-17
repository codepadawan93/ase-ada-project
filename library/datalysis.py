import pandas as pd
from utils.utils import Utils
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
        self.results = []

        self.pca_module = None
        self.efa_module = None
        self.cca_module = None
        self.hca_module = None
        self.lda_module = None

    # Reads a pandas dataframe into memory from a csv file
    def read_file(self, filename, index_col=0, na_vals=':'):
        table = pd.read_csv(filename, index_col=index_col, na_values=na_vals)
        self.obs_name = table.index
        self.var_name = table.columns[1:]
        try:
            self.data_frame = table
        except ValueError:
            Utils.log(TAG, ValueError)
        return self

    def read_multiple(self, file1, file2, f1_index_col=0, f2_index_col=0):
        self.t1 = pd.read_csv(file1, index_col=f1_index_col)
        self.t2 = pd.read_csv(file2, index_col=f2_index_col)
        return self

    # Reads a hardcoded or already-existing pandas dataframe into memory
    def read_data(self, data):
        try:
            self.data_frame = pd.DataFrame(data.values, data.index, data.column_labels)
        except ValueError:
            Utils.log(TAG, ValueError)
        return self

    # Obtain PCA of data using the PCA class
    def run_pca(self):
        self.pca_module = pca.PCA(self.data_frame, self.var_name)
        self.results = self.pca_module.get_results()
        return self

    # Perform EFA on data using the EFA class
    def run_efa(self):
        R, alpha, a, rxc, C = pca.PCA(self.data_frame, self.var_name).get_results()
        self.efa_module = efa.EFA(self.data_frame, self.obs_name, self.var_name)
        self.efa_module.explore(C, alpha, R) \
            .bartlett_test() \
            .kmo() \
            .analyse()
        self.results = self.efa_module.get_results()
        return self

    # x mark, y mark the columns where we split the dataset in two
    def run_cca(self, x_mark, y_mark):
        self.cca_module = cca.CCA(self.data_frame, self.var_name, x_mark, y_mark)
        self.cca_module.fit() \
            .compute_canonical_correlations() \
            .bartlett_wilks()
        return self

    #
    def run_hca(self, method=0, metric=0, partition_type="optimal", no_groups=None):
        self.hca_module = hca.HCA(self.data_frame)
        self.hca_module.classify(method, metric)
        if partition_type == "optimal":
            self.hca_module.optimal_partition()
        elif partition_type == "arbitrary" and no_groups is not None:
            self.hca_module.arbitrary_partition(no_groups)
        self.results = self.hca_module.get_results()
        return self

    def run_lda(self, categorical_mark=None, predictor_mark=None, predictor_var=None):
        self.lda_module = lda.LDA(self.t1, self.t2, categorical_mark, predictor_mark, predictor_var)
        self.results = self.lda_module.fit().apply().get_results()
        return self

    def get_results(self):
        return self.results

    def visualise(self):
        if(not self.pca_module is None):
            self.pca_module.visualise().show()
        if (not self.efa_module is None):
            self.efa_module.visualise().show()
        if (not self.cca_module is None):
            self.cca_module.visualise().show()
        if (not self.hca_module is None):
            self.hca_module.visualise().show()
        if (not self.lda_module is None):
            self.lda_module.visualise().show()
        return self

    # Writes a SPSS-style report in plain text at the specified location,
    # running all available tests on the provided data
    def put_report(self, filename):
        try:
            pass
        except ValueError:
            pass
        return None
