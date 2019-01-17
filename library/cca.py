import numpy as np
import pandas as pd
import sklearn.cross_decomposition as skl
import scipy.stats as sts
from utils.graphics import Graphics

'''
    Class for Canonical Correlation Analysis
'''

class CCA:
    def __init__(self, X, var_name, x_mark, y_mark):
        # Partition the data set
        self.X = X
        self.x_columns = var_name[:x_mark]
        self.y_columns = var_name[y_mark:]
        self.x_data = X[self.x_columns].values
        self.y_data = X[self.y_columns].values
        self.graphics = Graphics
        # Null out everything else
        self.model = self.x_scores = self.y_scores = self.raw_correlations = self.correlations = self.m = self.n = self.p = self.q = self.n = None

    def reset(self):
        self.model = self.x_scores = self.y_scores = self.raw_correlations = self.correlations = self.m = self.n = self.p = self.q = self.n = None

    # Run sklearn's CCA analysis on the arrays
    def fit(self):
        # Set the dimensions, or shapes
        self.n, self.p = np.shape(self.x_data)
        self.q = np.shape(self.y_data)[1]
        self.m = min(self.p, self.q)

        # Build the model
        self.model = skl.CCA(n_components=self.m)
        self.model.fit(self.x_data, self.y_data)

        # Canonical scores
        self.x_scores = self.model.x_scores_ # z
        self.y_scores = self.model.y_scores_ # u
        return self

    # Compute the canonical correlations
    def compute_canonical_correlations(self):
        self.raw_correlations = \
            [np.corrcoef(self.x_scores[:, i], self.y_scores[:, i], rowvar=False)[0, 1] for i in range(self.m)]
        self.correlations = np.array(self.raw_correlations)
        return self

    def bartlett_wilks(self):
        self.z = self.model.x_scores_
        self.u = self.model.y_scores_

        # Compute the canonical correlations
        r_list = [np.corrcoef(self.z[:, i], self.u[:, i], rowvar=False)[0, 1] for i in range(self.m)]
        self.r = np.array(r_list)
        r_inv = np.flipud(self.r)
        l = np.flipud(np.cumprod(1 - r_inv * self.r))
        dof = (self.p - np.arange(self.m)) * (self.q - np.arange(self.m))
        self.chi2_computed = (-self.n + 1 + (self.p + self.q + 1) / 2) * np.log(l)
        self.chi2_estimated = 1 - sts.chi2.cdf(self.chi2_computed, dof)
        return self

    def visualise(self):
        # Draw graphs
        chi2_computed_table = pd.DataFrame(self.chi2_computed, index=['r' + str(i) for i in range(1, self.m + 1)],
                                           columns=['chi2_computed'])
        self.graphics.correlogram(chi2_computed_table, "Bartlett-Wilks significance test", 0)
        chi2_estimated_table = pd.DataFrame(self.chi2_estimated, index=['r' + str(i) for i in range(1, self.m + 1)],
                                            columns=['chi2_estimated'])
        self.graphics.correlogram(chi2_estimated_table, "Bartlett-Wilks significance test", 0)
        return self

    def show(self):
        self.graphics.show();
