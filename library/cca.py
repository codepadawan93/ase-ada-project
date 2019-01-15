import numpy as np
import sklearn.cross_decomposition as skl
from . import utils
from . import efa

'''
    Class for Canonical Correlation Analysis
'''


class CCA:
    def __init__(self, x_data, y_data):
        self.x_data, self.y_data = x_data, y_data
        self.model = self.x_scores = self.y_scores = self._raw_correlations = self.correlations = self.m = self.n = self.p = self.q = self.n = None

    def _reset(self):
        self.model = self.x_scores = self.y_scores = self._raw_correlations = self.correlations = self.m = self.n = self.p = self.q = self.n = None

    # Run sklearn's CCA analysis on the arrays
    def fit(self, x_data, y_data):
        # Build the model
        self.n, self.p = np.shape(x_data)
        self.q = np.shape(y_data)[1]
        self.m = min(self.p, self.q)
        self.model = skl.CCA(n_components=self.m)
        self.model.fit(x_data, y_data)

        # Set the results
        # Canonical scores
        self.x_scores = self.model.x_scores_ # z
        self.y_scores = self.model.y_scores_ # u
        return self

    # Compute the canonical correlations
    def compute_canonical_correlations(self):
        self._raw_correlations = \
            [np.corrcoef(self.x_scores[:, i], self.y_scores[:, i], rowvar=False)[0, 1] for i in range(self.m)]
        self.correlations = np.array(self._raw_correlations)
        return self

    def bartlett_wilks(self):
        efa_model = efa.EFA(self.r)\
            .bartlett_wilks(self.n, self.q, self.p, self.m)
        self.chi2_computed, self.chi2_estimated = efa_model.chi2_computed, efa_model.chi2_estimated
        self.chi2_computed_table = utils.get_data_frame(self.chi2_computed, index=['r' + str(i) for i in range(1, self.m + 1)],
                                            cols=['chi2_computed'])
        return self
