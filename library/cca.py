import numpy as np
import sklearn.cross_decomposition as skl
'''
    Class for Canonical Correlation Analysis
'''


class CCA:
    def __init__(self, x_data, y_data):
        self.x_data, self.y_data = x_data, y_data
        self.model, self.x_scores, self.y_scores, self._raw_correlations, self.correlations= None

    # Run sklearn's CCA analysis on the arrays
    def _fit(self, x_data, y_data):
        # Build the model
        n, p = np.shape(x_data)
        q = np.shape(y_data)[1]
        self.m = min(p, q)
        self.model = skl.CCA(n_components=self.m)
        self.model.fit(x_data, y_data)

        # Set the results
        # Canonical scores
        self.x_scores = self.model.x_scores_ # z
        self.y_scores = self.model.y_scores_ # u

    # Compute the canonical correlations
    def _compute_canonical_correlations(self):
        self._raw_correlations = \
            [np.corrcoef(self.x_scores[:, i], self.y_scores[:, i], rowvar=False)[0, 1] for i in range(self.m)]
        self.correlations = np.array(self._raw_correlations)

    # Rest...
    # chi2_computed, chi2_estimated = utils.bartlett_wilks(r, n, p, q, m)
    # print("Chi square computed : ", chi2_computed)
    # print("Chi square test: ", chi2_estimated)
    #
    # chi2_computed_table = pd.DataFrame(chi2_computed, index=['r' + str(i) for i in range(1, m + 1)],
    #                                    columns=['chi2_computed'])
    # print("Chi square computed table: ", chi2_computed_table)
    # visual.correlogram(chi2_computed_table, "Bartlett-Wilks significance test", 0)
    #
    # chi2_estimated_table = pd.DataFrame(chi2_estimated, index=['r' + str(i) for i in range(1, m + 1)],
    #                                     columns=['chi2_estimated'])
    # print("Chi square estimated table: ", chi2_estimated_table)
    # visual.correlogram(chi2_estimated_table, "Bartlett-Wilks significance test", 0)
