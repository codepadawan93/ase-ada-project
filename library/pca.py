import numpy as np
import pandas as pd
from . import graphics

'''
    Class for solving a Principal Components Analysis (PCA) problem
    The class is able to calculate the following metrics:
        - R, the correlation matrix
        - alpha, the eigenvalues of the correlation matrix
        - a, the eigenvectors of the correlation matrix
        - Rxc, the correlation factors
        - C, the principal components
'''


class PCA:

    # Build object around an initial value matrix
    def __init__(self, X, var_name):
        self.var_name = var_name
        self.X = X
        # number of observations, umber of variables
        self.m, self.n = n = X.shape[0], X.shape[1]
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.R_x_c = self.r_tab = self.rxc_tab = self.C = None

    # Resets the state of the PCA object
    def _reset(self):
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.R_x_c = self.C = None
        return self

    # Performs the algorithm up to the specified point
    def _calculate(self, up_to='principal_components'):

        # Compute the correlation matrix and return if only that is needed
        self.R = np.corrcoef(self.X, rowvar=False)
        self.r_tab = pd.DataFrame(data=self.R, columns=self.var_name, index=self.var_name)

        if up_to == 'correlation':
            return self.R

        # Find the eigenvalues and eigenvectors from the corellation matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.R)

        # Sort the eigenvalues and the corresponding eigenvectors in descending order
        reversed_eigenvalues = [k for k in reversed(np.argsort(self.eigenvalues))]

        self.alpha = self.eigenvalues[reversed_eigenvalues]
        self.a = self.eigenvectors[:, reversed_eigenvalues]

        # TODO :: figure out what this does.
        # and return if a or alpha is what is asked
        for i in range(len(self.alpha)):
            minimum = np.min(self.a[:, i])
            maximum = np.max(self.a[:, i])
            if np.abs(minimum) > np.abs(maximum):
                self.a[:, i] = -self.a[:, i]

        if up_to == 'eigenvalues':
            return self.a

        if up_to == 'eigenvectors':
            return self.alpha

        # Compute the correlation factors and return if that is what is asked
        self.R_x_c = self.a * np.sqrt(self.alpha)

        if up_to == 'correlation_factors':
            return self.R_x_c

        self.rxc_tab = pd.DataFrame(self.R_x_c, index=self.var_name, columns=["C"+str(i+1) for i in range(self.m)])

        # Compute the principal components on standardized X and return
        avg_var = np.mean(self.X, axis=0)
        std_deviation = np.std(self.X, axis=0)
        X_std = (self.X - avg_var) / std_deviation
        self.C = X_std @ self.a

        if up_to == 'principal_components':
            return self.C

        # Deal with invalid input
        raise Exception('Invalid argument supplied to PCA::_calculate().')

    def setX(self, X):
        self._reset()
        self.X = X
        return self

    # Return the correlation matrix of the initial (causal) variables
    def get_correlation(self):
        self._calculate('correlation')
        return self

    # Return the eigenvalues of the correlation matrix
    def get_eigenvalues(self):
        self._calculate('eigenvalues')
        return self

    # Return the eigenvectors of the correlation matrix
    def get_eigenvectors(self):
        self._calculate('eigenvectors')
        return self

    def get_correlation_factors(self):
        self._calculate('correlation_factors')
        return self

    def get_principal_components(self):
        self._calculate('principal_components')
        return self

    def visualise(self):
        if self.X == None :
            return

        # Show the correlogram
        graphics.correlogram(self.r_tab)

        # Show factors correlogram
        graphics.correlogram(self.rxc_tab, "Correlogram of the factors")
        graphics.corrCircles(self.rxc_tab, 0, 1)