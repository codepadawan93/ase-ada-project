import numpy as np
import pandas as pd
from utils.utils import Utils
from utils.graphics import Graphics

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
    def __init__(self, X, var_name):
        self.graphics = Graphics
        self.var_name = var_name
        self.X = X[var_name].values
        print(var_name)
        self.n = X.shape[0]  # number of observation
        self.m = X.shape[1]  # number of variables
        Utils.replace_na(self.X)
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.correlation_fac = self.r_tab = self.correlation_fac_tab = self.C = None
        self.calculate()

    # Resets the state of the PCA object
    def reset(self):
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.correlation_fac = self.r_tab = self.correlation_fac_tab = self.C = None
        return self

    def calculate(self):

        #compute the corr matrix
        self.R = np.corrcoef(self.X, rowvar=False)
        self.r_tab = pd.DataFrame(data=self.R, columns=self.var_name, index=self.var_name)

        # compute eigenvectors and eigenvalues
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.R)

        # Sort the eigenvalues and the corresponding eigenvectors in descending order
        reversed_eigenvalues = [k for k in reversed(np.argsort(self.eigenvalues))]

        self.alpha = self.eigenvalues[reversed_eigenvalues]
        self.a = self.eigenvectors[:, reversed_eigenvalues]

        for i in range(len(self.alpha)):
            minimum = np.min(self.a[:, i])
            maximum = np.max(self.a[:, i])
            if np.abs(minimum) > np.abs(maximum):
                self.a[:, i] = -self.a[:, i]

        # Compute the correlation factors
        self.correlation_fac = self.a * np.sqrt(self.alpha)

        self.correlation_fac_tab = pd.DataFrame(self.correlation_fac, index=self.var_name, columns=["C" + str(i) for i in range(1, self.m)])

        # Compute the principal components on standardized X and return
        avg_var = np.mean(self.X, axis=0)
        std_deviation = np.std(self.X, axis=0)
        X_std = (self.X - avg_var) / std_deviation
        self.C = X_std @ self.a
        return self

    # Return the correlation matrix of the initial (causal) variables
    def get_correlation(self):
        return self.R

    # Return the eigenvalues of the correlation matrix
    def get_eigenvalues(self):
        return self.alpha

    # Return the eigenvectors of the correlation matrix
    def get_eigenvectors(self):
        return self.a

    def get_correlation_factors(self):
        return self.correlation_fac

    def get_principal_components(self):
        return self.C

    def get_results(self):
        return self.R, self.alpha, self.a, self.correlation_fac, self.C

    def visualise(self):
        # Show the correlogram
        self.graphics.correlogram(self.r_tab)

        # Show factors correlogram
        self.graphics.correlogram(self.correlation_fac_tab, "Correlogram of the factors")
        self.graphics.corrCircle(self.correlation_fac_tab, 0, 1)
        return self

    def show(self):
        self.graphics.show()