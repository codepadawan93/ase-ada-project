import numpy as np
import pandas as pd


class PCA:

    def __init__(self, X):
        self.X = X
        self.R = np.corrcoef(self.X, rowvar=False)
        self.m, self.n = n = X.shape[0], X.shape[1]
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.R_x_c = self.r_tab = self.rxc_tab = self.C = None

        # Resets the state of the PCA object

    def _reset(self):
        self.R = self.eigenvalues = self.eigenvectors = self.alpha = self.a = self.R_x_c = self.C = None
        return self

    def _calculate(self):

        #compute the corr matrix
        self.R = np.corrcoef(self.X, rowvar=False)
        self.r_tab = pd.DataFrame(data=self.R, columns=self.var_name, index=self.var_name)

        # compute eigenvectors and eigenvalues
        eigenVal = np.linalg.eig(self.R)
        eigenVect = np.linalg.eig(self.R)


        # Compute the correlation factors
        self.correlation_fac = self.a * np.sqrt(self.alpha)

        # Sort the eigenvalues and the corresponding eigenvectors in descending order
        reversed_eigenvalues = [k for k in reversed(np.argsort(self.eigenvalues))]
        self.alpha = self.eigenvalues[reversed_eigenvalues]
        self.a = self.eigenvectors[:, reversed_eigenvalues]

        # Compute the principal components on standardized X and return
        avg_var = np.mean(self.X, axis=0)
        std_deviation = np.std(self.X, axis=0)
        X_std = (self.X - avg_var) / std_deviation
        self.C = X_std @ self.a
        return self.C

            # Return the correlation matrix of the initial (causal) variables

        def getCorrelation(self):
            return self.R

            # Return the eigenvalues of the correlation matrix

        def getEigenValues(self):
            return self.alpha

            # Return the eigenvectors of the correlation matrix

        def getEigenVectors(self):
            return self.a

        def getCorrelationFactors(self):
            return self.correlation_fac

        def getPrincipalComponents(self):
            return self.C




