import numpy as np

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
    def __init__(self, X):
        self.X = X
        self.R, self.eigenvalues, self.eigenvectors, self.alpha, self.a, self.R_x_c, self.C = None

    # Resets the state of the PCA object
    def _reset(self):
        self.X, self.R, self.eigenvalues, self.eigenvectors, self.alpha, self.a, self.R_x_c, self.C = None

    # Performs the algorithm up to the specified point
    def _calculate(self, up_to='principal_components'):
        # Compute the correlation matrix and return if only that is needed
        self.R = np.corrcoef(self.X, rowvar=False)

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

    # Return the correlation matrix of the initial (causal) variables
    def get_correlation(self):
        return self._calculate('correlation')

    # Return the eigenvalues of the correlation matrix
    def get_eigenvalues(self):
        return self._calculate('eigenvalues')

    # Return the eigenvectors of the correlation matrix
    def get_eigenvectors(self):
        return self._calculate('eigenvectors')

    def get_correlation_factors(self):
        return self._calculate('correlation_factors')

    def get_principal_components(self):
        return self._calculate('principal_components')

    def get_data(self, data='*'):
        bundle = []
        if data == 'correlation' or data == '*':
            bundle.append({'correlation', self.get_correlation()})
        elif data == 'eigenvalues' or data == '*':
            bundle.append({'eigenvalues', self.get_eigenvalues()})
        elif data == 'eigenvectors' or data == '*':
            bundle.append({'eigenvectors', self.get_eigenvectors()})
        elif data == 'correlation_factors' or data == '*':
            bundle.append({'correlation_factors', self.get_correlation_factors()})
        elif data == 'principal_components' or data == '*':
            bundle.append({'principal_components', self.get_principal_components()})

        return bundle
