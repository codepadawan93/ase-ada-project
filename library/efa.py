import numpy as np
import scipy.stats as sts
import factor_analyzer as fa

'''
    Class for EFA (Exploratory Factor Analysis) 
'''


class EFA:
    def __init__(self, t):
        # Null out all props - mostly to see all of them
        self.t = t
        self.model = fa.FactorAnalyzer()
        self.S = None
        self.q = None
        self.beta = None
        self.common = None
        self.bartlett_test_results = None
        self.kmo = None
        self.message = None
        self.eigenvalues = None
        self.r_inv = None
        self.chi2_computed = None
        self.chi2_estimated = None

    def explore(self, C, alpha, R):
        n = np.shape(C)[0]

        # Compute scores
        self.S = C / np.sqrt(alpha)

        # Compute cosines
        C2 = C * C
        sumObs = np.sum(C2, axis=1)
        self.q = np.transpose(np.transpose(C2) / sumObs)

        # Compute contributions
        self.beta = C2 / (alpha * n)

        # Compute commonalities
        R2 = R * R
        self.common = np.cumsum(R2, axis=1)
        return self

    # Evaluate the information redundancy
    # Bartlett test
    def bartlett_test(self):
        self.bartlett_test_results = fa.calculate_bartlett_sphericity(self.t)
        return self

    def bartlett_wilks(self, n, p, q, m):
        self.r_inv = np.flipud(self.t)
        l = np.flipud(np.cumprod(1 - self.r_inv * self.t))
        dof = (p - np.arange(m)) * (q - np.arange(m))
        self.chi2_computed = (-n + 1 + (p + q + 1) / 2) * np.log(l)
        # Wanted to reimplement chi2 cdf but this implementation is better that what I could
        # come up with
        self.chi2_estimated = 1 - sts.chi2.cdf(self.chi2_computed, dof)
        return self

    # KMO test - Kaiser, Meyer, Olkin Measure Of Sampling Adequacy
    def kmo(self, threshold=0.5):
        self.kmo = fa.calculate_kmo(self.t)
        if self.kmo[1] < threshold:
            self.message = "There is no any significant factor!"
        return self

    # User the factor_analyzer's built-in analysis
    def analyse(self, rotate=False):
        _rotation = None
        if (rotate):
            _rotation = 'varimax'
        self.model.analyze(self.t, rotation=_rotation)
        self.eigenvalues = self.model.get_eigenvalues()
        return self


