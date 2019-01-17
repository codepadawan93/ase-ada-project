import numpy as np
import pandas as pd
import scipy.stats as sts
import pandas.api.types as pdt

'''
    Utility methods go here. All are static
'''

class Utils:

    logfile = "datalysis.log"

    # Make sure someone doesn't create Utils objects
    def __init__(self):
        raise TypeError("Utils is non-instantiatable.")

    @staticmethod
    def log(tag, message, file=logfile):
        file = open(file, "w+")
        file.write(f"${tag}:${message}\r\n")
        file.close()

    @staticmethod
    def putfile(filename, contents):
        file = open(filename, "w+")
        file.write(contents)
        file.close()

    # Replace NA (not available, not applicable) or NaN (not a number) cells
    # with column (variable) mean
    # for a pandas DataFrame
    @staticmethod
    def replace_na(X):
        avgs = np.nanmean(X, axis=0)
        pos = np.where(np.isnan(X))
        print(pos[:])
        X[pos] = avgs[pos[1]]
        return X

    # No offence but let's speak the Queen's
    # Standardise the column (variable) values
    # for a pandas DataFrame
    @staticmethod
    def standardise(X):
        avgs = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        Xstd = (X - avgs) / stds
        return Xstd

    @staticmethod
    def get_data_frame(matrix, index, cols):
        return pd.DataFrame(matrix, index=index, columns=cols)

    @staticmethod
    def invert(t, y=None):
        if type(t) is pd.DataFrame:
            for c in t.columns:
                minim = t[c].min();
                maxim = t[c].max()
                if abs(minim) > abs(maxim):
                    t[c] = -t[c]
                    if y is not None:
                        k = t.columns.get_loc(c)
                        y[:, k] = -y[:, k]
        else:
            for i in range(np.shape(t)[1]):
                minim = np.min(t[:, i]);
                maxim = np.max(t[:, i])
                if np.abs(minim) > np.abs(maxim):
                    t[:, i] = -t[:, i]

    @staticmethod
    # Replace NA by mean/mode
    def replace_na_df(t):
        for c in t.columns:
            if pdt.is_numeric_dtype(t[c]):
                if t[c].isna().any():
                    medie = t[c].mean()
                    t[c] = t[c].fillna(medie)
            else:
                if t[c].isna().any():
                    modul = t[c].mode()
                    t[c] = t[c].fillna(modul[0])

    @staticmethod
    def tabling(X, col_name=None, obs_name=None, table=None):
        X_tab = pd.DataFrame(X)
        if col_name is not None:
            X_tab.columns = col_name
        if obs_name is not None:
            X_tab.index = obs_name
        if table is None:
            X_tab.to_csv("table.csv")
        else:
            X_tab.to_csv(table)
        return X_tab

    @staticmethod
    def partition(h, k):
        n = np.shape(h)[0] + 1
        g = np.arange(0, n)
        for i in range(n - k):
            k1 = h[i, 0]
            k2 = h[i, 1]
            g[g == k1] = n + i
            g[g == k2] = n + i
        clusters = ['c' + str(i) for i in pd.Categorical(g).codes]
        return clusters

    @staticmethod
    def code(t, vars):
        for v in vars:
            t[v] = pd.Categorical(t[v]).codes
