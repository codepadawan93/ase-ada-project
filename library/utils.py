import numpy as np

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
    def replace_NA(X):
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


