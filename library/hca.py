import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hiclu
import scipy.spatial.distance as spad


class HCA:

    def __init__(self, table):
        self.table = table
        self.h, self.partitions, self.x, self.vars, self.h_v, self.thresholdForOptimal, self.thresholdForPartition7Gr, \
        self.method_obs, self.method_var, self.k = None

    # function to start w a ramndom partitioning
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

        def compute(self):
            self.vars = list(self.t)
            x = self.t[self.vars[3:]].values

            # Observation classification
            method_obs = list(hiclu._LINKAGE_METHODS)
            metric_obs = spad._METRICS_NAMES
            h = hiclu.linkage(x, method=method_obs[0], metric=metric_obs[0])


            # Build partition tables
            partitions = pd.DataFrame(index=table.index)

            # Determine the optimal partition
            m = np.shape(h)[0]
            j = np.argmax(h[1:m, 2] - h[:(m - 1), 2])
            k = m - j
            g_optimal = self.partition(self.h, self.k)
            partitions['Optimal_Partition'] = g_optimal

            threshold = (h[j, 2] + h[j + 1, 2]) / 2

            # Variable classification
            method_var = list(hiclu._LINKAGE_METHODS)
            metric_var = spad._METRICS_NAMES
            h_v = hiclu.linkage(x.transpose(), method=method_var[0], metric=metric_var[0])

            # Partition with 7 groups
            k = 7
            g = self.partition(self.h, self.k)
            partitions['Partition_' + str(k)] = g
            threshold = (h[m - k, 2] + h[m - k + 1, 2]) / 2

            return self.partitions, self.h, self.h_v, self.vars, self.thresholdForPartition7Gr, self.thresholdForOptimal, self.method_obs, self.method_var, self.k






