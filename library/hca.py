import numpy as np
import pandas as pd
from utils.utils import Utils
import scipy.cluster.hierarchy as hiclu
import scipy.spatial.distance as spad
from utils.graphics import Graphics

class HCA:

    def __init__(self, table):
        self.table = table
        Utils.replace_na_df(self.table)
        self.method_obs = self.method_var = list(hiclu._LINKAGE_METHODS)
        self.metric_obs = self.metric_var = spad._METRICS_NAMES
        # Build partition tables
        self.partitions = pd.DataFrame(index=self.table.index)
        self.threshold_optimal = self.threshold_arbitrary = None


    def classify(self, method=0, metric=0):
        self.chosen_obs_method = method
        self.chosen_obs_metric = metric

        self.vars = list(self.table)
        self.x = self.table[self.vars[3:]].values

        # Observation classification
        self.h = hiclu.linkage(self.x, method=self.method_obs[method], metric=self.metric_obs[metric])
        self.m = np.shape(self.h)[0]

        # Variable classification
        self.chosen_var_method = method
        self.chosen_var_metric = metric

        self.h_v = hiclu.linkage(self.x.transpose(), method=self.method_var[method], metric=self.metric_var[metric])
        return self

    def optimal_partition(self):
        # Determine the optimal partition
        self.j = np.argmax(self.h[1:self.m, 2] - self.h[:(self.m - 1), 2])
        self.k = self.m - self.j

        self.g_optimal = Utils.partition(self.h, self.k)
        self.partitions['Optimal_Partition'] = self.g_optimal
        self.threshold_optimal = (self.h[self.j, 2] + self.h[self.j + 1, 2]) / 2
        return self

    def arbitrary_partition(self, k):
        self.k = k
        self.g = Utils.partition(self.h, self.k)
        self.partitions['Partition_' + str(self.k)] = self.g
        self.threshold_arbitrary = (self.h[self.m - self.k, 2] + self.h[self.m - self.k, 2]) / 2
        return self

    def get_results(self):
        if(self.threshold_optimal is not None):
            return self.partitions, self.h, self.h_v, self.vars, self.threshold_optimal, self.method_obs, self.method_var, self.k
        else :
            return self.partitions, self.h, self.h_v, self.vars, self.threshold_arbitrary, self.method_obs, self.method_var, self.k

    def visualise(self):
        threshold = None
        if (self.threshold_optimal is not None):
            threshold = self.threshold_optimal
        else :
            threshold = self.threshold_arbitrary

        Graphics.dendrogram(self.h, self.table.index,
            "Observation Classifications | Method:" + self.method_obs[
                self.chosen_obs_method] + " | Metric:" + self.metric_obs[self.chosen_obs_metric],
            threshold=threshold)
        return self

    def show(self):
        Graphics.show()