import numpy as np
from utils.utils import Utils
import pandas as pd
import sklearn.discriminant_analysis as disc
from utils.graphics import Graphics

class LDA:
    def __init__(self, t1, t2, categorical_mark, predictor_mark, predictor_var):
        self.t1 = t1
        self.t2 = t2
        Utils.replace_na_df(self.t1)
        Utils.replace_na_df(self.t2)

        self.var = np.array(t1.columns)
        print(self.var)
        self.var_categorical = self.var[0:categorical_mark]
        print(self.var_categorical)
        Utils.code(self.t1, self.var_categorical)
        Utils.code(self.t2, self.var_categorical)
        print(self.t1)
        print(self.t2)
        # Select the predictor variables and the discriminant variable
        self.var_p = self.var[0:predictor_mark]
        self.var_c = predictor_var
        print(self.var_p)
        print(self.var_c)

        self.x = self.t1[self.var_p].values
        print(self.x)
        self.y = self.t1[self.var_c].values
        print(self.y)

    def fit(self):
        self.lda_model = disc.LinearDiscriminantAnalysis()
        self.lda_model.fit(self.x, self.y)

        self.class_setBase = self.lda_model.predict(self.x)
        self.table_classificationB = pd.DataFrame(
            data={str(self.var_c[0]): self.y, 'prediction': self.class_setBase},
            index=self.t1.index)
        self.tabel_clasificationB_err = self.table_classificationB[self.y != self.class_setBase]
        self.n = len(self.y)
        self.n_err = len(self.tabel_clasificationB_err)
        self.degree_of_credence = (self.n - self.n_err) * 100 / self.n
        return self

    def apply(self):
        self.class_setTest = self.lda_model.predict(self.t2[self.var_p].values)
        table_of_classification = pd.DataFrame(
            data={'prediction': self.class_setTest},
            index=self.t2.index
        )
        self.g = self.lda_model.classes_
        self.q = len(self.g)
        self.mat_c = pd.DataFrame(data=np.zeros((self.q, self.q)), index=self.g, columns=self.g)
        for i in range(self.n):
            self.mat_c.loc[self.y[i], self.class_setBase[i]] += 1
        accuracy_groups = np.diag(self.mat_c) * 100 / np.sum(self.mat_c, axis=1)
        self.mat_c['Accuracy'] = accuracy_groups

        # Instances on the first 2 axes of discrimination
        self.u = self.lda_model.scalings_
        self.z = self.x @ self.u
        self.xc = self.lda_model.means_
        self.zc = self.xc @ self.u
        return self

    def get_results(self):
        return None

    def visualise(self):
        # Get the number of discriminant axes (number of columns in u)
        r = np.shape(self.u)[1]
        if r > 1:
            Graphics.scatter_discriminant(self.z[:, 0], self.z[:, 1], self.y, self.t1.index, self.zc[:, 0], self.zc[:, 1], self.g)
        for i in range(r):
            Graphics.distribution(self.z[:, i], self.y, self.g, axis=i)
        return self

    def show(self):
        Graphics.show()

