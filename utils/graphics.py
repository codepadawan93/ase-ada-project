import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hiclu

class Graphics:
    def __init__(self):
        pass

    @staticmethod
    def correlogram(t, title=None, valmin=-1, valmax=1):
        f = plt.figure(title, figsize=(8, 7))
        f1 = f.add_subplot(1, 1, 1)
        f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
        sb.heatmap(np.round(t, 2), cmap='bwr', vmin=valmin, vmax=valmax, annot=True)

    @staticmethod
    def corrCircle(R, k1, k2, title="The Correlation Circles"):
        plt.figure(title, figsize=(6, 6))
        plt.title(title, fontsize=16, color='b', verticalalignment='bottom')
        T = [t for t in np.arange(0, np.math.pi * 2, 0.01)]
        X = [np.cos(t) for t in T]
        Y = [np.sin(t) for t in T]
        plt.plot(X, Y)
        plt.axhline(0, color='g')
        plt.axvline(0, color='g')
        plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r')
        plt.xlabel(R.columns[k1], fontsize=12, color='r', verticalalignment='top')
        plt.ylabel(R.columns[k2], fontsize=12, color='r', verticalalignment='bottom')
        for i in range(len(R)):
            plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])

    @staticmethod
    def variance(alpha, title='Variance Plot'):
        n = len(alpha)
        f = plt.figure(title, figsize=(10, 7))
        f1 = f.add_subplot(1, 1, 1)
        f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
        f1.set_xticks(np.arange(1, n + 1))
        f1.set_xlabel('Component', fontsize=12, color='r', verticalalignment='top')
        f1.set_ylabel('Variance', fontsize=12, color='r', verticalalignment='bottom')
        f1.plot(np.arange(1, n + 1), alpha, 'ro-')
        f1.axhline(1, c='g')
        j_Kaiser = np.where(alpha < 1)[0][0]
        eps = alpha[:n - 1] - alpha[1:]
        d = eps[:n - 2] - eps[1:]
        j_Cattel = np.where(d < 0)[0][0]
        f1.axhline(alpha[j_Cattel + 1], c='m')
        return j_Cattel + 2, j_Kaiser

    @staticmethod
    def scatter(x, y, label=None, tx="", ty="", title='Scatterplot'):
        f = plt.figure(title, figsize=(10, 7))
        f1 = f.add_subplot(1, 1, 1)
        f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
        f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
        f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
        f1.scatter(x=x, y=y, c='r')
        if label is not None:
            n = len(label)
            for i in range(n):
                f1.text(x[i], y[i], label[i])

    @staticmethod
    def t_scatter(x, y, x1, y1, label=None, label1=None, tx="", ty="", title='Scatterplot - Test Dataset'):
        f = plt.figure(title, figsize=(10, 7))
        f1 = f.add_subplot(1, 1, 1)
        f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
        f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
        f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
        f1.scatter(x=x, y=y, c='r')
        f1.scatter(x=x1, y=y1, c='b')
        if label is not None:
            n = len(label)
            p = len(label1)
            for i in range(n):
                f1.text(x[i], y[i], label[i], color='k')
            for i in range(p):
                f1.text(x1[i], y1[i], label1[i], color='k')

    @staticmethod
    def dendrogram(h, labels, title='Hierarchical classification', threshold=None):
        f = plt.figure(figsize=(12, 7))
        axis = f.add_subplot(1, 1, 1)
        axis.set_title(title, fontsize=16, color='b')
        hiclu.dendrogram(h, labels=labels, leaf_rotation=30, ax=axis, color_threshold=threshold)

    @staticmethod
    def scatter_discriminant(z1, z2, g, labels, zg1, zg2, labels_g):
        f = plt.figure(figsize=(10, 7))
        assert isinstance(f, plt.Figure)
        ax = f.add_subplot(1, 1, 1)
        assert isinstance(ax, plt.Axes)
        ax.set_title("Instances and centers in z1 and z2 axes ", fontsize=14, color='b')
        sb.scatterplot(z1, z2, g)
        sb.scatterplot(zg1, zg2, labels_g, s=200, legend=False)
        for i in range(len(labels)):
            ax.text(z1[i], z2[i], labels[i])
        for i in range(len(labels_g)):
            ax.text(zg1[i], zg2[i], labels_g[i], fontsize=26)

    @staticmethod
    def distribution(z, y, g, axis):
        f = plt.figure(figsize=(10, 7))
        assert isinstance(f, plt.Figure)
        ax = f.add_subplot(1, 1, 1)
        assert isinstance(ax, plt.Axes)
        ax.set_title("Group distribution. Axis " + str(axis + 1), fontsize=14, color='b')
        for v in g:
            sb.kdeplot(data=z[y == v], shade=True, ax=ax, label=v)

    @staticmethod
    def show():
        plt.show()
