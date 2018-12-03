# Driver for testing the PCA library based on prof. Vinte's model.
import pandas as pd
from library import pca as pc
from library import datalysis as dl

# Read input data from csv file
table = pd.read_csv("./resources/Teritorial.csv", index_col=1)

# Bring the table data into a numpy matrix (ndarray)
X = table.iloc[:, 1:].values

obs_name = table.index
var_name = table.columns[1:]

n = X.shape[0] # number of observations
m = X.shape[1] # number of variables
print(n, m)
print(X)

# Instantiate a PCA object
pca = pc.PCA(X)
R = pca.get_correlation()
alpha = pca.get_eigenvalues()
a = pca.get_eigenvectors()
Rxc = pca.get_correlation_factors()
C = pca.get_principal_components()
print(R)
print("Eigenvalues: ", alpha)
print("Eigenvectors: ", a)

# Test datalysis class...
analyser = dl.Datalysis()
analyser.read_file("./resources/Teritorial.csv", 1)


