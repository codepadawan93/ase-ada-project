# Driver for testing the PCA library based on prof. Vinte's model.
import pandas as pd
from library import pca as pc
from library import datalysis as dl

FILE_NAME = "./resources/Teritorial.csv"
CCA_FILE_NAME = "./resources/Energy.csv"

# Apply PCA: read a file, run all pca tests, visualise results, and then return them
pca_results = dl.Datalysis().read_file(FILE_NAME, index_col=1).run_pca().visualise().get_results()

# parse results: R correlation matrix, a eigenvalues, alpha eigenvectors, rxc correlation factors, C principal components
# R, alpha, a, rxc, C = pca_results

# Apply EFA, get results and show charts
efa_results = dl.Datalysis().read_file(FILE_NAME).run_efa().visualise().get_results();
# print(efa_results)

# Apply CCA
cca_results = dl.Datalysis().read_file(CCA_FILE_NAME, index_col=0).run_cca(4,4).visualise()

# Apply LDA

# Apply HCA

# Apply autorun

# Apply charts
