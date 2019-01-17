# Driver for testing the PCA library based on prof. Vinte's model.
from library import datalysis as dl

FILE_NAME = "./resources/Teritorial.csv"
CCA_FILE_NAME = "./resources/Energy.csv"
LDA_FILE_1 = "./resources/ProiectB.csv";
LDA_FILE_2 = "./resources/ProiectBEstimare.csv";

# Apply PCA: read a file, run all pca tests, visualise results, and then return them
pca_results = dl.Datalysis()\
    .read_file(FILE_NAME, index_col=1)\
    .run_pca()\
    .visualise()\
    .get_results()

# parse results: R correlation matrix, a eigenvalues, alpha eigenvectors, rxc correlation factors, C principal components
# R, alpha, a, rxc, C = pca_results

# Apply EFA, get results and show charts
efa_results = dl.Datalysis()\
    .read_file(FILE_NAME)\
    .run_efa()\
    .visualise()\
    .get_results()

# print(efa_results)

# Apply CCA
cca_results = dl.Datalysis()\
    .read_file(CCA_FILE_NAME, index_col=0) \
    .run_cca(x_mark=4, y_mark=4) \
    .visualise()\
    .get_results()

# Apply LDA
lda_results = dl.Datalysis()\
    .read_multiple(LDA_FILE_1, LDA_FILE_2, f1_index_col=0, f2_index_col=0)\
    .run_lda(categorical_mark=6, predictor_mark=11, predictor_var="VULNERAB")\
    .visualise()\
    .get_results()

# Apply HCA
hca_results = dl.Datalysis()\
    .read_file(FILE_NAME, index_col=0)\
    .run_hca(method=2, metric=1, partition_type="arbitrary", no_groups=1)\
    .visualise()\
    .get_results()
hca_results2 = dl.Datalysis()\
    .read_file(FILE_NAME, index_col=0)\
    .run_hca(method=1, metric=1, partition_type="optimal")\
    .visualise()\
    .get_results()

# Apply save report
# not implemented yet