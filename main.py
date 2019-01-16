# Driver for testing the PCA library based on prof. Vinte's model.
import pandas as pd
from library import pca as pc
from library import datalysis as dl

FILE_NAME = "./resources/Teritorial.csv"

# Apply PCA

# Instantiate a PCA object
# pca = pc.PCA(X)
# R = pca.getCorrelation()
# alpha = pca.getEigenValues()
# a = pca.getEigenVectors()
# Rxc = pca.getCorrelationFactors()
# C = pca.getPrincipalComponents()
# print(R)
# print("Eigenvalues: ", alpha)
# print("Eigenvectors: ", a)
#
# # Save the processed table
# X_Tab = pd.DataFrame(data=X, columns=var_name, index=obs_name)
# X_Tab.to_csv(path_or_buf="X.csv")
#
# # Save the correlation matrix
# R_Tab = pd.DataFrame(data=R, columns=var_name, index=var_name)
# R_Tab.to_csv(path_or_buf="R.csv")
#
# # Show the correlogram
# graphics.correlogram(R_Tab)
#
# # Save the correlation factors
# Rxc_Tab = pd.DataFrame(Rxc, index=var_name, columns=
#                        ["C"+str(k+1) for k in range(m)])
# Rxc_Tab.to_csv("Rxc.csv")
# # print(R_Tab)
#
# # Show factors correlogram
# graphics.correlogram(Rxc_Tab, "Correlogram of the factors")
# graphics.corrCircles(Rxc_Tab, 0, 1)
#
# # Show the eigenvalues graphic
# graphics.eighenValues(alpha)

pca_results = dl.Datalysis() \
    .read_file(FILE_NAME, 0) \
    .run_pca() \
    .results
# R correlation matrix, a eigenvalues, alpha eigenvectors, rxc correlation factors, C principal components
R, alpha, a, rxc, C = pca_results.R, pca_results.alpha, pca_results.a, pca_results.R_x_c, pca_results.C

# # Apply EFA
# efa_analyser = dl.Datalysis();
# efa_results = efa_analyser.read_file("./resources/Teritorial.csv").run_efa().results;
#
# S, q, beta, common = efa_results.S, efa_results.q, efa_results.beta, efa_results.common
# print("S: ", S)
# print("q: ", q)
# print("beta: ", beta)
# print("common: ", common)
# bartlett_test = efa_results.bartlett_test_results
# print("Bartlett Test:", bartlett_test)

#
# # Evaluate the information redundancy
# # Bartlett test
# Bartlett_test = fa.calculate_bartlett_sphericity(t)
# print("Bartlett Test:", Bartlett_test)
#
# # KMO test - Kaiser, Meyer, Olkin Measure Of Sampling Adequacy
# kmo = fa.calculate_kmo(t)
# print("Kaiser, Meyer, Olkin measure of sampling adequacy: ", kmo)
# vi.correlogram(kmo[0], " KMO Indices")
# print("KMO Total:", kmo[1])
#
# if kmo[1] < 0.5:
#     print("There is no any significant factor!")
#     exit(1)
#
# # Compose the model
# fa_model = fa.FactorAnalyzer()
# fa_model.analyze(t, rotation=None)
#
# # Extract the factorial coefficients
# loads = fa_model.loadings
# loads.to_csv("Fa_Loadings.csv")
#
# vi.correlogram(loads, "Factorial Coefficients")
# vi.corrCircle(loads, 0, 1, "Factorial Coefficients - 1,2")
# vi.corrCircle(loads, 0, 2, "Factorial Coefficients - 1,3")
#
# # Apply factorial rotation
# fa_model.analyze(t, rotation='varimax')
# load_rot = fa_model.loadings
# load_rot.to_csv("Fa_loadings_varimax.csv")
# vi.correlogram(load_rot, "Factorial Coefficients - Varimax")
# vi.corrCircle(load_rot, 0, 1, "Factorial Coefficients (Varimax) - 1,2")
# vi.corrCircle(load_rot, 0, 2, "Factorial Coefficients (Varimax)- 1,3")
#
# # Eigenvalues
# eigvalues = fa_model.get_eigenvalues()
# print(eigvalues)
# Apply CCA

# Apply LDA

# Apply HCA

# Apply autorun

# Apply charts
