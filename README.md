# College project for ADA (Algorithmic data analysis)
## Specifications
A set of classes used for Algorithmic data analysis and visualisation

## Components
### PCA - Principal Components Analysis class
- Used to decompose an initial value matrix into its principal components. Example usage:
```python
# Read input data from csv file using pandas
table = pd.read_csv("./resources/Teritorial.csv", index_col=1)

# Bring the table data into a numpy matrix (ndarray)
X = table.iloc[:, 1:].values

# Instantiate a PCA object
pca = pc.PCA(X)

# Obtain the principal components or intermediate values depending on needs. The following methods are avaliable:
pca.get_correlation()
pca.get_eigenvalues()
pca.get_eigenvectors()
pca.get_correlation_factors()
pca.get_principal_components()
```

## Notes
### Dependencies
- Python 3+
- pandas
- numpy
- matplotlib
- seaborn

## Author
- Kovacs Erik-Robert

## Credits
- Prof. Claudiu Vinte
