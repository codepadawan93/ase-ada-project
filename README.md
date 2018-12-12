# College project for ADA (Algorithmic data analysis)
## Specifications
A set of classes used for Algorithmic data analysis and visualisation

## Requirements
- Build a class hierarchy for implementing all the 6 analysis studied:
  - PCA
  - Factor Analysis
  - Canonical Correlation 
  - Discriminant Analysis
  - Cluster Analysis 
  - Correspondence Analysis
- I asked him to publish and how he would evaluate the project
- Prof. said he would take into account the interface, its ease of use
- The results must be easy to access, and it must be capable of rendering a visual representation of the output
- In short, we need to make a library for others focused on ease of use

## Datalysis
### Installation
Clone this project, create a new virtual env, install dependencies and start yours around it:
```bash
git clone https://github.com/codepadawan93/ase-ada-project.git .
pip install requirements.txt

code main.py 
```

### Easy to use API for running all available methods
Conclusions can be output to a file:
```python
from library import datalysis as dat

analyser = dat.Datalysis()
analyser \ 
    .read_file('./resources/CoolData.csv', index_col=1) \
    .run_all() \
    .put_report('./output/MyReport.txt')
```
or visualised directly at runtime :
```python
analyser \ 
    .read_file('./resources/CoolData.csv', index_col=1) \
    .run_all() \
    .visualise()
```

### PCA - Principal Components Analysis class
Used to decompose an initial value matrix into its principal components. Example usage:
```python
results = analyser \ 
    .read_file('./resources/CoolData.csv', index_col=1) \
    .run_pca() \
    .results
```

### EFA - Exploratory Factor Analysis class
Used to find underlying factors in the data that were not measured directly. Example usage:
```python
results = analyser \ 
    .read_file('./resources/CoolData.csv', index_col=1) \
    .run_efa() \
    .results
```

### CCA - Canonical Correlation Analysis class
- Not yet implemented

## Notes
### Dependencies
- Python 3.7+
- numpy
- scipy
- pandas
- factor-analyzer
- sklearn
- matplotlib
- seaborn

## Author
- Kovacs Erik-Robert

## Credits
- Prof. Claudiu Vinte
