# Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy

This project implements the "Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy", a new TWD for finding  meaningful distances between samples that incorporate the hidden hierarchical structure of the features. This README provides information on prerequisites, usage, and experiments.

## Prerequisites

To use this tool, make sure that you have the following Python packages installed:

- [Numpy](https://numpy.org/install/): For efficient multi-dimensional array operations.
- [SciPy](https://scipy.org/install/): Provides algorithms for optimization, integration, and statistics.
- [NetworkX](https://networkx.org): For the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
- [scikit-learn](https://scikit-learn.org/stable/install.html): For simple and efficient tools for data mining and data analysis.
- [random](https://docs.python.org/3/library/random.html): A built-in Python library for generating pseudo-random numbers. No installation is required.

Optional package for evaluation:
- [Nearest centroid classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html): For classification tasks, part of scikit-learn.


## Usage Example

To use the Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy, you can follow this example:

```python
from hyperbolic_diffusion_distance import *
from diffusion_operator_util import *

# data_matrix is the high-dimensional observation with n samples and m features 
twd_latent_feature = twd_hidden_feature(data_matrix, level_K)

```


## Experiments Overview
This script is designed to facilitate the computation of the Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy between datasets. Before executing any experiments, please make sure you have fulfilled all the prerequisites and installed all necessary Python packages.



### Word-Document Data Analysis

- **Objective**: To demonstrate the applicability of "Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy" in analyzing real-world datasets and compare its performance with traditional methods.
- **Data Sources**:
  - **WMD Dataset**: Accessible at [WMD GitHub repository](https://github.com/mkusner/wmd). 
  - **S-WMD Dataset**: Available at [S-WMD GitHub repository](https://github.com/gaohuang/S-WMD). 

### Single-Cell Gene Expression Data Analysis

- **Objective**: To showcase our TWD's potential in analyzing high-dimensional biological data, particularly in distinguishing cell types based on gene expression profiles.
- **Datasets**:
  - **Zeisel et al. Data**: A comprehensive single-cell RNA sequencing dataset from the mouse cortex and hippocampus. It offers a challenging testbed for cell type classification and analysis.
  - **CBMC Data**: Focuses on single-cell gene expression in human cells, providing a rich dataset for testing the efficacy of "Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy" in biomedical research.
- **Data Access**: Both datasets can be downloaded from the [scGeneFit-python GitHub repository](https://github.com/solevillar/scGeneFit-python/tree/62f88ef0765b3883f592031ca593ec79679a52b4/scGeneFit/data_files).

## Reference

For more detailed information on the datasets and methodologies used in these experiments, please refer to the following references:

1. **Kusner, M. J., Sun, Y., Kolkin, N. I., and Weinberger, K. Q.** (2015).From word embeddings to document distances. In Inter- national Conference on Machine Learning
2. **Huang, G., Guo, C., Kusner, M. J., Sun, Y., Sha, F., and Weinberger, K. Q.** (2016) Supervised word mover's distance. In Advances in Neural Information Processing Systems.
3. **Dumitrascu, B., Villar, S., Mixon, D. G., & Engelhardt, B. E.** (2021). Optimal marker gene selection for cell type discrimination in single cell analyses. *Nature Communications*, 12(1):1–8.
4. **Zeisel, A., et al.** (2015). Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq. *Science*, 347(6226):1138–1142.
5. **Stoeckius, M., et al.** (2017). Simultaneous epitope and transcriptome measurement in single cells. *Nature Methods*, 14(9):865–868.

