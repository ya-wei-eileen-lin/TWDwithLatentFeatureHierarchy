import numpy as np
import random
from hyperbolic_diffusion_utils import *

np.random.seed(42)
random.seed(42)
k_c = 19  # Number of iterations or hyperbolic distances to compute

if __name__ == '__main__':
    # Load word document and single-cell RNA sequencing (scRNA-seq) data from files
    word_doc = np.load('data/word_doc.npy')
    scrna_data = np.load('data/scrna_data.npy')

    # Initialize lists to store hyperbolic diffusion tree-width (HDTW) distances
    hdtw_doc = []
    hdtw_cell = []

    # Compute Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy for each k in k_c
    for k in range(k_c):

        # Compute HDTW for word document data
        for word in range(len(word_doc)):
            tmp_word = twd_hidden_feature(word_doc[word], k + 1)
            hdtw_doc.append(tmp_word)

        # Compute HDTW for scRNA-seq data
        for cell in range(len(scrna_data)):
            tmp_cell = twd_hidden_feature(scrna_data[cell], k + 1)
            hdtw_cell.append(tmp_cell)
