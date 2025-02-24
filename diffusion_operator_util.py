import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
from scipy import linalg

TOLERANCE = 1e-6

def compute_diffusion_operator_normalized(data, distance_matrix=None, affinity=None, sigma=2, num_eigenvalues=20, full_spectrum=False):
    """
    Compute normalized diffusion operator for data.
    Code modified from Lin, Y.-W. E., Coifman, R. R., Mishne, G., and Talmon, R. 
    Hyperbolic diffusion embedding and distance for hierarchical representation learning. 
    In International Conference on Machine Learning, pp. 21003–21025. PMLR, 2023.
    """
    if distance_matrix is None:
        distance_matrix = pairwise_distances(data)

    Gaussian_kernel = affinity if affinity is not None else np.exp(-distance_matrix / sigma)

    degree_matrix_inv = np.diag(1 / np.sum(Gaussian_kernel, axis=1))
    normalized_kernel = degree_matrix_inv @ Gaussian_kernel @ degree_matrix_inv

    sqrt_degree_matrix_inv = np.diag(1 / np.sqrt(np.sum(normalized_kernel, axis=1)))
    sqrt_degree_matrix = np.diag(np.sqrt(np.sum(normalized_kernel, axis=1)))
    markov_matrix = sqrt_degree_matrix_inv @ normalized_kernel @ sqrt_degree_matrix_inv
    
    if full_spectrum:
        eigenvalues, eigenvectors = linalg.eig(markov_matrix)
    else:
        eigenvalues, eigenvectors = linalg.eigsh(markov_matrix, k=num_eigenvalues, which='LM')

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    eigenvalues = np.maximum(eigenvalues, TOLERANCE)

    left_eigenvectors = sqrt_degree_matrix_inv @ eigenvectors
    right_eigenvectors = eigenvectors.T @ sqrt_degree_matrix

    # print('Eigenvalues: {}'.format(eigenvalues))
    # print('right_eigenvectors: {}'.format(right_eigenvectors))
    # print('left_eigenvectors: {}'.format(left_eigenvectors))

    return eigenvalues, left_eigenvectors, right_eigenvectors
